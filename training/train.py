"""
Self-supervised training of the embedding model.

Training pairs are generated from unlabelled stacks: for a voxel on the
foreground (Otsu mask), a crop around it and the same crop rotated in 3D around
its centre form a positive pair; both are independently augmented with noise and
blur. The model learns to map the two to the same
embedding (see ``models.py``).

Evaluation uses a separate benchmark stack: crops at every ``offset``-th
foreground voxel are matched against the rotated versions of all of them, and
the script reports how often the best match is the right one and how far off
it is otherwise.

The crops are written to memory-mapped files in ``--cache-dir`` first, so the
training itself does not hold them in RAM.

Usage
-----
::

    python train.py --benchmark data/DeepD3_Benchmark.tif \\
        --training-dir data/DeepD3_Training --output-dir results

Outputs in ``--output-dir`` (``<tag>`` is ``<name>_<crop xy>_<crop z>``):

* ``spine_embedder_ssl_<tag>.pth`` -- model weights, for ``inference/``
* ``loss_<tag>.npy`` -- mean training loss per epoch
* ``distances_<tag>.npy`` -- per benchmark point, the distance (nm) between
  the point and its best match
"""

import argparse
import os

import cv2
import numpy as np
import tifffile as tif
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform, gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

import models

# training crops drawn at random from the foreground of each training stack
POINTS_PER_STACK = 3000


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------

def normalize_stack(stack):
    """Normalize a 3D stack to the range 0-255 (uint8)."""
    return cv2.normalize(stack, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def create_mask(stack):
    """Binary foreground mask (0/255) of a 3D stack, Otsu threshold per slice."""
    mask = np.zeros_like(stack, dtype=np.uint8)
    for i in range(stack.shape[0]):
        mask[i] = cv2.threshold(stack[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return mask


# ----------------------------------------------------------------------------
# crops and augmentation
# ----------------------------------------------------------------------------

def rotation_matrix_3d(ax, ay, az):
    """Rotation matrix for (z, y, x) coordinates, angles in degrees.

    ``ax`` rotates in the z-y plane, ``ay`` in the z-x plane (both tilt the
    stack), and ``az`` in the y-x plane, i.e. in-plane about the optical axis.
    Applied in the order ax, ay, az.
    """
    ax, ay, az = np.deg2rad([ax, ay, az])

    sx, sy, sz = np.sin([ax, ay, az])
    cx, cy, cz = np.cos([ax, ay, az])

    Rx = np.array([[ cx, -sx, 0],
                   [ sx,  cx, 0],
                   [  0,   0, 1]], dtype=np.float32)

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)

    Rz = np.array([[1,  0,  0],
                   [0, cz, -sz],
                   [0, sz,  cz]], dtype=np.float32)

    return Rz @ Ry @ Rx


def rotate_3d_around_point(stack, point, pad_val, ax=None, ay=None, az=None, order=1):
    """Rotate a stack (Z, Y, X) by random angles around ``point`` (z, y, x).

    Each angle is drawn uniformly from the integers in ``[-a, a]`` for the
    given bound ``a`` in degrees. Defaults: 3 degrees for the two tilts and
    the full circle (-180 to 179) in-plane. Voxels rotated in from outside
    the stack are set to ``pad_val``.
    """
    if ax is None:
        ax = np.random.randint(-3, 4)
    else:
        ax = np.random.randint(-ax, ax + 1)

    if ay is None:
        ay = np.random.randint(-3, 4)
    else:
        ay = np.random.randint(-ay, ay + 1)

    if az is None:
        az = np.random.randint(-180, 180)
    else:
        az = np.random.randint(-az, az + 1)

    R = rotation_matrix_3d(ax, ay, az)
    point = np.asarray(point, dtype=float)

    # offset that makes the rotation happen around `point`
    offset = point - R @ point

    rotated = affine_transform(
        stack,
        R,
        offset=offset,
        order=order,
        mode="constant",
        cval=pad_val,
        prefilter=(order > 1),
    )

    return rotated


def crop_roi(image, point, size_x, size_y, size_z):
    """Crop one 3D window out of ``image`` (Z, Y, X), centred on ``point``.

    ``point`` (z, y, x) lands at index ``size // 2`` on each axis. Returns an
    array (size_z, size_y, size_x); where the window extends past the image it
    is filled by reflection.
    """
    z_dim, y_dim, x_dim = image.shape
    zc, yc, xc = map(int, point)

    # start/end on each axis such that (end - start) == size
    sx = xc - (size_x // 2)
    ex = sx + size_x
    sy = yc - (size_y // 2)
    ey = sy + size_y
    sz = zc - (size_z // 2)
    ez = sz + size_z

    cropped = image[max(sz, 0):min(ez, z_dim),
                    max(sy, 0):min(ey, y_dim),
                    max(sx, 0):min(ex, x_dim)]

    if cropped.shape == (size_z, size_y, size_x):
        return cropped

    # the window hangs over a border: reflect-pad the missing part
    pad_z = (max(0, -sz), max(0, ez - z_dim))
    pad_y = (max(0, -sy), max(0, ey - y_dim))
    pad_x = (max(0, -sx), max(0, ex - x_dim))
    return np.pad(cropped, (pad_z, pad_y, pad_x), mode="reflect")


def augment_crop(crop, amount=0.1, gauss_sigma=10, blur_sigma=(0.2, 1.0, 1.0)):
    """Random augmentation of a crop with values in 0-255.

    Each of the following is applied with probability 0.5: salt-and-pepper
    noise on a fraction ``amount`` of the voxels, Gaussian noise with standard
    deviation ``gauss_sigma``, and Gaussian blur with ``blur_sigma`` (z, y, x). 
    The result is clipped to 0-255 and keeps the dtype of ``crop``.
    """
    crop_noisy = crop.astype(np.float32).copy()

    if np.random.rand() < 0.5:
        # salt and pepper noise
        rand = np.random.rand(*crop.shape)
        crop_noisy[rand < amount/2] = crop.min()
        crop_noisy[rand > 1 - amount/2] = crop.max()

    if np.random.rand() < 0.5:
        # gaussian noise
        noise = np.random.normal(0, gauss_sigma, crop.shape)
        crop_noisy += noise

    if np.random.rand() < 0.5:
        # gaussian blur
        crop_noisy = gaussian_filter(crop_noisy, sigma=blur_sigma)

    return np.clip(crop_noisy, 0, 255).astype(crop.dtype)


# ----------------------------------------------------------------------------
# training and test data
# ----------------------------------------------------------------------------

def create_memmap_data(train_stacks, train_masks, stack_norm, mask, offset=10,
                       size_x=128, size_y=128, size_z=7, cache_dir="cache",
                       points_per_stack=POINTS_PER_STACK):
    """Write augmented crop pairs to memory-mapped float32 files.

    Training: ``points_per_stack`` random foreground voxels from each training
    stack (all of them if there are fewer; the unused rows then stay zero).
    Test: every ``offset``-th foreground voxel of the benchmark stack.

    For each voxel the files hold an augmented crop (``stat_*``) and an
    augmented copy rotated around the crop centre (``rot_*``), each of shape
    (1, size_z, size_y, size_x).

    Parameters
    ----------
    train_stacks, train_masks : list of ndarray (Z, Y, X)
        Normalised training stacks and their foreground masks.
    stack_norm, mask : ndarray (Z, Y, X)
        Normalised benchmark stack and its foreground mask.
    offset : int
        Take every ``offset``-th foreground voxel of the benchmark.
    size_x, size_y, size_z : int
        Crop size.
    cache_dir : str
        The files go to ``<cache_dir>/<size_x>_<size_z>/``.
    points_per_stack : int
        Training crops per training stack.

    Returns
    -------
    points : ndarray (N, 3)
        The benchmark voxels (z, y, x), in the order of the test files.
    path : str
        Folder of the files.
    """
    path = os.path.join(cache_dir, f"{size_x}_{size_z}")
    os.makedirs(path, exist_ok=True)

    points = np.argwhere(mask > 0)[::offset]
    print("Number of points in mask:", points.shape)
    N = len(points)
    shape_test = (N, 1, size_z, size_y, size_x)
    shape_train = (len(train_masks) * points_per_stack, 1, size_z, size_y, size_x)

    stat_mm = np.memmap(os.path.join(path, "stat_aug_train.dat"),
                        dtype="float32", mode="w+", shape=shape_train)
    rot_mm = np.memmap(os.path.join(path, "rot_aug_train.dat"),
                       dtype="float32", mode="w+", shape=shape_train)
    stat_test_mm = np.memmap(os.path.join(path, "stat_aug_test.dat"),
                             dtype="float32", mode="w+", shape=shape_test)
    rot_test_mm = np.memmap(os.path.join(path, "rot_aug_test.dat"),
                            dtype="float32", mode="w+", shape=shape_test)

    # random foreground voxels of each training stack
    points_train = []
    for train_mask in train_masks:
        train_points = np.argwhere(train_mask > 0)
        if len(train_points) > points_per_stack:
            selected_indices = np.random.choice(len(train_points), size=points_per_stack, replace=False)
            train_points = train_points[selected_indices]
        points_train.append(train_points)
    print(len(points_train), points_train[0].shape)

    # the crop centre, around which the copies are rotated
    px, py, pz = int(size_x/2), int(size_y/2), int(size_z/2)

    # training data
    for i, points_stack in tqdm(enumerate(points_train), total=len(points_train)):
        for j, p in enumerate(points_stack):
            crop_stat = crop_roi(train_stacks[i], p, size_x=size_x, size_y=size_y, size_z=size_z)
            crop_stat_aug = augment_crop(crop_stat)
            crop_rot = rotate_3d_around_point(crop_stat, (pz, py, px), pad_val=float(train_stacks[i].min()), ax=3, ay=3, az=180)
            crop_rot_aug = augment_crop(crop_rot)

            idx = i * points_per_stack + j
            stat_mm[idx, 0] = crop_stat_aug
            rot_mm[idx, 0] = crop_rot_aug

    # test data
    min_val = float(stack_norm.min())
    for idx, p in tqdm(enumerate(points), total=N):
        crop_stat = crop_roi(stack_norm, p, size_x=size_x, size_y=size_y, size_z=size_z)
        crop_stat_aug = augment_crop(crop_stat)
        crop_rot = rotate_3d_around_point(crop_stat, (pz, py, px), pad_val=min_val, ax=3, ay=3, az=180)
        crop_rot_aug = augment_crop(crop_rot)

        stat_test_mm[idx, 0] = crop_stat_aug
        rot_test_mm[idx, 0] = crop_rot_aug

    stat_mm.flush()
    rot_mm.flush()
    stat_test_mm.flush()
    rot_test_mm.flush()

    return points, path


def load_memmap_pairs(path, split, n, size_xy, size_z):
    """The (crop, rotated crop) tensors of one split ("train" or "test"),
    backed by the files of ``create_memmap_data``."""
    def load(kind):
        return torch.from_file(
            os.path.join(path, f"{kind}_aug_{split}.dat"),
            dtype=torch.float32,
            size=n * 1 * size_z * size_xy * size_xy,
        ).view(n, 1, size_z, size_xy, size_xy)
    return load("stat"), load("rot")


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def get_embeddings_loader(model, loader, device):
    """Embeddings of the unrotated crops of a (crop, rotated crop) loader."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for stat, _ in loader:
            stat = stat.to(device, non_blocking=True)
            emb = model(stat)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def get_rot_embeddings_loader(model, loader, device):
    """Embeddings of the rotated crops of a (crop, rotated crop) loader."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for _, rot_test in loader:
            rot_test = rot_test.to(device, non_blocking=True)
            emb = model(rot_test)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def match_embeddings(emb1, emb2, chunk_size=64):
    """Best match in ``emb2`` for every embedding in ``emb1``, by cosine
    similarity, without building the full [N1, N2] similarity matrix.

    Accepts [N, D] or [N, T, D]; with a slice dimension the similarity is
    averaged over all slice pairs.

    Returns
    -------
    best_matches : LongTensor [N1]
        Index into ``emb2``.
    best_scores : Tensor [N1]
        Similarity of that match.
    """
    if emb1.dim() == 2:
        emb1 = emb1.unsqueeze(1)
        emb2 = emb2.unsqueeze(1)

    N1, T1, D = emb1.shape
    N2, T2, D = emb2.shape

    emb1 = F.normalize(emb1, dim=2)
    emb2 = F.normalize(emb2, dim=2)

    best_matches = torch.zeros(N1, dtype=torch.long)
    best_scores = torch.full((N1,), -1e9)

    for j in range(0, N2, chunk_size):
        emb2_chunk = emb2[j:j+chunk_size]  # [C, T2, D]
        C = emb2_chunk.size(0)

        sim_chunk = torch.zeros(N1, C, device=emb1.device)

        for t2 in range(T2):
            e2 = emb2_chunk[:, t2, :]  # [C, D]

            partial = torch.einsum('ntd,cd->ntc', emb1, e2)
            partial = partial.mean(dim=1)  # [N1, C]

            sim_chunk += partial

        sim_chunk /= T2  # [N1, C]

        # update best matches
        chunk_best_scores, chunk_best_idx = sim_chunk.max(dim=1)

        better = chunk_best_scores > best_scores
        best_scores[better] = chunk_best_scores[better]
        best_matches[better] = chunk_best_idx[better] + j

        # free memory explicitly
        del sim_chunk, partial
        torch.cuda.empty_cache()

    return best_matches, best_scores


def evaluate(model, eval_loader, points_test, voxel_size, device):
    """Match every benchmark crop against all rotated crops.

    Prints the fraction of crops whose best match is their own rotated copy
    and the 95th percentile of the distance to the best match. Returns the
    distances, in the units of ``voxel_size``.
    """
    model.eval()
    emb_stat = get_embeddings_loader(model, eval_loader, device)
    emb_rot_test = get_rot_embeddings_loader(model, eval_loader, device)
    emb_stat = emb_stat.cpu()
    emb_rot_test = emb_rot_test.cpu()

    matches, similarity = match_embeddings(emb_stat, emb_rot_test)

    correct = 0
    for i in range(len(matches)):
        if matches[i] == i:
            correct += 1

    print(f"Correct matches: {correct}/{len(matches)} = {correct/len(matches)*100:.1f}%")

    # distance between each point and the point it was matched to
    distances = []
    voxel_size = np.asarray(voxel_size)
    for i in range(len(matches)):
        matched_idx = matches[i].item()
        dist = np.linalg.norm((points_test[i] - points_test[matched_idx]) * voxel_size)
        distances.append(dist)

    percentile95 = np.percentile(distances, 95)
    print(f"95th percentile distance between matched points: {percentile95:.2f} nm")
    return distances


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def load_training_stacks(folder, channel=0):
    """Normalised stacks and foreground masks of every .tif in ``folder``.

    The training stacks have the axes (Z, C, Y, X); ``channel`` is used.
    """
    training_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]

    train_stacks = []
    train_masks = []
    for p in training_paths:
        train_stack = tif.imread(p)
        train_stack_norm = normalize_stack(train_stack[:, channel, :, :])
        train_masks.append(create_mask(train_stack_norm))
        train_stacks.append(train_stack_norm)
    return train_stacks, train_masks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Self-supervised training of the embedding model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--benchmark", required=True,
                       help="stack (Z, Y, X) for the evaluation, e.g. DeepD3_Benchmark.tif")
    paths.add_argument("--training-dir", required=True,
                       help="folder with the training stacks (.tif, axes Z, C, Y, X)")
    paths.add_argument("--output-dir", default="results",
                       help="where the model, the loss and the distances are saved")
    paths.add_argument("--cache-dir", default="cache",
                       help="where the crops are written as memory-mapped files "
                            "(needs a lot of space)")
    paths.add_argument("--dinov3-repo", default=models.DEFAULT_DINOV3_REPO,
                       help="local clone of the DINOv3 repository "
                            "(default can also be set with DINOV3_REPO)")

    data = parser.add_argument_group("data")
    data.add_argument("--crop-size-xy", type=int, default=128, help="crop width and height")
    data.add_argument("--crop-size-z", type=int, default=7, help="crop depth in slices")
    data.add_argument("--offset", type=int, default=5,
                      help="use every n-th foreground voxel of the benchmark")
    data.add_argument("--points-per-stack", type=int, default=POINTS_PER_STACK,
                      help="training crops per training stack")
    data.add_argument("--voxel-size", type=float, nargs=3, default=(500.0, 94.0, 94.0),
                      metavar=("Z", "Y", "X"),
                      help="voxel size of the benchmark in nm, for the distances")

    train = parser.add_argument_group("training")
    train.add_argument("--backbone", default="dinov3_vitb16", help="DINOv3 hub model")
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42,
                       help="NumPy seed for point sampling and augmentation")
    train.add_argument("--name", default=None,
                       help="name used in the output files (default: dinov3_seed<seed>)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    name = args.name or f"dinov3_seed{args.seed}"
    size_x = size_y = args.crop_size_xy
    size_z = args.crop_size_z
    offset = args.offset

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())

    np.random.seed(args.seed)

    # benchmark stack, for the evaluation
    stack = tif.imread(args.benchmark)
    print("Original stack shape:", stack.shape)
    stack_norm = normalize_stack(stack)
    print("Normalized stack shape:", stack_norm.shape)
    mask = create_mask(stack_norm)
    print("Mask shape: ", mask.shape)

    # training stacks
    train_stacks, train_masks = load_training_stacks(args.training_dir)

    save_str = f'{size_x}_{size_z}'
    print(save_str)
    points_test, path = create_memmap_data(
        train_stacks, train_masks, stack_norm, mask, offset=offset,
        size_x=size_x, size_y=size_y, size_z=size_z, cache_dir=args.cache_dir,
        points_per_stack=args.points_per_stack)

    n_train = len(train_masks) * args.points_per_stack
    stat, rot = load_memmap_pairs(path, "train", n_train, size_x, size_z)
    stat_test, rot_test = load_memmap_pairs(path, "test", len(points_test), size_x, size_z)

    train_dataset = torch.utils.data.TensorDataset(stat, rot)
    eval_dataset = torch.utils.data.TensorDataset(stat_test, rot_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.DINOv3Encoder(embedding_dim=args.embedding_dim, model=args.backbone,
                                 repo_dir=args.dinov3_repo).to(device)

    # training
    loss = models.train(model=model, loader=train_loader, lr=args.lr, device=device,
                        epochs=args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{name}_{save_str}"
    np.save(os.path.join(args.output_dir, f"loss_{tag}.npy"), np.asarray(loss))
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"spine_embedder_ssl_{tag}.pth"))

    # evaluation on the benchmark
    distances = evaluate(model, eval_loader, points_test, args.voxel_size, device)
    np.save(os.path.join(args.output_dir, f"distances_{tag}.npy"), np.asarray(distances))


if __name__ == "__main__":
    main()
