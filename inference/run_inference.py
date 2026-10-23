"""
Transfer 3D landmarks from every timepoint of a stack to the next one.

Input is a 4D TIFF (T, Z, Y, X) and a CSV with the columns TRACK_ID,
POSITION_X, POSITION_Y, POSITION_Z, POSITION_T. Two modes:

``--mode track``
    Start from the landmarks of one timepoint (by default the first one in the
    CSV), predict where they are at the next timepoint, use the predictions as
    the landmarks for the step after, and so on to the last timepoint -- the
    same as the GUI does. Only the starting timepoint has to be labelled.
    With ``--labels-every-timepoint`` the query points of each step are read
    from the CSV instead, wherever it has labels for that timepoint.
``--mode evaluate``
    For every transition ``t -> t + 1``, transfer the landmarks labelled at
    both timepoints and compare the predictions with the labels at ``t + 1``.

Each transfer has three steps:

1. coarse x/y from LightGlue matches on the maximum-intensity projections
2. coarse z from plane-by-plane LightGlue matching
3. refinement against per-landmark similarity maps of learned embeddings,
   with a term that preserves the geometry of the point set (``optimize.py``)

Usage
-----
Track the landmarks of one stack::

    python run_inference.py --mode track --stacks data/stack.tif \\
        --labels data/stack.csv --model checkpoints/spine_embedder.pth

Evaluate against the labels::

    python run_inference.py --stacks data/stack.tif --labels data/stack.csv \\
        --model checkpoints/spine_embedder.pth

Every stack in a folder that has a CSV of the same name::

    python run_inference.py --stacks data/tiffs --labels data/csvs \\
        --model checkpoints/spine_embedder.pth --output results/my_dataset

Results are written to ``<output>/<stack name>/``; see README.md.
"""

import os

# PyTorch and OpenCV/NumPy can each bring their own OpenMP runtime on Windows,
# which aborts the interpreter on import. Must be set before they are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

import optimize
from utils import (
    DEFAULT_DINOV3_REPO,
    DINOv3Encoder,
    LazyEmbeddings,
    compute_embeddings_batch,
    create_result_csv,
    crop_around_point,
    plot_point_transfer,
    roi_bounds,
    roi_coords,
    transfer_points,
    visualize_matching,
    visualize_row,
    z_registration,
)

# ----------------------------------------------------------------------------
# fixed settings
# ----------------------------------------------------------------------------

# embedding model -- must match the checkpoint
EMBEDDING_DIM = 128
BACKBONE = "dinov3_vitb16"
EMBEDDING_BATCH_SIZE = 64

# coarse registration
MAX_KEYPOINTS = 2048        # SuperPoint keypoints per image
TRANSFER_NEIGHBOURS = 20    # matched features interpolated per landmark
PLOT_EVERY_NTH_MATCH = 5    # thin out the matches drawn in the overview

# refinement, see optimize.optimize_point_cloud
OPTIMIZER_SETTINGS = dict(
    lambda_sim=1.0,
    lambda_shape=0.2,
    lambda_anchor=0.3,
    max_delta=20,
    use_all_pairs=True,
)

LABEL_COLUMNS = ["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T"]
STACK_SUFFIXES = (".tif", ".tiff")
DONE_MARKER_NAME = ".done"


@dataclass(frozen=True)
class Settings:
    """Everything a run needs besides the stack and its labels."""

    path_model: str                  # checkpoint of the embedding model
    dinov3_repo: str                 # local clone of the DINOv3 repository
    crop_size_z: int = 3             # z-slices per embedding crop
    roi_size_xy: int = 32            # x/y size of the similarity ROI, pixels
    roi_size_z: int = 9              # z size of the similarity ROI, slices
    voxel_size_xy: float = 1.0       # label units per pixel in x/y
    voxel_size_z: float = 1.0        # label units per slice
    mode: str = "evaluate"           # "evaluate" or "track"
    start_timepoint: int = None      # track: first timepoint; None = first in the CSV
    labels_every_timepoint: bool = False   # track: query points from the CSV at every step


# ----------------------------------------------------------------------------
# pipeline steps
# ----------------------------------------------------------------------------

def load_query_points(labels, t, scaling_um_to_px_xy, scaling_um_to_px_z):
    """Landmarks at timepoint ``t`` that are also labelled at ``t + 1``.

    Returns an integer array (N, 3) of (z, y, x) pixel coordinates, in the
    order of the rows of ``labels``.
    """
    track_ids_t0 = set(labels.loc[labels["POSITION_T"] == t, "TRACK_ID"])
    track_ids_t1 = set(labels.loc[labels["POSITION_T"] == t + 1, "TRACK_ID"])
    common_ids = track_ids_t0 & track_ids_t1
    mask = (labels["POSITION_T"] == t) & (labels["TRACK_ID"].isin(common_ids))

    query_points = (
        labels.loc[mask, "POSITION_Z"],
        labels.loc[mask, "POSITION_Y"],
        labels.loc[mask, "POSITION_X"],
    )
    query_points = np.asarray(query_points).T  # (N, 3) with columns (z, y, x)
    query_points[:, 0] = query_points[:, 0] * scaling_um_to_px_z
    query_points[:, 1] = query_points[:, 1] * scaling_um_to_px_xy
    query_points[:, 2] = query_points[:, 2] * scaling_um_to_px_xy
    return query_points.astype(int)


def tidy_id(value):
    """TRACK_ID as read from a CSV -> plain Python value (1.0 becomes 1)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def read_query_points(labels, t, voxel_size_xy=1.0, voxel_size_z=1.0,
                      volume_shape=None):
    """Landmarks of timepoint ``t`` from a label table, in pixels.

    ``TRACK_ID`` and ``POSITION_T`` are optional: without ``POSITION_T`` every
    row belongs to ``t``, without ``TRACK_ID`` the points are numbered from 1.
    With ``volume_shape`` (Z, Y, X), points outside the stack are dropped.

    Returns
    -------
    ids : list
    points : ndarray (N, 3)
        (z, y, x) in pixels, not rounded.
    n_outside : int
        Number of points dropped for lying outside the stack.
    """
    if "POSITION_T" in labels.columns:
        rows = labels[labels["POSITION_T"] == t]
    else:
        rows = labels

    points = np.column_stack([
        rows["POSITION_Z"].to_numpy(float) / voxel_size_z,
        rows["POSITION_Y"].to_numpy(float) / voxel_size_xy,
        rows["POSITION_X"].to_numpy(float) / voxel_size_xy,
    ]).reshape(-1, 3)
    if "TRACK_ID" in rows.columns:
        ids = [tidy_id(v) for v in rows["TRACK_ID"].to_numpy()]
    else:
        ids = list(range(1, len(rows) + 1))

    if volume_shape is None:
        return ids, points, 0
    # the pipeline works on integer voxels inside the stack
    shape = np.asarray(volume_shape)
    inside = np.all((points >= 0) & (points.astype(int) <= shape - 1), axis=1)
    ids = [i for i, keep in zip(ids, inside) if keep]
    return ids, points[inside], int((~inside).sum())


def tracks_to_table(tracks, voxel_size_xy=1.0, voxel_size_z=1.0):
    """Tracks as a table in the format of the label CSV.

    ``tracks`` maps each timepoint to ``(ids, points)`` with points (N, 3) as
    (z, y, x) in pixels; the positions are written in label units.
    """
    frames = []
    for t in sorted(tracks):
        ids, points = tracks[t]
        frames.append(pd.DataFrame({
            "TRACK_ID": ids,
            "POSITION_X": np.round(points[:, 2] * voxel_size_xy, 6),
            "POSITION_Y": np.round(points[:, 1] * voxel_size_xy, 6),
            "POSITION_Z": np.round(points[:, 0] * voxel_size_z, 6),
            "POSITION_T": t,
        }))
    return pd.concat(frames, ignore_index=True)[LABEL_COLUMNS]


def normalize_stack(stack, upper):
    """Min-max normalise a stack to [0, upper] as float32."""
    return cv2.normalize(stack.astype(np.float32), None, 0, upper, cv2.NORM_MINMAX)


def match_projections(mip0, mip1, extractor, matcher, device):
    """LightGlue matches between two projections (Y, X).

    Returns the matched keypoints of both images as tensors [M, 2] in (x, y).
    """
    # SuperPoint expects 3 channels
    image0 = mip0.unsqueeze(0).repeat(3, 1, 1)
    image1 = mip1.unsqueeze(0).repeat(3, 1, 1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    return kpts0[matches[..., 0]], kpts1[matches[..., 1]]


def compute_similarity_maps(query_emb, centers, embedder, roi_size, volume_shape,
                            progress=None):
    """Cosine similarity between each landmark's embedding and every voxel of
    the ROI around its coarse prediction. Returns (N, Z_roi, Y_roi, X_roi).

    ``progress(done, total)``, if given, is called after every batch of
    embedded voxels and replaces the console progress bar.
    """
    if progress is not None:
        # ROIs overlap and embeddings are cached, so count each voxel once
        voxels = np.concatenate([
            roi_coords(roi_bounds(centers[j], roi_size, volume_shape, keep_size=True))
            for j in range(len(query_emb))])
        total = len(np.unique(np.ravel_multi_index(voxels.T, volume_shape)))
        embedder.progress = lambda done: progress(done, total)

    similarity_maps = []
    for j in tqdm(range(len(query_emb)), desc="Similarity maps",
                  disable=progress is not None):
        bounds = roi_bounds(centers[j], roi_size, volume_shape, keep_size=True)
        roi_emb = embedder(roi_coords(bounds))       # (M, D)
        sim = query_emb[j] @ roi_emb.T               # (M,)

        nz, ny, nx = [hi - lo for lo, hi in bounds]
        similarity_maps.append(sim.reshape(nz, ny, nx))
    return np.stack(similarity_maps, axis=0)


def load_models(path_model, dinov3_repo=None, device=None):
    """Load the embedding model and the LightGlue feature matcher.

    Parameters
    ----------
    path_model : str
        Checkpoint (state dict) of the embedding model.
    dinov3_repo : str, optional
        Local clone of the DINOv3 repository, see ``DINOv3Encoder``.
    device : str, optional
        Defaults to CUDA when available.

    Returns
    -------
    model, extractor, matcher, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DINOv3Encoder(embedding_dim=EMBEDDING_DIM, model=BACKBONE,
                          repo_dir=dinov3_repo).to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()

    extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return model, extractor, matcher, device


@dataclass
class TransferResult:
    """Output of ``transfer_landmarks``. All points are (z, y, x) in pixels."""

    points: np.ndarray               # (N, 3) refined positions at t + 1
    coarse_points: np.ndarray        # (N, 3) coarse LightGlue prediction
    z_offset: int                    # plane offset between the two stacks
    similarity_maps: np.ndarray      # (N, Z_roi, Y_roi, X_roi)
    info: dict                       # diagnostics of optimize_point_cloud
    stack_t0_norm: np.ndarray        # stacks normalised to [0, 255], float32
    stack_t1_norm: np.ndarray


def transfer_landmarks(stack_t0, stack_t1, query_points, model, extractor,
                       matcher, device, crop_size_z, roi_size_xy, roi_size_z,
                       spacing=(1.0, 1.0, 1.0), on_coarse_xy=None, progress=None):
    """Transfer landmarks from one stack to the next.

    Parameters
    ----------
    stack_t0, stack_t1 : ndarray (Z, Y, X)
        Stacks at timepoints ``t`` and ``t + 1``.
    query_points : ndarray (N, 3) of int
        Landmarks in ``stack_t0`` as (z, y, x) pixel coordinates.
    model, extractor, matcher, device
        As returned by ``load_models``.
    crop_size_z : int
        Number of z-slices in each embedding crop.
    roi_size_xy, roi_size_z : int
        Size of the ROI around the coarse prediction in which the similarity
        map is computed.
    spacing : (z, y, x)
        Physical voxel size; makes the geometry term of the refinement
        isotropic.
    on_coarse_xy : callable, optional
        Called as ``on_coarse_xy(mip0, mip1, pred_xy, m_kpts0, m_kpts1)`` as
        soon as the coarse x/y transfer is available.
    progress : callable, optional
        Called as ``progress(stage, done, total)`` with ``stage`` one of
        ``"coarse_xy"``, ``"z_registration"``, ``"embeddings"`` and
        ``"refinement"``. An exception raised by it aborts the run, which is
        how a caller cancels.

    Returns
    -------
    TransferResult
    """
    def report(stage, done, total):
        if progress is not None:
            progress(stage, done, total)

    def stage_progress(stage):
        if progress is None:
            return None
        return lambda done, total: progress(stage, done, total)

    roi_size = (roi_size_z, roi_size_xy, roi_size_xy)

    stack_t0_norm_0255 = normalize_stack(stack_t0, 255)
    stack_t1_norm_0255 = normalize_stack(stack_t1, 255)
    mip0 = torch.from_numpy(np.max(normalize_stack(stack_t0, 1), axis=0)).float()
    mip1 = torch.from_numpy(np.max(normalize_stack(stack_t1, 1), axis=0)).float()
    Z, Y, X = stack_t0.shape

    # ---- 1. coarse x/y: feature matches on the projections -----------------
    report("coarse_xy", 0, 1)
    m_kpts0, m_kpts1 = match_projections(mip0, mip1, extractor, matcher, device)
    pred_xy = transfer_points(m_kpts0, m_kpts1, query_points,
                              k=TRANSFER_NEIGHBOURS)
    report("coarse_xy", 1, 1)
    if on_coarse_xy is not None:
        on_coarse_xy(mip0, mip1, pred_xy, m_kpts0, m_kpts1)

    # ---- embeddings of the landmarks at t ----------------------------------
    query_emb = []
    for q in query_points:
        query_crop = crop_around_point(stack_t0_norm_0255, q, size_z=crop_size_z)
        query_emb.append(compute_embeddings_batch(query_crop, model, device))
    query_emb = np.stack(query_emb).squeeze()  # (N, D)

    # ---- 2. coarse z: plane-by-plane matching ------------------------------
    z_offset = z_registration(stack_t0, stack_t1, extractor, matcher, device,
                              show=False,
                              progress=stage_progress("z_registration"))

    pred_lg = np.column_stack([
        query_points[:, 0] + z_offset,  # z
        pred_xy[:, 1],                  # y
        pred_xy[:, 0],                  # x
    ])

    # ---- 3. similarity maps around the coarse prediction -------------------
    embedder = LazyEmbeddings(stack_t1_norm_0255, model, device,
                              batch_size=EMBEDDING_BATCH_SIZE,
                              size_z=crop_size_z)
    similarity_maps = compute_similarity_maps(
        query_emb, pred_lg, embedder, roi_size, (Z, Y, X),
        progress=stage_progress("embeddings"))

    # ---- 4. joint refinement of all landmarks ------------------------------
    report("refinement", 0, 1)
    centres = optimize.roi_centers(pred_lg, roi_size, (Z, Y, X))
    q_opt, info = optimize.optimize_point_cloud(
        points_A=query_points,
        points0=pred_lg,
        likelihood_maps=similarity_maps,
        spacing=spacing,
        volume_shape=stack_t1.shape,
        roi_center=centres,
        **OPTIMIZER_SETTINGS,
    )
    report("refinement", 1, 1)

    return TransferResult(
        points=q_opt,
        coarse_points=pred_lg,
        z_offset=z_offset,
        similarity_maps=similarity_maps,
        info=info,
        stack_t0_norm=stack_t0_norm_0255,
        stack_t1_norm=stack_t1_norm_0255,
    )


def snap_to_slices(points, n_slices):
    """Round the z coordinate of (N, 3) points to the nearest existing slice."""
    points = points.copy()
    points[:, 0] = np.clip(np.round(points[:, 0]), 0, n_slices - 1)
    return points


def run_stack(path_stacks, path_labels, path_model, path_result_csv,
              path_similarity_results, crop_size_z, roi_size_xy, roi_size_z,
              scaling_um_to_px_xy, scaling_px_to_um_xy, scaling_um_to_px_z,
              scaling_px_to_um_z, dinov3_repo=None):
    """Run the pipeline on every transition ``t -> t + 1`` of one stack and
    score the predictions against the labels at ``t + 1``.

    Parameters
    ----------
    path_stacks : str
        4D TIFF with axes (T, Z, Y, X).
    path_labels : str
        CSV with the columns in ``LABEL_COLUMNS``.
    path_model : str
        Checkpoint (state dict) of the embedding model.
    path_result_csv : str
        Prefix of the result tables; ``_t{t}_t{t+1}.csv`` is appended.
    path_similarity_results : str
        Prefix of the per-landmark figures; ``T{t+1}_TRACKID{id}.png`` is
        appended.
    crop_size_z : int
        Number of z-slices in each embedding crop.
    roi_size_xy, roi_size_z : int
        Size of the ROI around the coarse prediction in which the similarity
        map is computed.
    scaling_um_to_px_xy, scaling_px_to_um_xy : float
        Conversion between the x/y units of the labels and pixels.
    scaling_um_to_px_z, scaling_px_to_um_z : float
        Conversion between the z units of the labels and slices.
    dinov3_repo : str, optional
        Local clone of the DINOv3 repository, see ``DINOv3Encoder``.
    """
    stacks = tiff.imread(path_stacks)
    print("Original shape:", stacks.shape)
    labels = pd.read_csv(path_labels)
    tracks = labels[LABEL_COLUMNS]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model, extractor, matcher, device = load_models(path_model, dinov3_repo, device)

    all_distances = []

    for i in range(len(stacks) - 1):
        query_points = load_query_points(labels, i, scaling_um_to_px_xy,
                                         scaling_um_to_px_z)
        print("Number of query points:", np.asarray(query_points).shape)

        stack_t1 = stacks[i + 1]

        def show_coarse_transfer(mip0, mip1, pred_xy, m_kpts0, m_kpts1):
            plot_point_transfer(
                mip0,
                mip1,
                query_points,
                pred_xy,
                m_kpts0[::PLOT_EVERY_NTH_MATCH].cpu(),
                m_kpts1[::PLOT_EVERY_NTH_MATCH].cpu(),
                timestep=i,
            )
            plt.show()

        res = transfer_landmarks(
            stacks[i], stack_t1, query_points, model, extractor, matcher, device,
            crop_size_z, roi_size_xy, roi_size_z,
            spacing=(scaling_px_to_um_z, scaling_px_to_um_xy, scaling_px_to_um_xy),
            on_coarse_xy=show_coarse_transfer,
        )
        pred_lg, similarity_maps = res.coarse_points, res.similarity_maps
        print(similarity_maps.shape)

        # z snaps to the nearest slice; everything goes back to label units
        best_match_coordinates = snap_to_slices(res.points, stack_t1.shape[0])
        best_match_coordinates[:, 0] = best_match_coordinates[:, 0] * scaling_px_to_um_z
        best_match_coordinates[:, 1:] = best_match_coordinates[:, 1:] * scaling_px_to_um_xy

        # ---- results -------------------------------------------------------
        output_csv = path_result_csv + f"_t{i}_t{i+1}.csv"
        df, distances = create_result_csv(output_csv, tracks, best_match_coordinates,
                                          i, pred_lg, scaling_px_to_um_xy,
                                          scaling_px_to_um_z)
        all_distances.extend(distances.tolist())

        print("\nDistance statistics:")
        print(df["DISTANCE"].describe())

        print(f"\nMean distance:   {df['DISTANCE'].mean():.3f}")
        print(f"Median distance: {df['DISTANCE'].median():.3f}")

        visualize_matching(res.stack_t0_norm, res.stack_t1_norm, df, i,
                           scaling_um_to_px_xy, show=True)

        Path(path_similarity_results).parent.mkdir(parents=True, exist_ok=True)
        # for row_idx in range(len(df)):
        #     visualize_row(df, row_idx, res.stack_t1_norm, similarity_maps, i,
        #                   path_similarity_results, scaling_um_to_px_xy,
        #                   crop_size=roi_size_xy, show=True)

    print("\nOverall distance statistics across all time points:")
    print("Mean distance:   {:.3f}".format(np.mean(all_distances)))
    print("Median distance: {:.3f}".format(np.median(all_distances)))
    print("95th percentile: {:.3f}".format(np.percentile(all_distances, 95)))

    # NOTE: this figure is neither shown nor saved here; it only appears with
    # the next plt.show() call.
    plt.figure(figsize=(8, 5))
    plt.hist(all_distances, bins=50)
    plt.xlabel("Distance between prediction and human labels in um")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()


def track_stack(path_stack, path_labels, path_tracks_csv, settings):
    """Track the landmarks of one timepoint through the rest of the stack.

    The landmarks of the starting timepoint are transferred to the next
    timepoint, the predictions become the landmarks of the next step, and so
    on -- the same as stepping through the stack in the GUI without
    corrections. The tracks are written to ``path_tracks_csv`` in the format
    of the label CSV, and rewritten after every step, so a run that stops
    half-way keeps the timepoints already done.

    With ``settings.labels_every_timepoint`` the query points of every step
    are the CSV labels of that timepoint; a timepoint without labels falls
    back to the predictions. As in the GUI, the file then holds the labels of
    the starting timepoint and the predictions of every later one, each
    predicted from the labels of the timepoint before.

    Returns the tracks as a DataFrame.
    """
    s = settings
    stack = tiff.imread(path_stack)
    print("Stack shape:", stack.shape)
    if stack.ndim != 4 or stack.shape[0] < 2:
        raise ValueError(f"expected a 4D stack (T, Z, Y, X) with at least two "
                         f"timepoints, got the shape {stack.shape}")
    n_t, n_z = stack.shape[:2]

    labels = pd.read_csv(path_labels)
    if s.labels_every_timepoint and "POSITION_T" not in labels.columns:
        raise ValueError("--labels-every-timepoint needs a POSITION_T column "
                         "in the label CSV")
    t_start = s.start_timepoint
    if t_start is None:
        t_start = int(labels["POSITION_T"].min()) if "POSITION_T" in labels.columns else 0
    if not 0 <= t_start < n_t - 1:
        raise ValueError(f"start timepoint {t_start} is not in 0 ... {n_t - 2}")

    ids, query, n_outside = read_query_points(
        labels, t_start, s.voxel_size_xy, s.voxel_size_z, stack.shape[1:])
    print(f"Timepoint {t_start}: {len(ids)} query points"
          + (f" ({n_outside} outside the stack skipped)" if n_outside else ""))

    def check_enough(t):
        if len(ids) < 2:
            raise ValueError(f"need at least two query points at timepoint {t}, "
                             f"found {len(ids)}: the refinement uses the "
                             "distances between them")

    check_enough(t_start)
    model, extractor, matcher, device = load_models(s.path_model, s.dinov3_repo)
    print("Using device:", device)

    tracks = {t_start: (ids, query)}
    for t in range(t_start, n_t - 1):
        print(f"\nTimepoint {t} -> {t + 1}")
        if s.labels_every_timepoint and t > t_start:
            # the saved predictions for t stay in the file
            label_ids, label_points, n_outside = read_query_points(
                labels, t, s.voxel_size_xy, s.voxel_size_z, stack.shape[1:])
            if label_ids:
                ids, query = label_ids, label_points
                print(f"{len(ids)} query points from the CSV"
                      + (f" ({n_outside} outside the stack skipped)" if n_outside else ""))
            else:
                print("No labels for this timepoint in the CSV: using the predictions")
            check_enough(t)

        res = transfer_landmarks(
            stack[t], stack[t + 1], query.astype(int), model, extractor,
            matcher, device, s.crop_size_z, s.roi_size_xy, s.roi_size_z,
            spacing=(s.voxel_size_z, s.voxel_size_xy, s.voxel_size_xy),
        )
        pred = snap_to_slices(res.points, n_z)
        shift = np.linalg.norm((pred - query) * [s.voxel_size_z, s.voxel_size_xy,
                                                 s.voxel_size_xy], axis=1)
        print(f"z-shift {int(res.z_offset):+d} slices; displacement of the "
              f"points: median {np.median(shift):.3f}, max {shift.max():.3f}")

        tracks[t + 1] = (ids, pred)
        table = tracks_to_table(tracks, s.voxel_size_xy, s.voxel_size_z)
        table.to_csv(path_tracks_csv, index=False)
        query = pred

    print(f"\nSaved {table['TRACK_ID'].nunique()} tracks over timepoints "
          f"{t_start}-{n_t - 1} to {path_tracks_csv}")
    return table


# ----------------------------------------------------------------------------
# batch handling
# ----------------------------------------------------------------------------

def find_stack_label_pairs(stacks_dir, labels_dir, stack_suffixes=STACK_SUFFIXES):
    """Return (name, stack_path, label_path) for every stack that has a CSV
    with the same name. Stacks without a CSV and CSVs without a stack are
    reported and ignored."""
    stacks = {p.stem: p for p in sorted(Path(stacks_dir).iterdir())
              if p.is_file() and p.suffix.lower() in stack_suffixes}
    labels = {p.stem: p for p in sorted(Path(labels_dir).iterdir())
              if p.is_file() and p.suffix.lower() == ".csv"}

    only_stack = sorted(stacks.keys() - labels.keys())
    only_label = sorted(labels.keys() - stacks.keys())
    if only_stack:
        print(f"[warn] {len(only_stack)} stack(s) without csv, ignored: {', '.join(only_stack)}")
    if only_label:
        print(f"[warn] {len(only_label)} csv(s) without stack, ignored: {', '.join(only_label)}")

    return [(name, stacks[name], labels[name]) for name in sorted(stacks.keys() & labels.keys())]


def collect_pairs(path_stacks, path_labels):
    """Resolve ``--stacks`` / ``--labels`` to a list of (name, stack, csv).

    Two files are one pair; two directories are searched for stacks and CSVs
    with matching names.
    """
    if path_stacks.is_dir() and path_labels.is_dir():
        pairs = find_stack_label_pairs(path_stacks, path_labels)
        print(f"Found {len(pairs)} matching stack/csv pairs in {path_stacks}")
        return pairs

    for p in (path_stacks, path_labels):
        if not p.exists():
            raise FileNotFoundError(p)
    if not (path_stacks.is_file() and path_labels.is_file()):
        raise ValueError("--stacks and --labels must be either two files or "
                         "two directories")
    if path_stacks.stem != path_labels.stem:
        print(f"[warn] names differ: {path_stacks.name} vs {path_labels.name}")
    return [(path_stacks.stem, path_stacks, path_labels)]


def is_done(result_dir, require_marker=True, marker_name=DONE_MARKER_NAME):
    """Whether a result folder is complete.

    With ``require_marker`` only a folder holding the marker file written
    after a successful run counts; otherwise any non-empty folder does, which
    includes runs that failed half-way.
    """
    result_dir = Path(result_dir)
    if require_marker:
        return (result_dir / marker_name).exists()
    return result_dir.exists() and any(result_dir.iterdir())


def run_one(name, path_stack, path_label, result_dir, settings):
    """Run the pipeline for one stack/label pair. Returns True on success.

    An exception is printed rather than raised so that a batch carries on with
    the next stack. The done marker is only written after a successful run.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run ] {name}")
    try:
        if settings.mode == "track":
            track_stack(path_stack, path_label,
                        result_dir / f"{name}_tracks.csv", settings)
        else:
            run_stack(
                str(path_stack),
                str(path_label),
                settings.path_model,
                str(result_dir / name),
                str(result_dir / "similarity_maps" / f"{name}_"),
                settings.crop_size_z,
                settings.roi_size_xy,
                settings.roi_size_z,
                1.0 / settings.voxel_size_xy,
                settings.voxel_size_xy,
                1.0 / settings.voxel_size_z,
                settings.voxel_size_z,
                dinov3_repo=settings.dinov3_repo,
            )
    except Exception:
        print(f"[fail] {name}")
        traceback.print_exc()
        return False
    (result_dir / DONE_MARKER_NAME).touch()
    return True


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Transfer 3D landmarks between consecutive timepoints of a "
                    "stack: track them from one timepoint on, or compare the "
                    "predictions with labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("evaluate", "track"), default="evaluate",
                        help="track: start from the labels of one timepoint and "
                             "carry the predictions through the stack; "
                             "evaluate: transfer the labels of every timepoint "
                             "and score the predictions against the labels of "
                             "the next one")
    parser.add_argument("--start-timepoint", type=int, default=None,
                        help="track: timepoint whose labels are the query points "
                             "(default: the first timepoint in the CSV)")
    parser.add_argument("--labels-every-timepoint", action="store_true",
                        help="track: take the query points of every step from the "
                             "CSV labels of that timepoint instead of the "
                             "predictions (where the CSV has labels for it)")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--stacks", type=Path, required=True,
                       help="4D TIFF (T, Z, Y, X), or a directory of them")
    paths.add_argument("--labels", type=Path, required=True,
                       help="CSV with the columns " + ", ".join(LABEL_COLUMNS)
                            + "; or a directory of CSVs named like the stacks")
    paths.add_argument("--model", type=Path, required=True,
                       help="checkpoint of the embedding model (.pth)")
    paths.add_argument("--output", type=Path, default=Path("results"),
                       help="results are written to <output>/<stack name>/")
    paths.add_argument("--dinov3-repo", type=Path, default=Path(DEFAULT_DINOV3_REPO),
                       help="local clone of the DINOv3 repository "
                            "(default can also be set with DINOV3_REPO)")

    params = parser.add_argument_group("parameters")
    params.add_argument("--crop-size-z", type=int, default=3,
                        help="z-slices per embedding crop")
    params.add_argument("--roi-size-xy", type=int, default=32,
                        help="x/y size of the similarity ROI in pixels")
    params.add_argument("--roi-size-z", type=int, default=8,
                        help="z size of the similarity ROI in slices")
    params.add_argument("--voxel-size-xy", type=float, default=1.0,
                        help="size of a pixel in the units of the label CSV; "
                             "1 if the labels are in pixels")
    params.add_argument("--voxel-size-z", type=float, default=1.0,
                        help="distance between slices in the units of the "
                             "label CSV; 1 if the labels are in slices")

    batch = parser.add_argument_group("batch")
    batch.add_argument("--skip-done", action="store_true",
                       help="skip stacks whose results already exist")
    batch.add_argument("--no-done-marker", action="store_true",
                       help="with --skip-done: treat any non-empty result "
                            f"folder as done instead of requiring the "
                            f"{DONE_MARKER_NAME} file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.labels_every_timepoint and args.mode != "track":
        raise SystemExit("--labels-every-timepoint only applies to --mode track")

    settings = Settings(
        path_model=str(args.model),
        dinov3_repo=str(args.dinov3_repo),
        crop_size_z=args.crop_size_z,
        roi_size_xy=args.roi_size_xy,
        roi_size_z=args.roi_size_z,
        voxel_size_xy=args.voxel_size_xy,
        voxel_size_z=args.voxel_size_z,
        mode=args.mode,
        start_timepoint=args.start_timepoint,
        labels_every_timepoint=args.labels_every_timepoint,
    )

    pairs = collect_pairs(args.stacks, args.labels)
    args.output.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = 0, 0, []
    for name, path_stack, path_label in pairs:
        result_dir = args.output / name

        if args.skip_done and is_done(result_dir, not args.no_done_marker):
            print(f"[skip] {name} - already computed")
            skipped += 1
            continue

        if run_one(name, path_stack, path_label, result_dir, settings):
            done += 1
        else:
            failed.append(name)

    print(f"\nDone: {done} | skipped: {skipped} | failed: {len(failed)}")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
