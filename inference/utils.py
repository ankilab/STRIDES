"""
Building blocks of the landmark-transfer pipeline.

The pipeline moves a set of 3D landmarks (z, y, x) from the stack at timepoint
``t`` to the stack at timepoint ``t + 1``:

1. coarse x/y   -- LightGlue matches on the maximum-intensity projections,
                   interpolated to the landmarks (``transfer_points``)
2. coarse z     -- plane-by-plane LightGlue matching (``z_registration``)
3. similarity   -- embedding of every voxel in a small ROI around the coarse
                   guess, compared against the landmark's own embedding
                   (``DINOv3Encoder``, ``LazyEmbeddings``, ``roi_bounds``)
4. refinement   -- joint optimisation over all landmarks (see ``optimize.py``)

This module holds steps 1-3 plus result export and the figures. All point
arrays are ordered (z, y, x) unless a function says otherwise; LightGlue
keypoints and the ``*_xy`` arrays are ordered (x, y).
"""

import os

# PyTorch and OpenCV/NumPy can each bring their own OpenMP runtime on Windows,
# which aborts the interpreter on import. Must be set before they are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from lightglue.utils import rbd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from scipy.spatial import cKDTree

from optimize import soft_center_of_mass

# Local clone of https://github.com/facebookresearch/dinov3. Override with the
# DINOV3_REPO environment variable or the ``repo_dir`` argument of the encoder.
DEFAULT_DINOV3_REPO = os.environ.get(
    "DINOV3_REPO", str(Path(__file__).resolve().parent.parent / "dinov3"))


# ----------------------------------------------------------------------------
# embedding model
# ----------------------------------------------------------------------------

class DINOv3Encoder(nn.Module):
    """DINOv3 backbone applied slice by slice to a small 3D crop.

    The RGB patch embedding is replaced by a single-channel one (initialised
    with the channel mean of the pretrained weights), each z-slice is encoded
    independently, projected to ``embedding_dim`` and max-pooled over z. The
    output is L2-normalised, so a dot product of two embeddings is their cosine
    similarity.

    Parameters
    ----------
    embedding_dim : int
        Size of the output embedding.
    model : str
        Name of the DINOv3 hub entry point, e.g. ``"dinov3_vitb16"``.
    freeze_backbone : bool
        Freeze all backbone weights and train the projection head only.
    repo_dir : str or Path, optional
        Local clone of the DINOv3 repository. Defaults to
        ``DEFAULT_DINOV3_REPO``.
    """

    def __init__(self, embedding_dim=128, model="dinov3_vitb16",
                 freeze_backbone=False, repo_dir=None):
        super().__init__()

        self.backbone = torch.hub.load(
            str(DEFAULT_DINOV3_REPO if repo_dir is None else repo_dir),
            model,
            source="local",
            pretrained=True,
        )

        # swap the 3-channel patch embedding for a 1-channel one
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.backbone.patch_embed.proj = new_proj

        feat_dim = self.backbone.embed_dim  # 768 for ViT-B

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x):
        """x: [B, C, T, H, W] with T the z-slices. Returns [B, embedding_dim]."""
        B, C, T, H, W = x.shape

        # encode every slice on its own
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.backbone(x)            # [B*T, feat_dim]
        feats = self.fc(feats)              # [B*T, D]

        # max-pool over the slices
        feats = feats.view(B, T, -1)
        feats = feats.max(dim=1).values

        return F.normalize(feats, dim=-1)   # [B, D]


# ----------------------------------------------------------------------------
# crops and embeddings
# ----------------------------------------------------------------------------

def crop_around_point(stack, point, size_x=128, size_y=128, size_z=7):
    """Crop one 3D window out of ``stack`` (Z, Y, X), centred on ``point``.

    Parameters
    ----------
    stack : ndarray (Z, Y, X)
    point : (z, y, x)
        Window centre; cast to int. It lands at index ``size // 2`` on each
        axis.
    size_x, size_y, size_z : int
        Size of the window.

    Returns
    -------
    torch.Tensor [1, 1, size_z, size_y, size_x]
        Batch and channel dimensions are added. Where the window extends past
        the stack it is filled by reflection.
    """
    z_dim, y_dim, x_dim = stack.shape
    zc, yc, xc = map(int, point)

    # start/end on each axis such that (end - start) == size
    sx = xc - (size_x // 2)
    ex = sx + size_x
    sy = yc - (size_y // 2)
    ey = sy + size_y
    sz = zc - (size_z // 2)
    ez = sz + size_z

    cropped = stack[max(sz, 0):min(ez, z_dim),
                    max(sy, 0):min(ey, y_dim),
                    max(sx, 0):min(ex, x_dim)]

    if cropped.shape == (size_z, size_y, size_x):
        return torch.from_numpy(cropped[np.newaxis, np.newaxis, :])

    # the window hangs over a border: reflect-pad the missing part
    pad_z = (max(0, -sz), max(0, ez - z_dim))
    pad_y = (max(0, -sy), max(0, ey - y_dim))
    pad_x = (max(0, -sx), max(0, ex - x_dim))
    padded = np.pad(cropped, (pad_z, pad_y, pad_x), mode="reflect")

    return torch.from_numpy(padded[np.newaxis, np.newaxis, :])


def compute_embeddings_batch(crops, model, device):
    """Embed a batch of crops [B, 1, Z, Y, X]. Returns a CPU tensor [B, D]."""
    with torch.no_grad():
        emb = model(crops.to(device))
        if emb.shape[0] != crops.shape[0]:
            raise ValueError(
                f"model returned {tuple(emb.shape)} for a batch of "
                f"{crops.shape[0]} — the batch dimension must be preserved")
        return emb.cpu()


class LazyEmbeddings:
    """Embeds voxels of a stack on demand and caches the result by (z, y, x).

    The ROIs of neighbouring landmarks overlap, so caching avoids embedding the
    same voxel twice.

    Parameters
    ----------
    stack : ndarray (Z, Y, X)
    model : nn.Module
        Embedding model, see ``DINOv3Encoder``.
    device : str or torch.device
    batch_size : int
        Number of crops per forward pass.
    size_z : int
        Number of slices in each crop, see ``crop_around_point``.
    progress : callable, optional
        Called as ``progress(n_cached)`` after every batch with the number of
        voxels embedded so far. An exception raised by it aborts the run.
    """

    def __init__(self, stack, model, device, batch_size=64, size_z=7,
                 progress=None):
        self.stack = stack
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.size_z = size_z
        self.progress = progress
        self.cache = {}

    def __call__(self, coords):
        """coords: (M, 3) voxel coordinates. Returns embeddings (M, D)."""
        keys = [tuple(int(v) for v in p) for p in coords]

        # unique, order-preserving list of the points not embedded yet
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))

        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            crops = torch.cat(
                [crop_around_point(self.stack, p, size_z=self.size_z)
                 for p in chunk],
                dim=0,
            )
            embs = compute_embeddings_batch(crops, self.model, self.device)
            for p, e in zip(chunk, embs):
                self.cache[p] = e
            if self.progress is not None:
                self.progress(len(self.cache))

        return np.stack([self.cache[k] for k in keys])  # (M, D)


# ----------------------------------------------------------------------------
# regions of interest
# ----------------------------------------------------------------------------

def roi_bounds(center, roi_size, shape, keep_size=True):
    """Bounds of the ROI window around ``center``.

    Parameters
    ----------
    center : (z, y, x)
    roi_size : (Z_roi, Y_roi, X_roi)
    shape : (Z, Y, X)
        Shape of the stack.
    keep_size : bool
        If True, a window that would hang over a border is shifted back inside
        the volume instead of being cropped, so every ROI has the same shape
        (needed to stack the similarity maps into one array).

    Returns
    -------
    list of (lo, hi)
        One half-open interval per axis: ``[(z0, z1), (y0, y1), (x0, x1)]``.
    """
    bounds = []
    for c, r, n in zip(center, roi_size, shape):
        r = int(r)
        lo = int(c) - r // 2
        hi = lo + r                      # exclusive, always exactly r wide

        if keep_size:
            lo = min(max(lo, 0), max(0, n - r))
            hi = min(lo + r, n)
        else:
            lo, hi = max(0, lo), min(n, hi)

        bounds.append((lo, hi))
    return bounds


def roi_coords(bounds):
    """All voxel coordinates inside the ROI, in C order. Returns (M, 3)."""
    (z0, z1), (y0, y1), (x0, x1) = bounds
    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1),
        np.arange(y0, y1),
        np.arange(x0, x1),
        indexing="ij",
    )
    return np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)


def roi_origin(center, size, n):
    """Lower bound of the window ``roi_bounds(keep_size=True)`` places on one
    axis: shifted back inside the volume rather than cropped."""
    lo = int(center) - int(size) // 2
    return int(min(max(lo, 0), max(0, int(n) - int(size))))


# ----------------------------------------------------------------------------
# coarse registration
# ----------------------------------------------------------------------------

def transfer_points(matches_img1, matches_img2, query_points, k=10, eps=1e-6):
    """Transfer points from image 1 to image 2 using matched features.

    Each query point is moved by the inverse-distance-weighted mean
    displacement of its ``k`` nearest matched features.

    Parameters
    ----------
    matches_img1 : torch.Tensor [M, 2]
        Feature coordinates (x, y) in image 1.
    matches_img2 : torch.Tensor [M, 2]
        Corresponding feature coordinates (x, y) in image 2.
    query_points : ndarray [N, 3]
        Columns are (z, y, x). z is ignored.
    k : int
        Number of nearest matches used for the interpolation.
    eps : float
        Added to the distances so a query point on top of a feature does not
        divide by zero.

    Returns
    -------
    predicted_xy : ndarray [N, 2]
        Predicted (x, y) coordinates in image 2.
    """
    pts1 = matches_img1.detach().cpu().numpy()
    pts2 = matches_img2.detach().cpu().numpy()
    displacement = pts2 - pts1

    query_xy = query_points[:, [2, 1]]  # (x, y)

    k = min(k, len(pts1))
    dists, idx = cKDTree(pts1).query(query_xy, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    # inverse-distance weights
    weights = 1.0 / (dists + eps)
    weights /= weights.sum(axis=1, keepdims=True)

    disp_interp = np.sum(displacement[idx] * weights[..., None], axis=1)

    return query_xy + disp_interp


def z_registration(stack_t0, stack_t1, extractor, matcher, device, show=False,
                   progress=None):
    """Integer z-offset between two stacks from plane-by-plane feature matching.

    Every plane of ``stack_t0`` is matched against every plane of ``stack_t1``
    with LightGlue; the summed match confidence fills a score matrix. Two
    stacks shifted by ``k`` planes produce a ridge along the ``k``-th diagonal
    of that matrix, so the offset is the diagonal with the largest total score.
    Only diagonals that cover at least half the planes of ``stack_t0`` are
    considered.

    Parameters
    ----------
    stack_t0, stack_t1 : ndarray (Z, Y, X)
    extractor, matcher
        LightGlue feature extractor (e.g. SuperPoint) and matcher.
    device : str or torch.device
    show : bool
        Display the score matrix.
    progress : callable, optional
        Called as ``progress(done, total)`` after every plane of ``stack_t0``.
        An exception raised by it aborts the run.

    Returns
    -------
    k_best : int
        Plane ``j`` of ``stack_t0`` corresponds to plane ``j + k_best`` of
        ``stack_t1``. A sub-plane estimate (parabolic fit through the peak) is
        printed for information only.
    """
    scores = np.zeros((stack_t0.shape[0], stack_t1.shape[0]), dtype=np.float32)

    for j in range(stack_t0.shape[0]):
        plane0 = cv2.normalize(stack_t0[j], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        for i in range(stack_t1.shape[0]):
            plane1 = cv2.normalize(stack_t1[i], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

            # SuperPoint expects 3 channels
            img0 = np.repeat(plane0[np.newaxis, :, :], 3, axis=0)
            img1 = np.repeat(plane1[np.newaxis, :, :], 3, axis=0)

            feats0 = extractor.extract(torch.from_numpy(img0).to(device))
            feats1 = extractor.extract(torch.from_numpy(img1).to(device))
            matches01 = rbd(matcher({"image0": feats0, "image1": feats1}))

            if matches01["scores"].numel() > 0:
                scores[j, i] = torch.sum(matches01["scores"]).item()

        if progress is not None:
            progress(j + 1, stack_t0.shape[0])

    if show:
        plt.imshow(scores, cmap="viridis")
        plt.title("Mean Match Scores between Planes of Stack T0 and T1")
        plt.xlabel("Planes of Stack T1")
        plt.ylabel("Planes of Stack T0")
        plt.colorbar()
        plt.show()

    # scores[j, i]: plane j of T0 against plane i of T1
    M, N = scores.shape
    min_overlap = M // 2            # require enough overlapping planes

    offsets = np.arange(-(M - 1), N)
    diag_score = np.full(offsets.shape, -np.inf)
    for idx, k in enumerate(offsets):
        js = np.arange(max(0, -k), min(M, N - k))   # planes where j and j+k both exist
        if len(js) >= min_overlap:
            diag_score[idx] = scores[js, js + k].sum()

    p = int(np.argmax(diag_score))
    if not np.isfinite(diag_score[p]):
        raise ValueError(
            f"no plane offset between a {M}-plane and a {N}-plane stack "
            f"overlaps in at least {min_overlap} planes")
    k_best = offsets[p]

    # parabolic sub-plane refinement (not possible at either end of the range)
    k_sub = float(k_best)
    if 0 < p < len(diag_score) - 1:
        y0, y1, y2 = diag_score[p - 1], diag_score[p], diag_score[p + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        k_sub = k_best + delta

    print(f"Best plane offset: {k_best}, sub-plane refinement: {k_sub:.3f}")

    return k_best


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------

def create_result_csv(output_csv, tracks, best_match_coordinates, i, pred_lg,
                      scaling_px_to_um_xy, scaling_px_to_um_z):
    """Write predictions and labels for the transition ``i -> i + 1`` to a CSV.

    Only tracks that are labelled at both timepoints are written. The
    predictions must be in the order of those tracks at timepoint ``i``.

    Parameters
    ----------
    output_csv : str or Path
    tracks : DataFrame
        Columns TRACK_ID, POSITION_X, POSITION_Y, POSITION_Z, POSITION_T.
    best_match_coordinates : ndarray (N, 3)
        Final predictions (z, y, x) at timepoint ``i + 1``, in the units of the
        labels.
    i : int
        Index of the first timepoint.
    pred_lg : ndarray (N, 3)
        Coarse LightGlue predictions (z, y, x), in pixels.
    scaling_px_to_um_xy, scaling_px_to_um_z : float
        Convert ``pred_lg`` to the units of the labels.

    Returns
    -------
    df : DataFrame
        One row per track with the columns ``TRACK_ID``, ``T{i}_X/Y/Z``
        (label at ``i``), ``T{i+1}_X/Y/Z`` (label at ``i + 1``), ``LG_X/Y/Z``
        (coarse prediction), ``PRED_X/Y/Z`` (final prediction) and
        ``DISTANCE`` (Euclidean distance between prediction and label at
        ``i + 1``).
    distances : Series
        The ``DISTANCE`` column.
    """
    t0 = tracks[tracks["POSITION_T"] == i].copy()
    t1 = tracks[tracks["POSITION_T"] == i + 1].copy()

    common_ids = set(t0["TRACK_ID"]) & set(t1["TRACK_ID"])
    t0 = t0[t0["TRACK_ID"].isin(common_ids)]
    t1 = t1[t1["TRACK_ID"].isin(common_ids)]

    assert len(t0) == len(best_match_coordinates), (
        f"Number of predictions ({len(best_match_coordinates)}) "
        f"does not match number of T={i} rows ({len(t0)})"
    )

    # attach the predictions
    t0["PRED_Z"] = best_match_coordinates[:, 0]
    t0["PRED_Y"] = best_match_coordinates[:, 1]
    t0["PRED_X"] = best_match_coordinates[:, 2]
    t0["LG_Z"] = pred_lg[:, 0] * scaling_px_to_um_z
    t0["LG_Y"] = pred_lg[:, 1] * scaling_px_to_um_xy
    t0["LG_X"] = pred_lg[:, 2] * scaling_px_to_um_xy

    t0 = t0.rename(columns={
        "POSITION_X": f"T{i}_X",
        "POSITION_Y": f"T{i}_Y",
        "POSITION_Z": f"T{i}_Z",
    })
    t1 = t1.rename(columns={
        "POSITION_X": f"T{i+1}_X",
        "POSITION_Y": f"T{i+1}_Y",
        "POSITION_Z": f"T{i+1}_Z",
    })

    t0 = t0[[
        "TRACK_ID",
        f"T{i}_X", f"T{i}_Y", f"T{i}_Z",
        "PRED_X", "PRED_Y", "PRED_Z",
        "LG_X", "LG_Y", "LG_Z",
    ]]
    t1 = t1[[
        "TRACK_ID",
        f"T{i+1}_X", f"T{i+1}_Y", f"T{i+1}_Z",
    ]]

    df = pd.merge(t0, t1, on="TRACK_ID", how="inner")

    # Euclidean distance between prediction and label
    distances = np.sqrt(
        (df["PRED_X"] - df[f"T{i+1}_X"])**2 +
        (df["PRED_Y"] - df[f"T{i+1}_Y"])**2 +
        (df["PRED_Z"] - df[f"T{i+1}_Z"])**2
    )

    # mean absolute error per axis
    dx = np.mean(np.abs(df["PRED_X"] - df[f"T{i+1}_X"]))
    dy = np.mean(np.abs(df["PRED_Y"] - df[f"T{i+1}_Y"]))
    dz = np.mean(np.abs(df["PRED_Z"] - df[f"T{i+1}_Z"]))

    print(f"dx = {dx}")
    print(f"dy = {dy}")
    print(f"dz = {dz}")

    df["DISTANCE"] = distances

    cols = [
        "TRACK_ID",
        f"T{i}_X", f"T{i}_Y", f"T{i}_Z",
        f"T{i+1}_X", f"T{i+1}_Y", f"T{i+1}_Z",
        "LG_X", "LG_Y", "LG_Z",
        "PRED_X", "PRED_Y", "PRED_Z",
        "DISTANCE",
    ]
    df[cols].to_csv(output_csv, index=False)

    print(f"Saved {len(df)} tracks to {output_csv}")

    return df, distances


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def plot_point_transfer(img1, img2, query_points, predicted_xy, m_kpts0,
                        m_kpts1, point_size=20, line_alpha=0.5, line_width=2.0,
                        timestep=0):
    """Side-by-side view of the coarse x/y transfer.

    Matched features are drawn in orange, landmarks in magenta, each connected
    to its counterpart in the other image.

    Parameters
    ----------
    img1, img2 : ndarray (Y, X)
        The two images, typically maximum-intensity projections.
    query_points : ndarray [N, 3]
        Landmarks in image 1; columns are (z, y, x).
    predicted_xy : ndarray [N, 2]
        Predicted landmark coordinates in image 2 as (x, y).
    m_kpts0, m_kpts1 : array [M, 2]
        Matched feature coordinates (x, y) in image 1 and image 2.
    point_size, line_alpha, line_width : float
        Marker size and style of the connecting lines.
    timestep : int
        Index of the first timepoint, used in the titles.

    Returns
    -------
    fig, (ax1, ax2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   constrained_layout=True)

    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")

    qx = query_points[:, 2]
    qy = query_points[:, 1]
    px = predicted_xy[:, 0]
    py = predicted_xy[:, 1]

    ax1.scatter(m_kpts0[:, 0], m_kpts0[:, 1], s=point_size, marker="o", color="orange")
    ax1.scatter(qx, qy, s=point_size, marker="x", color="magenta")
    ax2.scatter(m_kpts1[:, 0], m_kpts1[:, 1], s=point_size, marker="o", color="orange")
    ax2.scatter(px, py, s=point_size, marker="x", color="magenta")

    # landmark -> predicted landmark
    for x1, y1, x2, y2 in zip(qx, qy, px, py):
        fig.add_artist(ConnectionPatch(
            xyA=(x2, y2),
            xyB=(x1, y1),
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            alpha=line_alpha,
            linewidth=line_width,
            color="magenta",
        ))

    # matched feature -> matched feature
    for x1, y1, x2, y2 in zip(m_kpts0[:, 0], m_kpts0[:, 1],
                              m_kpts1[:, 0], m_kpts1[:, 1]):
        fig.add_artist(ConnectionPatch(
            xyA=(x2, y2),
            xyB=(x1, y1),
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            alpha=line_alpha,
            linewidth=line_width,
            color="orange",
        ))

    ax1.set_title(f"T={timestep}")
    ax2.set_title(f"T={timestep+1}")

    ax1.set_axis_off()
    ax2.set_axis_off()

    return fig, (ax1, ax2)


def visualize_matching(stack_t0, stack_t1, df, i, scaling_um_to_px_xy,
                       show=False):
    """Predictions against labels on the two maximum-intensity projections.

    Left: labels at timepoint ``i``. Right: labels (green) and predictions
    (magenta) at ``i + 1``. Every landmark on the left is connected to its
    label and to its prediction on the right.

    Parameters
    ----------
    stack_t0, stack_t1 : ndarray (Z, Y, X)
    df : DataFrame
        Result table from ``create_result_csv``.
    i : int
        Index of the first timepoint.
    scaling_um_to_px_xy : float
        Converts the X/Y columns of ``df`` to pixels.
    show : bool
        Display the figure.
    """
    mip0 = np.max(stack_t0, axis=0)
    mip1 = np.max(stack_t1, axis=0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 10),
                                   constrained_layout=True)

    ax0.imshow(mip0, cmap="gray")
    ax1.imshow(mip1, cmap="gray")

    ax0.set_xlim(0, mip0.shape[1])
    ax0.set_ylim(mip0.shape[0], 0)
    ax1.set_xlim(0, mip1.shape[1])
    ax1.set_ylim(mip1.shape[0], 0)

    ax0.set_title(f"T={i}")
    ax1.set_title(f"T={i+1}")

    ax0.scatter(
        df[f"T{i}_X"] * scaling_um_to_px_xy,
        df[f"T{i}_Y"] * scaling_um_to_px_xy,
        s=10,
        c="lime",
        label="T0",
    )
    ax1.scatter(
        df[f"T{i+1}_X"] * scaling_um_to_px_xy,
        df[f"T{i+1}_Y"] * scaling_um_to_px_xy,
        s=10,
        c="lime",
        marker="o",
        label="Label",
    )
    ax1.scatter(
        df["PRED_X"] * scaling_um_to_px_xy,
        df["PRED_Y"] * scaling_um_to_px_xy,
        s=10,
        c="magenta",
        marker="x",
        label="Prediction",
    )

    for _, row in df.iterrows():
        # T0 -> prediction
        fig.add_artist(ConnectionPatch(
            xyA=(row["PRED_X"] * scaling_um_to_px_xy, row["PRED_Y"] * scaling_um_to_px_xy),
            coordsA=ax1.transData,
            xyB=(row[f"T{i}_X"] * scaling_um_to_px_xy, row[f"T{i}_Y"] * scaling_um_to_px_xy),
            coordsB=ax0.transData,
            color="magenta",
            alpha=0.7,
            linewidth=2,
        ))

        # T0 -> label
        fig.add_artist(ConnectionPatch(
            xyA=(row[f"T{i+1}_X"] * scaling_um_to_px_xy, row[f"T{i+1}_Y"] * scaling_um_to_px_xy),
            coordsA=ax1.transData,
            xyB=(row[f"T{i}_X"] * scaling_um_to_px_xy, row[f"T{i}_Y"] * scaling_um_to_px_xy),
            coordsB=ax0.transData,
            color="lime",
            alpha=0.7,
            linewidth=2,
        ))

    ax1.legend()

    ax0.set_axis_off()
    ax1.set_axis_off()

    if show:
        plt.show()


def get_crop(img_slice, center_x, center_y, size=33):
    """A (size, size) crop centred on (center_x, center_y), zero-padded at the
    border. The centre pixel lands at index ``size // 2``, matching the window
    convention used by ``roi_bounds`` and ``crop_around_point``.
    """
    half = size // 2
    x0, y0 = center_x - half, center_y - half
    x1, y1 = x0 + size, y0 + size          # exactly `size` wide, for odd OR even
    H, W = img_slice.shape

    crop = np.zeros((size, size), dtype=img_slice.dtype)
    sx0, sx1 = max(x0, 0), min(x1, W)
    sy0, sy1 = max(y0, 0), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img_slice[sy0:sy1, sx0:sx1]
    return crop


def visualize_row(df, row_idx, stack, sim_maps, i, path,
                  scale=1.0, crop_size=33, crop_size_z=None,
                  sim_origin=None, com_temperature=0.08, com_window=4,
                  share_scale=True, show=False, dpi=130):
    """Per-landmark inspection figure: image crop (top row) and similarity map
    (bottom row), one column per z-slice.

    Markers: label at ``i + 1`` (green circle), coarse LightGlue prediction
    (cyan square), final prediction (magenta triangle), maximum of the
    similarity map (red star) and its soft centre of mass (yellow cross).

    Parameters
    ----------
    df : DataFrame
        Result table from ``create_result_csv``.
    row_idx : int
        Row of ``df`` (and index into ``sim_maps``) to draw.
    stack : ndarray (Z, Y, X)
        Stack at timepoint ``i + 1``.
    sim_maps : array (N, Z_roi, Y_roi, X_roi)
        Similarity volume per landmark.
    i : int
        Index of the first timepoint.
    path : str
        Prefix of the output file; ``T{i+1}_TRACKID{id}.png`` is appended.
    scale : float
        Multiplier taking the df's X/Y units to stack pixels
        (1 / scaling_px_to_um). Z is assumed already in slices.
    crop_size : int
        Width/height of the image crop, in pixels.
    crop_size_z : int, optional
        How many slices to draw. Defaults to every slice of the similarity
        map; a smaller number takes a window centred on the peak slice.
    sim_origin : (z0, y0, x0), optional
        Origin of the similarity volume inside the stack, in pixels. Computed
        from the LG prediction with the ``roi_bounds`` rule if omitted -- pass
        the real bounds if you have them.
    com_temperature, com_window
        Settings of ``optimize.soft_center_of_mass`` for the yellow marker.
    share_scale : bool
        One imshow intensity range across all panels in a row. False restores
        per-panel autoscaling.
    show : bool
        Display the figure after saving it.
    dpi : int
        Resolution of the saved figure.

    Returns
    -------
    out : str
        Path of the saved figure.
    """
    row = df.iloc[row_idx]
    sim = np.asarray(sim_maps[row_idx], dtype=float)
    if sim.ndim != 3:
        raise ValueError(f"sim_maps[{row_idx}] must be (Z,Y,X), got {sim.shape}")
    nz_sim, ny_sim, nx_sim = sim.shape
    half = crop_size // 2

    def px(name, axis):                        # df value -> stack pixels
        v = row[f"{name}_{axis}"]
        return v if axis == "Z" else v * scale

    tname = f"T{i+1}"
    points = {name: (px(name, "X"), px(name, "Y"), int(round(px(name, "Z"))))
              for name in (tname, "LG", "PRED")}
    colors = {tname: "lime", "LG": "cyan", "PRED": "magenta"}
    markers = {tname: "o", "LG": "s", "PRED": "^"}

    # --- where each panel sits in the stack ---------------------------------
    # Both rows are centred on LG: that is what the ROI was built around, so the
    # two panels share an origin and a marker sits at the same offset in each.
    lg_x, lg_y, lg_z = points["LG"]
    img_cx, img_cy = int(round(lg_x)), int(round(lg_y))
    img_x0, img_y0 = img_cx - half, img_cy - half

    if sim_origin is None:
        sim_z0 = roi_origin(lg_z, nz_sim, stack.shape[0])
        sim_y0 = roi_origin(round(lg_y), ny_sim, stack.shape[1])
        sim_x0 = roi_origin(round(lg_x), nx_sim, stack.shape[2])
    else:
        sim_z0, sim_y0, sim_x0 = (int(v) for v in sim_origin)

    # --- peak and centre of mass, in sim index space ------------------------
    max_zi, max_r, max_c = np.unravel_index(np.argmax(sim), sim.shape)

    w = np.clip(sim, 0, None)
    if w.sum() > 0:
        # same estimator as the optimiser, on a min-max normalised map
        norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        com, _ = soft_center_of_mass(norm[None], np.zeros(3),
                                     temperature=com_temperature,
                                     window=com_window)
        com_z, com_r, com_c = com[0]
        com_label = "soft centre of mass"
    else:
        # nothing positive in the map: fall back to the peak
        com_z, com_r, com_c = max_zi, max_r, max_c
        com_label = "centre of mass"
    com_zi = int(round(com_z))

    # --- which slices to draw ----------------------------------------------
    n_show = nz_sim if crop_size_z is None else int(min(crop_size_z, nz_sim))
    start = 0 if n_show >= nz_sim else int(
        np.clip(max_zi - n_show // 2, 0, nz_sim - n_show))
    z_indices = range(start, start + n_show)

    # One intensity scale across all panels in a row. Per-panel autoscaling
    # stretches a slice that holds nothing but noise to full contrast, which
    # reads as signal.
    svmin = svmax = ivmin = ivmax = None
    if share_scale:
        shown = sim[start:start + n_show]
        svmin, svmax = float(shown.min()), float(shown.max())
        z_abs_all = [sim_z0 + zi for zi in z_indices
                     if 0 <= sim_z0 + zi < stack.shape[0]]
        if z_abs_all:
            crops = np.stack([get_crop(stack[z], img_cx, img_cy, crop_size)
                              for z in z_abs_all])
            ivmin, ivmax = float(crops.min()), float(crops.max())

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6.4), squeeze=False)

    for col, zi in enumerate(z_indices):
        z_abs = sim_z0 + zi                       # absolute slice in the stack
        ax_img, ax_sim = axes[0, col], axes[1, col]

        if 0 <= z_abs < stack.shape[0]:
            ax_img.imshow(get_crop(stack[z_abs], img_cx, img_cy, crop_size),
                          cmap="gray", vmin=ivmin, vmax=ivmax)
        else:
            ax_img.set_facecolor("0.15")
        ax_img.set_title(f"z={z_abs}", fontsize=10)
        ax_img.axis("off")

        ax_sim.imshow(sim[zi], cmap="viridis", vmin=svmin, vmax=svmax)
        ax_sim.axis("off")

        for name, (x, y, z) in points.items():
            if z != z_abs:
                continue
            for ax, cx_, cy_ in ((ax_img, x - img_x0, y - img_y0),
                                 (ax_sim, x - sim_x0, y - sim_y0)):
                ax.scatter(cx_, cy_, c=colors[name], marker=markers[name],
                           s=60, edgecolors="black", linewidths=0.8, zorder=5)

        if zi == max_zi:
            for ax, cx_, cy_ in ((ax_sim, max_c, max_r),
                                 (ax_img, sim_x0 + max_c - img_x0,
                                  sim_y0 + max_r - img_y0)):
                ax.scatter(cx_, cy_, c="red", marker="*", s=150,
                           edgecolors="white", linewidths=0.8, zorder=6)
        if zi == com_zi:
            for ax, cx_, cy_ in ((ax_sim, com_c, com_r),
                                 (ax_img, sim_x0 + com_c - img_x0,
                                  sim_y0 + com_r - img_y0)):
                ax.scatter(cx_, cy_, c="yellow", marker="X", s=120,
                           edgecolors="black", linewidths=0.8, zorder=6)

    # axis("off") also hides ylabels, so label the rows on the figure instead
    for ax, text in ((axes[0, 0], "crop"), (axes[1, 0], "similarity")):
        ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=11, rotation=90)

    handles = [Line2D([0], [0], marker=markers[n], color="w",
                      markerfacecolor=colors[n], markeredgecolor="black",
                      markersize=8, label=n) for n in points]
    handles += [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="white", markersize=12, label="max similarity"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=10, label=com_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.suptitle(f"TRACK_ID={row['TRACK_ID']}   "
                 f"DISTANCE={row['DISTANCE']:.3f}   "
                 f"slices {sim_z0 + start}-{sim_z0 + start + n_show - 1} "
                 f"of {stack.shape[0]}")
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.96))

    out = f"{path}T{i+1}_TRACKID{row['TRACK_ID']}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out
