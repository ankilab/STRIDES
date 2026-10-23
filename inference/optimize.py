"""
Landmark point-cloud refinement.

Given landmarks in image A, a coarse guess for their locations in image B, and a
local similarity volume per landmark, refine the locations in B by minimising

    E(d) = lambda_sim   * data(d)      # agree with the similarity maps
         + lambda_shape * shape(d)     # preserve the geometry of the A point set
         + lambda_anchor* anchor(d)    # stay near the coarse guess

Two things make the weights usable. The terms are dimensionless (the data term
is a normalised negative log-likelihood, the shape term is measured in
multiples of an explicit distance tolerance, the anchor in fractions of the
offset bound), and they are auto-balanced so that a one-voxel displacement
costs about the same in each -- without that the data term is diluted by 1/N
while the shape term is not, and lambda_sim has to be ~100x lambda_shape just
to be heard.

The objective supplies an exact analytic gradient, so L-BFGS-B needs one
evaluation per iteration instead of 3N+1, and the line search sees a value and
a derivative that actually agree. Both come from the same cubic B-spline basis;
``test_point_cloud_opt.py`` verifies them against finite differences.
"""

from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter, spline_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

__all__ = ["optimize_point_cloud", "build_knn_edges", "BSplineStack",
           "soft_center_of_mass", "roi_centers", "effective_roi_size",
           "prepare_maps_for_sweep"]


# ----------------------------------------------------------------------------
# interpolation: cubic B-spline, value and exact gradient from the same basis
# ----------------------------------------------------------------------------

class BSplineStack:
    """Cubic B-spline interpolation of a stack of 3D volumes, one per point.

    ``volumes[i]`` is sampled only at point ``i``. Both the value and the
    gradient come from the same basis functions, so they are exactly consistent
    -- unlike sampling ``map_coordinates`` for the value and ``np.gradient`` for
    the derivative, which makes the L-BFGS-B line search unreliable.

    Taps that fall outside the volume are clamped (replicate boundary), so the
    cost flattens outside the ROI instead of stepping off a cliff.
    """

    def __init__(self, volumes):
        vol = np.ascontiguousarray(volumes, dtype=np.float64)
        if vol.ndim != 4:
            raise ValueError(f"volumes must be (N,Z,Y,X), got {vol.shape}")
        coeff = vol
        for axis in (1, 2, 3):
            coeff = spline_filter1d(coeff, order=3, axis=axis, mode="mirror")
        self.coeff = coeff
        self.n, *self.shape = vol.shape
        self._pid = np.arange(self.n)
        self._tap = np.arange(4)[:, None]

    @staticmethod
    def _basis(t):
        """Cubic B-spline weights and their derivatives for the 4 taps."""
        t2 = t * t
        t3 = t2 * t
        u = 1.0 - t
        w = np.stack([
            u * u * u / 6.0,
            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
            (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
            t3 / 6.0,
        ])
        dw = np.stack([
            -0.5 * u * u,
            0.5 * (3.0 * t2 - 4.0 * t),
            0.5 * (-3.0 * t2 + 2.0 * t + 1.0),
            0.5 * t2,
        ])
        return w, dw

    def __call__(self, pos):
        """pos: (N,3) voxel coordinates. Returns value (N,) and gradient (N,3)."""
        pos = np.asarray(pos, dtype=float)
        base = np.floor(pos).astype(np.int64)
        frac = pos - base

        w, dw, idx = [], [], []
        for a in range(3):
            wa, dwa = self._basis(frac[:, a])
            w.append(wa)
            dw.append(dwa)
            idx.append(np.clip(base[:, a][None, :] - 1 + self._tap,
                               0, self.shape[a] - 1))

        # (4,4,4,N) block of spline coefficients around each point
        c = self.coeff[self._pid,
                       idx[0][:, None, None, :],
                       idx[1][None, :, None, :],
                       idx[2][None, None, :, :]]

        wz, wy, wx = w
        dz, dy, dx = dw
        val = np.einsum("an,bn,cn,abcn->n", wz, wy, wx, c, optimize=True)
        grad = np.stack([
            np.einsum("an,bn,cn,abcn->n", dz, wy, wx, c, optimize=True),
            np.einsum("an,bn,cn,abcn->n", wz, dy, wx, c, optimize=True),
            np.einsum("an,bn,cn,abcn->n", wz, wy, dx, c, optimize=True),
        ], axis=1)
        return val, grad


# ----------------------------------------------------------------------------
# geometry graph
# ----------------------------------------------------------------------------

def build_knn_edges(points, k=8):
    """Undirected k-nearest-neighbour edges, preserving local structure only."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    k = int(min(k, n - 1))
    if k < 1:
        return np.zeros((0, 2), dtype=int)

    _, idx = cKDTree(points).query(points, k=k + 1)
    idx = np.atleast_2d(idx)

    edges = set()
    for i in range(n):
        for j in idx[i, 1:]:
            j = int(j)
            if j < n and j != i:
                edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=int)


# ----------------------------------------------------------------------------
# map preparation
# ----------------------------------------------------------------------------

def _prepare_maps(likelihood_maps, eps, max_neglog, higher_is_better):
    """Similarity volumes -> clipped negative-log cost volumes + confidences."""
    maps = np.asarray(likelihood_maps, dtype=np.float64)
    if not higher_is_better:
        maps = -maps

    lo = maps.min(axis=(1, 2, 3), keepdims=True)
    hi = maps.max(axis=(1, 2, 3), keepdims=True)
    maps = (maps - lo) / (hi - lo + eps)

    # Confidence: a sharp, unimodal map has most of its mass near zero after
    # normalisation; a broad or noisy one has a high median. Landmarks whose
    # map is uninformative should defer to the geometry term.
    conf = 1.0 - np.median(maps, axis=(1, 2, 3))
    # A perfectly flat map normalises to all-zeros, which would otherwise score
    # as maximally confident. It carries no information at all.
    degenerate = (hi - lo).reshape(-1) <= 1e-12 * np.maximum(np.abs(hi).reshape(-1), 1.0)
    conf[degenerate] = 0.0
    conf = np.clip(conf, 1e-3, None)
    conf = conf / conf.mean()
    conf = np.clip(conf, 0.05, 20.0)      # no single landmark may dominate
    conf = conf / conf.mean()

    neglog = np.minimum(-np.log(maps + eps), max_neglog)
    return maps, neglog, conf, neglog.min(axis=(1, 2, 3))


def roi_centers(points0, roi_size, volume_shape, keep_size=True):
    """Where each landmark actually sits inside its own ROI, in voxels.

    Mirrors the window placement of ``roi_bounds``: with ``keep_size=True`` a
    window that would hang off the volume is SHIFTED back inside rather than
    cropped, so every ROI keeps the same shape -- but the landmark is then no
    longer at the centre of its own ROI. Feed the result to
    ``optimize_point_cloud(roi_center=...)``.

    This matters for more than the border landmarks. The window origin is
    ``int(c) - r // 2``, so:

    * a landmark with a fractional coordinate sits ``frac(c)`` past the centre,
      up to a whole voxel of bias, on every landmark;
    * an EVEN ``roi_size`` puts the landmark at ``r//2`` while the array centre
      is ``(r-1)/2`` -- a constant half-voxel offset everywhere;
    * a shifted window is off by however much the shift took, which is what
      produces negative coordinates near a face.

    ``roi_size`` must be the size you REQUESTED from ``roi_bounds``, not the
    shape of the maps you got back -- the two differ on an axis where the stack
    is thinner than the requested ROI, and the window origin is computed from
    the requested size. On such an axis the ROI covers the whole stack, which
    is well defined and uniform; ``effective_roi_size`` reports it.

    If you already have the bounds, this is just
    ``points0 - [[lo for lo, hi in b] for b in bounds]``.
    """
    p = np.asarray(points0, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points0 must be (N,3), got {p.shape}")
    r = np.broadcast_to(np.asarray(roi_size, dtype=int), (3,)).astype(int)
    n = np.asarray(volume_shape, dtype=int).reshape(3)
    if (r < 1).any() or (n < 1).any():
        raise ValueError(f"roi_size {tuple(r)} and volume_shape {tuple(n)} "
                         "must both be positive on every axis")

    # An axis where the requested ROI is larger than the stack (a 7-slice stack
    # with roi_size_z=8) is fine: roi_bounds clamps the window to the whole
    # axis, every ROI is still the same shape, and the landmark's index is just
    # its own coordinate. The clamp below reproduces that -- max(0, n - r) is 0.
    lo = np.floor(p).astype(np.int64) - (r // 2)
    if keep_size:
        lo = np.minimum(np.maximum(lo, 0), np.maximum(0, n - r))
    else:
        lo = np.maximum(lo, 0)

    centres = p - lo
    eff = np.minimum(r, n)
    # One voxel of slack: trackers differ on whether a Z=7 stack spans indices
    # [0, 6] or extent [0, 7), so z=6.9 is legitimate. A units mix-up is off by
    # a factor, not by half a voxel, so this still catches it.
    tol = 1.0
    off = (centres < -tol) | (centres > eff - 1 + tol)
    if off.any():
        bad = np.flatnonzero(off.any(1))
        raise ValueError(
            f"landmark(s) {bad[:5].tolist()} fall outside their own ROI by "
            f"more than {tol} voxel -- points0 is probably not in voxels of a "
            f"{tuple(int(v) for v in n)} volume")
    return centres


def effective_roi_size(roi_size, volume_shape):
    """The ROI shape ``roi_bounds`` actually returns: elementwise min of the
    requested size and the stack, since a window cannot exceed the axis."""
    r = np.broadcast_to(np.asarray(roi_size, dtype=int), (3,))
    n = np.asarray(volume_shape, dtype=int).reshape(3)
    return tuple(int(v) for v in np.minimum(r, n))


def soft_center_of_mass(maps, roi_center, temperature=0.08, window=5,
                        spread_floor=0.5):
    """Per-map soft-argmax: the mass-weighted centre of the dominant peak.

    ``maps`` must already be normalised to [0,1]. Weights are
    ``exp((m - max) / temperature)``, so ``temperature`` interpolates between
    the hard argmax (-> 0) and the plain centre of mass (-> inf). Plain centre
    of mass over the whole ROI is not useful: with tens of thousands of
    background voxels their combined mass dominates any peak and the estimate
    collapses towards the ROI centre. ``window`` therefore restricts the
    average to a cube of that half-width around each map's argmax, which is
    what makes the estimate local rather than global.

    Returns
    -------
    target : (N,3)
        Offset from ``roi_center``, in voxels.
    spread : (N,3)
        Per-axis standard deviation of the same weighted distribution -- the
        width of the peak, i.e. how precisely that landmark is localised.
        Floored at ``spread_floor`` so a one-voxel spike cannot claim infinite
        precision.
    """
    maps = np.asarray(maps, dtype=float)
    n = maps.shape[0]
    shape = maps.shape[1:]
    axes = np.meshgrid(*[np.arange(s, dtype=float) for s in shape],
                       indexing="ij")

    centres = np.asarray(roi_center, dtype=float)
    if centres.ndim == 1:
        centres = np.broadcast_to(centres, (n, 3))

    target = np.empty((n, 3))
    spread = np.empty((n, 3))
    for i in range(n):
        m = maps[i]
        if window:
            peak = np.unravel_index(np.argmax(m), shape)
            sl = tuple(slice(max(0, p - window), min(s, p + window + 1))
                       for p, s in zip(peak, shape))
            sub = m[sl]
            sub_axes = [a[sl] for a in axes]
        else:
            sub, sub_axes = m, axes

        w = np.exp((sub - sub.max()) / temperature)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:      # degenerate map
            target[i] = 0.0
            spread[i] = np.max(shape)
            continue
        w = w / total
        t = np.array([float((w * a).sum()) for a in sub_axes])
        v = np.array([float((w * (a - ti) ** 2).sum())
                      for a, ti in zip(sub_axes, t)])
        target[i] = t - centres[i]
        spread[i] = np.sqrt(np.maximum(v, 0.0))

    return target, np.maximum(spread, spread_floor)


def prepare_maps_for_sweep(likelihood_maps, roi_center=None, data_mode="com",
                           com_temperature=0.08, com_window=32,
                           higher_is_better=True, max_neglog=8.0, eps=1e-8):
    """Precompute everything in ``optimize_point_cloud`` that the weights do
    not affect: map normalisation, the negative-log volumes, the per-landmark
    confidences, and the soft centre-of-mass targets.

    Pass the result as ``prepared=`` to sweep lambdas over the same data. The
    ``roi_center`` you give here must match the one you pass to the solver.
    """
    maps_norm, neglog, conf, best_cost = _prepare_maps(
        likelihood_maps, eps, max_neglog, higher_is_better)
    out = {"maps_norm": maps_norm, "neglog": neglog, "conf": conf,
           "best_cost": best_cost, "key": (max_neglog, higher_is_better, eps),
           "com": None, "com_key": (com_temperature, com_window)}

    if data_mode != "gradient":
        n, nz, ny, nx = neglog.shape
        size = np.array([nz - 1.0, ny - 1.0, nx - 1.0])
        rc = (np.broadcast_to(size / 2.0, (n, 3)) if roi_center is None
              else np.broadcast_to(np.asarray(roi_center, float), (n, 3)))
        rc = np.clip(rc, 0.0, size)
        out["com"] = soft_center_of_mass(maps_norm, rc,
                                         temperature=com_temperature,
                                         window=com_window)
    return out


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------

def optimize_point_cloud(
    points_A,
    points0,
    likelihood_maps,
    *,
    spacing=(1.0, 1.0, 1.0),
    lambda_sim=1.0,
    lambda_shape=1.0,
    lambda_anchor=0.05,
    max_delta=20.0,
    use_all_pairs=True,
    k_edges=8,
    shape_tol=2.0,
    allow_scale=False,
    max_log_scale=0.25,
    auto_balance=True,
    balance_probe=1.0,
    confidence="contrast",
    higher_is_better=True,
    max_neglog=8.0,
    tension_flag=0.15,
    roi_center=None,
    volume_shape=None,
    edge_margin=2.0,
    data_mode="com",
    com_temperature=0.08,
    com_window=32,
    smooth_schedule=(1.5, 0.0),
    delta0=None,
    maxiter=300,
    eps=1e-8,
    prepared=None,
    verbose=False,
):
    """Refine landmark locations in image B.

    Parameters
    ----------
    points_A : (N,3)
        Landmarks in image A, in (z,y,x) voxel indices.
    points0 : (N,3)
        Coarse predicted locations in image B, in (z,y,x) voxel indices. Each
        similarity volume is assumed to be centred on the corresponding point.
    likelihood_maps : (N,Z,Y,X)
        Local similarity volume per landmark. Higher = better match unless
        ``higher_is_better=False``.
    spacing : (3,)
        Physical voxel size (z,y,x). Distances in the shape term are computed in
        physical units, which matters whenever the data is anisotropic.
    lambda_sim, lambda_shape, lambda_anchor : float
        Relative weights of the three terms. With ``auto_balance=True`` these
        are directly comparable, so 1 / 1 / 0.05 is a reasonable starting
        point and the result is not razor-sensitive to them. The anchor should
        stay small: its real job is to break the translation and rotation null
        space of the shape term, not to hold the points in place.
    data_mode : {"com", "gradient", "com_init"}
        How the similarity maps drive the points. ``"com"`` is the default: it
        won on synthetic noise benchmarks and on real data.

        ``"gradient"`` slides each point downhill on the interpolated
        -log map, so it lands on the *nearest local* maximum and depends on
        ``smooth_schedule`` to widen the basin. Sub-voxel accurate, but it
        chases noise and can stall on the clipped plateau of a sharply peaked
        map.

        ``"com"`` precomputes each map's soft centre of mass (see
        ``soft_center_of_mass``) and pulls the point towards it with a
        quadratic, inverse-variance-weighted cost. The window is centred on the
        map's *global* argmax, so there is no nearest-peak trap and no
        smoothing schedule; the data term becomes convex, and averaging over
        the peak suppresses noise instead of following it. On noisy synthetic
        maps this halves the error (see the test suite). The cost is that it
        commits to one peak per landmark up front: if the global argmax is
        wrong, only the geometry term can pull that landmark back.

        ``"com_init"`` uses the centre of mass as the starting point and then
        runs the gradient mode -- a middle ground that keeps sub-voxel
        refinement but still follows the local surface.
    com_temperature : float
        Softmax temperature on the [0,1]-normalised map. Towards 0 this
        approaches the hard argmax; larger values average more of the peak in.
    com_window : int
        Half-width in voxels of the cube around the argmax that the average is
        taken over. Roughly the peak radius: 3-5 is insensitive over a wide
        range of peak widths, below 2 biases towards the argmax, and large
        values let background and neighbouring structure back in.
    roi_center : (3,) or (N,3), optional
        Where ``points0[k]`` sits inside ``likelihood_maps[k]``, in voxels.
        Defaults to the geometric centre of the volume, which is WRONG for any
        ROI whose window was shifted or cropped to fit the image, and off by
        ``frac(c)`` (or half a voxel, for an even ROI size) even for interior
        landmarks. Use ``roi_centers()`` to compute it. The offset bounds become
        asymmetric to match, which also keeps the result inside the image.
    volume_shape : (3,), optional
        (Z, Y, X) of the full image B, in voxels. Pass it whenever landmarks
        can sit near a face of the volume. The offset bounds are otherwise
        relative to the ROI only, and a ROI centred on a landmark near the top
        of a thin stack extends past it -- so the optimiser can legitimately
        return a negative absolute coordinate. With this set, the returned
        points are guaranteed to lie inside the image.
    max_delta : float or (3,)
        Bound on the offset per axis, in voxels. Intersected with the ROI
        extent (and with ``volume_shape`` when given), so the search leaves
        neither the similarity volume nor the image.
    use_all_pairs : bool
        True preserves global shape (O(N^2) pairs); False uses a kNN graph and
        preserves local structure only, which tolerates non-rigid deformation.
    shape_tol : float
        Pairwise distance error you are willing to accept, in physical units.
        The shape residual is measured in multiples of this, so ``shape_tol``
        sets the scale of the term and ``lambda_shape`` sets its weight
        relative to the other two. Set it to the accuracy you actually expect
        from the landmarks; too small and the geometry term swamps the data.
    k_edges : int
        Neighbours per landmark in the kNN graph. Only used when
        ``use_all_pairs=False``.
    allow_scale : bool
        Fit a global scale factor between the two point sets, instead of
        demanding exact isometry. Use when A and B differ in size or spacing.
    max_log_scale : float
        Bound on ``|log(scale)|`` when ``allow_scale=True``.
    auto_balance : bool
        Rescale the three terms so that a ``balance_probe``-voxel displacement
        costs about the same in each. This is what makes the lambdas
        comparable; with it off you are back to hand-tuning across orders of
        magnitude. The applied factors are reported in ``info["term_scale"]``.
    balance_probe : float
        Standard deviation, in voxels, of the random displacement used to
        measure the gradient of each term for ``auto_balance``.
    confidence : {"contrast", "none"} or (N,) array
        Per-landmark weight on the data term. ``"contrast"`` derives it from
        the maps (a sharp, unimodal map counts more than a broad or noisy
        one), ``"none"`` weights all landmarks equally, and an array is used
        as given after normalising it to mean 1.
    higher_is_better : bool
        True if the maps are similarities, False if they are costs.
    max_neglog : float
        Upper clip of the negative-log cost volumes. Also normalises the data
        term of the gradient modes.
    tension_flag : float
        Fraction of ``max_neglog`` above which a landmark is ``flagged`` in
        the gradient modes. In ``"com"`` mode the threshold is fixed at three
        peak widths.
    edge_margin : float
        Distance in voxels from a face of the ROI within which a map's maximum
        counts as ``peak_at_edge``.
    smooth_schedule : sequence of float
        Gaussian sigmas (voxels) applied to the cost volumes, optimised in
        order. Coarse-to-fine; a first pass at sigma>0 avoids local minima.
        Use ``(0.0,)`` for a single unsmoothed pass.
    delta0 : (N,3), optional
        Warm start.
    maxiter : int
        L-BFGS-B iterations per pass of the schedule.
    eps : float
        Numerical floor in the map normalisation and the pair distances.
    prepared : dict, optional
        Output of ``prepare_maps_for_sweep``. Everything in it depends only on
        the similarity maps, never on the weights, so a parameter search that
        reuses it skips ~40% of each call. Results are identical -- the test
        suite checks that.
    verbose : bool
        Print the objective value and iteration count of every pass.

    Returns
    -------
    q_opt : (N,3)
        Refined landmark coordinates in image B.
    info : dict
        ``delta``, ``scale``, ``cost`` (per-term breakdown), ``confidence``,
        ``pair_residual`` / ``pair_rms``, ``term_scale``, and the raw scipy
        ``result``. The two worth actually looking at:

        ``tension`` (N,)
            How much similarity cost each landmark pays at the solution versus
            the best it could have reached in its own ROI. Near zero for a
            landmark whose map and geometry agree; large when the geometry
            overruled the map. This is the outlier detector -- a wrong map, a
            landmark misannotated in A, and a landmark that genuinely moved in
            B all show up here.
        ``flagged`` (N,) bool
            ``tension`` above ``tension_flag``. Inspect these by hand rather
            than trusting them.

        ``peak_at_edge`` (N,) bool
            The map's own maximum lies within ``edge_margin`` voxels of a face
            of its ROI. The true match is then most likely OUTSIDE the volume
            you extracted, and no setting will recover it -- extract a larger
            ROI, or re-centre on this result and run a second pass. Check this
            first whenever a landmark looks like it left the ROI.

        ``at_bound`` marks landmarks pinned against the offset bound, which
        means ``max_delta`` is too small for a ROI that does have room. In
        ``com`` mode the target stops short of the boundary by construction, so
        ``peak_at_edge`` is the reliable indicator there and this one rarely
        fires.

        The remaining entries are intermediate quantities kept for
        inspection: ``point_cost``, ``com_target`` / ``com_spread``,
        ``peak_index``, ``bound_lo`` / ``bound_hi``, ``pairs``,
        ``mean_pair_distance`` and ``success``.

    Notes
    -----
    A robust (Huber) pairwise loss was tried and removed: because the optimiser
    resolves a conflicting landmark by *not moving it* rather than by leaving a
    large distance residual, the outlying pairs are never large enough to be
    down-weighted, and the softened tail only weakened the fit. Use ``tension``
    to find bad landmarks, and the kNN graph (``use_all_pairs=False``) when the
    deformation is genuinely non-rigid.
    """
    points_A = np.asarray(points_A, dtype=float)
    points0 = np.asarray(points0, dtype=float)
    spacing = np.asarray(spacing, dtype=float).reshape(3)

    n = len(points_A)
    if points0.shape != (n, 3) or points_A.shape != (n, 3):
        raise ValueError("points_A and points0 must both be (N,3)")
    if np.asarray(likelihood_maps).shape[0] != n:
        raise ValueError("likelihood_maps must have one volume per landmark")

    if data_mode not in ("gradient", "com", "com_init"):
        raise ValueError(f"unknown data_mode {data_mode!r}")

    if prepared is None:
        maps_norm, neglog, conf, best_cost = _prepare_maps(
            likelihood_maps, eps, max_neglog, higher_is_better)
    else:
        if prepared.get("key") != (max_neglog, higher_is_better, eps):
            raise ValueError(
                "prepared was built with different map settings "
                f"{prepared.get('key')} than this call "
                f"{(max_neglog, higher_is_better, eps)}")
        maps_norm = prepared["maps_norm"]
        neglog = prepared["neglog"]
        conf = prepared["conf"]
        best_cost = prepared["best_cost"]
        if len(neglog) != n:
            raise ValueError("prepared holds a different number of landmarks")
    _, nz, ny, nx = neglog.shape

    if isinstance(confidence, str):
        if confidence == "none":
            conf = np.ones(n)
        elif confidence != "contrast":
            raise ValueError(f"unknown confidence mode {confidence!r}")
    else:
        conf = np.asarray(confidence, dtype=float).reshape(n)
        conf = np.clip(conf, 1e-3, None)
        conf = conf / conf.mean()

    # ---- ROI centre and per-axis bounds ------------------------------------
    size = np.array([nz - 1.0, ny - 1.0, nx - 1.0])
    if roi_center is None:
        roi_center = np.broadcast_to(size / 2.0, (n, 3)).astype(float)
    else:
        roi_center = np.asarray(roi_center, dtype=float)
        if roi_center.shape == (3,):
            roi_center = np.broadcast_to(roi_center, (n, 3)).astype(float)
        if roi_center.shape != (n, 3):
            raise ValueError(f"roi_center must be (3,) or (N,3), got "
                             f"{roi_center.shape}")
        if (roi_center < -1.0).any() or (roi_center > size + 1.0).any():
            bad = np.flatnonzero(((roi_center < -1.0)
                                  | (roi_center > size + 1.0)).any(1))
            raise ValueError(
                f"roi_center lies outside its own volume for landmark(s) "
                f"{bad[:5].tolist()}: the maps are {(nz, ny, nx)} but the "
                "centres do not fall inside them")
        roi_center = np.clip(roi_center, 0.0, size)

    # Asymmetric per-landmark bounds: a ROI clipped against the image border is
    # not centred on its landmark, so the reachable offset differs by side.
    md = np.broadcast_to(np.asarray(max_delta, float), (3,))
    lo = np.maximum(-md, -roi_center)
    hi = np.minimum(md, size - roi_center)

    # The bounds above are relative to the ROI, which knows nothing about where
    # the image ends. A landmark near a face of the volume sits in a ROI that
    # sticks out past it, and the optimiser will happily return a NEGATIVE
    # absolute coordinate. Intersect with the volume when we are told its size.
    if volume_shape is not None:
        vol_max = np.asarray(volume_shape, dtype=float).reshape(3) - 1.0
        off = (points0 < -1.0) | (points0 > vol_max + 1.0)   # 1 voxel of slack
        if off.any():
            bad = np.flatnonzero(off.any(1))
            raise ValueError(
                f"points0 lies outside volume_shape for landmark(s) "
                f"{bad[:5].tolist()} -- check that points0 is in voxels of the "
                "same volume, not physical units")
        lo = np.maximum(lo, -points0)
        hi = np.minimum(hi, vol_max - points0)
    if (hi < lo - 1e-9).any():
        raise ValueError("empty offset bounds -- roi_center and volume_shape "
                         "disagree about where the landmarks are")
    hi = np.maximum(hi, lo)
    span = np.maximum((hi - lo) / 2.0, 1e-3)
    anchor_scale = float(max(span.mean(), 1e-3))

    # ---- soft centre-of-mass targets ---------------------------------------
    if data_mode == "gradient":
        com_target = com_spread = None
    elif prepared is not None and prepared.get("com") is not None:
        if prepared["com_key"] != (com_temperature, com_window):
            raise ValueError("prepared holds a soft-COM computed with "
                             f"{prepared['com_key']}, not "
                             f"{(com_temperature, com_window)}")
        com_target, com_spread = prepared["com"]
        com_target = np.clip(com_target, lo, hi)
    else:
        com_target, com_spread = soft_center_of_mass(
            maps_norm, roi_center, temperature=com_temperature,
            window=com_window)
        com_target = np.clip(com_target, lo, hi)

    # ---- geometry graph ----------------------------------------------------
    if use_all_pairs:
        pairs = np.array(list(combinations(range(n), 2)), dtype=int)
    else:
        pairs = build_knn_edges(points_A, k=k_edges)
    if len(pairs) == 0:
        raise ValueError("no pairs in the geometry graph")

    pi, pj = pairs[:, 0], pairs[:, 1]
    d_a = np.linalg.norm((points_A[pi] - points_A[pj]) * spacing, axis=1)
    d_mean = float(d_a.mean())
    n_pairs = len(pairs)
    tol = float(shape_tol)
    if tol <= 0:
        raise ValueError("shape_tol must be positive")

    # ---- the three terms, evaluated separately -----------------------------
    def make_terms(sampler):
        """f(x) -> ((data, g), (shape, g, d/dlog_s), (anchor, g))."""

        def terms(x):
            delta = x[: 3 * n].reshape(n, 3)
            log_s = x[3 * n] if allow_scale else 0.0
            scale = np.exp(log_s)

            # data ------------------------------------------------------------
            if sampler is None:
                # Quadratic pull towards each map's soft centre of mass,
                # weighted by the inverse variance of that estimate. Convex:
                # no local minima, and no smoothing schedule needed.
                z = (delta - com_target) / com_spread
                data = float(np.mean(conf[:, None] * z ** 2))
                g_data = (2.0 / (3.0 * n)) * conf[:, None] * z / com_spread
            else:
                val, dval = sampler(roi_center + delta)
                data = float(np.mean(conf * val)) / max_neglog
                g_data = (conf[:, None] * dval) / (n * max_neglog)

            # shape -----------------------------------------------------------
            q = points0 + delta
            diff = (q[pi] - q[pj]) * spacing
            d_b = np.sqrt(np.einsum("ij,ij->i", diff, diff) + eps)
            resid = (scale * d_a - d_b) / tol

            shape = float(np.mean(0.5 * resid ** 2))
            gp = (-(resid / (n_pairs * tol * d_b)))[:, None] * diff * spacing
            g_shape = np.zeros_like(q)
            np.add.at(g_shape, pi, gp)
            np.add.at(g_shape, pj, -gp)
            g_shape_logs = float(np.mean(resid * scale * d_a / tol))

            # anchor ----------------------------------------------------------
            anchor = float(np.mean((delta / anchor_scale) ** 2))
            g_anchor = 2.0 * delta / (3.0 * n * anchor_scale ** 2)

            return ((data, g_data), (shape, g_shape, g_shape_logs),
                    (anchor, g_anchor))

        return terms

    def balance_scales(terms, x_ref):
        """Rescale the terms so that a `balance_probe`-voxel displacement costs
        about the same in each of them.

        Without this the weights are not comparable: the data term is diluted
        by 1/N (one landmark is one of N summands) while the shape term is not
        (that landmark's error shows up in all of its pairs at once), so
        lambda_sim would have to be ~100x lambda_shape just to be heard.
        """
        if not auto_balance:
            return (1.0, 1.0, 1.0)
        rng = np.random.default_rng(0)
        acc = np.zeros(3)
        for _ in range(4):
            x = np.array(x_ref, dtype=float, copy=True)
            probe = x[: 3 * n].reshape(n, 3) + rng.normal(0.0, balance_probe,
                                                          size=(n, 3))
            x[: 3 * n] = np.clip(probe, lo, hi).ravel()
            (_, gd), (_, gs, _), (_, ga) = terms(x)
            acc += (np.linalg.norm(gd), np.linalg.norm(gs), np.linalg.norm(ga))
        acc /= 4.0
        ref = float(acc.max())
        floor = ref * 1e-3
        return tuple(ref / max(float(v), floor) for v in acc)

    def make_objective(terms, scales):
        w_d = lambda_sim * scales[0]
        w_s = lambda_shape * scales[1]
        w_a = lambda_anchor * scales[2]

        def fun(x):
            (dv, gd), (sv, gs, gs_logs), (av, ga) = terms(x)
            total = w_d * dv + w_s * sv + w_a * av
            grad = (w_d * gd + w_s * gs + w_a * ga).ravel()
            if allow_scale:
                grad = np.concatenate([grad, [w_s * gs_logs]])
            return total, grad

        return fun

    # ---- initialisation and bounds ----------------------------------------
    if delta0 is None:
        x0 = np.zeros(3 * n + (1 if allow_scale else 0))
    else:
        d0 = np.clip(np.asarray(delta0, float).reshape(n, 3), lo, hi)
        x0 = np.concatenate([d0.ravel(), [0.0]]) if allow_scale else d0.ravel()

    bounds = [(lo[k, a], hi[k, a]) for k in range(n) for a in range(3)]
    if allow_scale:
        bounds.append((-max_log_scale, max_log_scale))

    # ---- coarse-to-fine optimisation --------------------------------------
    if data_mode == "com_init":                 # COM as the starting point only
        x0[: 3 * n] = com_target.ravel()

    schedule = (None,) if data_mode == "com" else (smooth_schedule or (0.0,))

    result = None
    scales = None
    for sigma in schedule:
        if sigma is None:
            terms = make_terms(None)            # convex quadratic data term
        else:
            vol = neglog if sigma <= 0 else gaussian_filter(
                neglog, sigma=(0.0, sigma, sigma, sigma), mode="nearest")
            terms = make_terms(BSplineStack(vol))
        if scales is None:                      # balance once, on the guess
            scales = balance_scales(terms, x0)
        result = minimize(
            make_objective(terms, scales), x0,
            jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": maxiter, "maxfun": 10 * maxiter},
        )
        x0 = result.x
        if verbose:
            tag = "com " if sigma is None else f"sig={sigma:4.2f}"
            print(f"{tag}  f={result.fun:.6f}  nit={result.nit}  "
                  f"{result.message}")

    delta = result.x[: 3 * n].reshape(n, 3)
    log_s = float(result.x[3 * n]) if allow_scale else 0.0
    q_opt = points0 + delta

    # ---- diagnostics -------------------------------------------------------
    final = BSplineStack(neglog)
    point_cost, _ = final(roi_center + delta)
    diff = (q_opt[pi] - q_opt[pj]) * spacing
    d_b = np.linalg.norm(diff, axis=1)
    resid = np.exp(log_s) * d_a - d_b

    # Does each map's peak sit against a face of its own ROI? If so the true
    # optimum is probably outside the volume you extracted, and no amount of
    # optimising inside it will find the right answer. This is the signal that
    # "the point wanted to leave the ROI" -- more reliable than at_bound in
    # com mode, where the target stops short of the boundary by construction.
    peak_idx = np.array([np.unravel_index(np.argmax(m), m.shape)
                         for m in maps_norm], dtype=float)
    peak_at_edge = ((peak_idx < edge_margin)
                    | (peak_idx > size - edge_margin)).any(axis=1)

    # "tension": how much similarity cost each landmark is paying at the
    # solution, versus the best it could have reached inside its own ROI. High
    # tension means the geometry term overruled that landmark's map -- either
    # the map is wrong, the landmark is misannotated in A, or it genuinely
    # moved. These are the ones to inspect; the value is near zero otherwise.
    if data_mode == "com":
        # How many peak-widths the geometry pulled each landmark away from its
        # own centre-of-mass estimate. Same role as the neglog tension below.
        tension = np.linalg.norm((delta - com_target) / com_spread, axis=1)
        flagged = tension > 3.0
    else:
        tension = point_cost - best_cost
        flagged = tension > tension_flag * max_neglog

    info = {
        "delta": delta,
        "scale": float(np.exp(log_s)),
        "point_cost": point_cost,
        "tension": tension,
        "flagged": flagged,
        "confidence": conf,
        "com_target": com_target,
        "com_spread": com_spread,
        "pair_residual": resid,
        "pair_rms": float(np.sqrt(np.mean(resid ** 2))),
        "mean_pair_distance": d_mean,
        "at_bound": np.any((delta - lo < 1e-6) | (hi - delta < 1e-6), axis=1),
        "peak_at_edge": peak_at_edge,
        "peak_index": peak_idx,
        "bound_lo": lo,
        "bound_hi": hi,
        "pairs": pairs,
        "term_scale": {"data": scales[0], "shape": scales[1],
                       "anchor": scales[2]},
        "success": bool(result.success),
        "result": result,
    }
    info["cost"] = {
        "data": float(np.mean(conf * point_cost)) / max_neglog,
        "shape": float(np.mean(0.5 * (resid / tol) ** 2)),
        "anchor": float(np.mean((delta / anchor_scale) ** 2)),
        "total": float(result.fun),
    }
    return q_opt, info