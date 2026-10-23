"""
Graphical front end of the landmark-transfer pipeline.

Workflow
--------
1. Open a 4D TIFF (T, Z, Y, X). Timepoints 0 and 1 are shown side by side.
2. Mark the query points in the left stack, or load them from a CSV.
3. Run the inference. The re-identified points appear in the right stack and
   can be corrected by hand.
4. "Save & next timepoint" writes the tracks to a CSV and moves on to the next
   pair of timepoints, with the re-identified points as the new query points.

Mouse, in both stacks: click selects a point, dragging moves it, Shift + click
places the selected point on the slice shown, right click removes the point.

Start it with::

    python gui.py
    python gui.py --stack data/stack.tif --model checkpoints/spine_embedder.pth

The checkpoint, the DINOv3 repository and the voxel size are set under
"Settings" and remembered between sessions.
"""

import os

# PyTorch and OpenCV/NumPy can each bring their own OpenMP runtime on Windows,
# which aborts the interpreter on import. Must be set before they are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import traceback
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
except ImportError:                       # PyQt6 works as well
    from PyQt6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal

import matplotlib

matplotlib.use("QtAgg")
from matplotlib import patheffects
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

import run_inference as pipeline
from utils import DEFAULT_DINOV3_REPO

Qt = QtCore.Qt

TRACK_COLUMNS = ["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T"]

# colours of the overlays
COLOR_QUERY = "#39ff14"        # query points (left stack)
COLOR_RESULT = "#ff40ff"       # re-identified points (right stack)
COLOR_SELECTED = "#ffe600"
COLOR_LINK = "#00e5ff"         # line from a query point to its re-identified position
CANVAS_BACKGROUND = "#202020"

# a dark outline keeps rings and IDs readable on bright structures
RING_OUTLINE = [patheffects.withStroke(linewidth=3.2, foreground="black")]
TEXT_OUTLINE = [patheffects.withStroke(linewidth=2.8, foreground="black")]

NEAR_SLICES = 2                # points this many slices away are drawn faded
PICK_RADIUS_PX = 12            # screen distance within which a click hits a point
DRAG_START_PX = 4              # mouse travel before a click becomes a drag
UNDO_STEPS = 50

STAGE_TEXT = {
    "coarse_xy": "Matching the projections …",
    "z_registration": "Z-registration …",
    "embeddings": "Computing embeddings …",
    "refinement": "Refining the positions …",
}


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------

@dataclass
class Settings:
    """User settings; stored between sessions with QSettings."""

    model_path: str = ""
    dinov3_repo: str = str(DEFAULT_DINOV3_REPO)
    voxel_size_xy: float = 1.0       # label units per pixel
    voxel_size_z: float = 1.0        # label units per slice
    crop_size_z: int = 3
    roi_size_xy: int = 32
    roi_size_z: int = 9

    @classmethod
    def load(cls):
        store = QtCore.QSettings("STRIDES", "gui")
        settings = cls()
        for f in fields(cls):
            value = store.value(f.name)
            if value is not None:
                try:
                    setattr(settings, f.name, f.type(value))
                except (TypeError, ValueError):
                    pass
        return settings

    def save(self):
        store = QtCore.QSettings("STRIDES", "gui")
        for f in fields(self):
            store.setValue(f.name, getattr(self, f.name))


class SettingsDialog(QtWidgets.QDialog):
    """Paths of the model and the parameters of the pipeline."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)

        self.model_path = QtWidgets.QLineEdit(settings.model_path)
        self.dinov3_repo = QtWidgets.QLineEdit(settings.dinov3_repo)

        self.voxel_size_xy = self._float_box(settings.voxel_size_xy)
        self.voxel_size_z = self._float_box(settings.voxel_size_z)
        self.crop_size_z = self._int_box(settings.crop_size_z, 1, 99)
        self.roi_size_xy = self._int_box(settings.roi_size_xy, 2, 512)
        self.roi_size_z = self._int_box(settings.roi_size_z, 1, 99)

        form = QtWidgets.QFormLayout()
        form.addRow("Model checkpoint (.pth)",
                    self._with_browse(self.model_path, self._browse_model))
        form.addRow("DINOv3 repository",
                    self._with_browse(self.dinov3_repo, self._browse_repo))
        form.addRow("Pixel size x/y (label units)", self.voxel_size_xy)
        form.addRow("Slice distance z (label units)", self.voxel_size_z)
        form.addRow("Embedding crop, z-slices", self.crop_size_z)
        form.addRow("Search ROI x/y (pixels)", self.roi_size_xy)
        form.addRow("Search ROI z (slices)", self.roi_size_z)

        hint = QtWidgets.QLabel(
            "Leave both sizes at 1 if the label CSV is in pixels and slices. "
            "Set them before loading labels.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    @staticmethod
    def _float_box(value):
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(4)
        box.setRange(0.0001, 1000.0)
        box.setValue(value)
        return box

    @staticmethod
    def _int_box(value, lo, hi):
        box = QtWidgets.QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        return box

    def _with_browse(self, line_edit, slot):
        button = QtWidgets.QPushButton("Browse …")
        button.clicked.connect(slot)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)
        row.addWidget(button)
        widget = QtWidgets.QWidget()
        widget.setLayout(row)
        return widget

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Model checkpoint", self.model_path.text(),
            "Checkpoints (*.pth *.pt);;All files (*)")
        if path:
            self.model_path.setText(path)

    def _browse_repo(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "DINOv3 repository", self.dinov3_repo.text())
        if path:
            self.dinov3_repo.setText(path)

    def apply_to(self, settings):
        settings.model_path = self.model_path.text().strip()
        settings.dinov3_repo = self.dinov3_repo.text().strip()
        settings.voxel_size_xy = self.voxel_size_xy.value()
        settings.voxel_size_z = self.voxel_size_z.value()
        settings.crop_size_z = self.crop_size_z.value()
        settings.roi_size_xy = self.roi_size_xy.value()
        settings.roi_size_z = self.roi_size_z.value()


# ----------------------------------------------------------------------------
# inference in a background thread
# ----------------------------------------------------------------------------

class Cancelled(Exception):
    """Raised inside the pipeline's progress callback to stop the run."""


class InferenceThread(QtCore.QThread):
    """Runs ``run_inference.transfer_landmarks`` without blocking the window.

    The models are loaded on first use and kept in ``model_cache`` for the
    following runs.
    """

    progress = Signal(str, int, int)      # stage, done, total
    status = Signal(str)
    succeeded = Signal(object)            # TransferResult
    failed = Signal(str, str)             # message, traceback
    cancelled = Signal()

    def __init__(self, stack_t0, stack_t1, query_points, settings, model_cache,
                 parent=None):
        super().__init__(parent)
        self.stack_t0 = stack_t0
        self.stack_t1 = stack_t1
        self.query_points = query_points
        self.settings = settings
        self.model_cache = model_cache
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _report(self, stage, done, total):
        if self._cancel:
            raise Cancelled
        self.progress.emit(stage, done, total)

    def run(self):
        s = self.settings
        try:
            key = (s.model_path, s.dinov3_repo)
            if self.model_cache.get("key") != key:
                self.status.emit("Loading the models …")
                self.model_cache["models"] = pipeline.load_models(
                    s.model_path, s.dinov3_repo)
                self.model_cache["key"] = key
            model, extractor, matcher, device = self.model_cache["models"]

            result = pipeline.transfer_landmarks(
                self.stack_t0, self.stack_t1, self.query_points,
                model, extractor, matcher, device,
                s.crop_size_z, s.roi_size_xy, s.roi_size_z,
                spacing=(s.voxel_size_z, s.voxel_size_xy, s.voxel_size_xy),
                progress=self._report,
            )
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self.succeeded.emit(result)


# ----------------------------------------------------------------------------
# image view
# ----------------------------------------------------------------------------

class Toolbar(NavigationToolbar2QT):
    """Matplotlib toolbar reduced to navigation and saving the view."""

    toolitems = [item for item in NavigationToolbar2QT.toolitems
                 if item[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save")]


class PairView(QtWidgets.QWidget):
    """Two stacks side by side with point overlays.

    The two images share their axes, so zooming or panning one moves the
    other. Mouse wheel scrolls through the slices, Ctrl + wheel zooms.

    Mouse presses are reported in image coordinates; what they do is decided
    by the main window.
    """

    pressed = Signal(int, float, float, int, bool)   # side, x, y, button, shift
    dragged = Signal(int, float, float)              # side, x, y
    released = Signal(int, float, float, bool)       # side, x, y, was a drag
    blocked = Signal()                               # press ignored: zoom/pan tool on
    scrolled = Signal(int)                           # +1 / -1 slice

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(facecolor=CANVAS_BACKGROUND)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toolbar = Toolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.axes = None
        self.images = [None, None]
        self.volumes = [None, None]
        self.mips = [None, None]
        self.limits = [(0.0, 1.0), (0.0, 1.0)]
        self.gain = 1.0
        self._overlay = []
        self._press = None             # (side, screen x, screen y) while the left button is down
        self._dragging = False

        self.fig.text(0.5, 0.5, "Open a 4D TIFF stack (T, Z, Y, X)",
                      color="0.6", ha="center", va="center", fontsize=12)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    # ---- images ------------------------------------------------------------

    def reset(self):
        """Forget the current stacks, so that the next pair starts unzoomed."""
        self.volumes = [None, None]

    def set_volumes(self, vol0, vol1):
        """Show a new pair of stacks (Z, Y, X). The zoom is kept as long as
        the image size does not change."""
        new_shape = self.volumes[0] is None or self.volumes[0].shape[1:] != vol0.shape[1:]
        self.volumes = [vol0, vol1]
        self.mips = [None, None]
        self.limits = [self._display_range(v) for v in self.volumes]

        if new_shape:
            self.fig.clear()
            self._overlay = []
            self.axes = self.fig.subplots(1, 2, sharex=True, sharey=True)
            self.fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005,
                                     top=0.94, wspace=0.02)
            for side, ax in enumerate(self.axes):
                ax.set_axis_off()
                self.images[side] = ax.imshow(
                    self.volumes[side][0], cmap="gray", interpolation="nearest")
            self.toolbar.update()          # forget the zoom history
            self.toolbar.push_current()    # the full view is "home"

    @staticmethod
    def _display_range(volume):
        """Robust intensity range of a stack, from a subsample."""
        sample = volume[:, ::4, ::4]
        lo, hi = np.percentile(sample, (0.5, 99.9))
        if hi <= lo:
            hi = lo + 1
        return float(lo), float(hi)

    def plane(self, side, z, projection):
        """Slice ``z`` of one stack, or its maximum-intensity projection."""
        if projection:
            if self.mips[side] is None:
                self.mips[side] = self.volumes[side].max(axis=0)
            return self.mips[side]
        return self.volumes[side][z]

    def draw(self, slices, projection, titles, overlays, show_ids, links=()):
        """Redraw both panels.

        slices : (z_left, z_right)
        overlays : per side a list of dicts with the keys ``points`` (N, 3),
            ``ids``, ``colors`` (one per point) and ``selected`` (index or
            None)
        links : list of ((x, y) left, (x, y) right, highlighted) -- lines
            from a point in the left panel to one in the right panel
        """
        if self.axes is None:
            return
        for artist in self._overlay:
            artist.remove()
        self._overlay = []

        for side, ax in enumerate(self.axes):
            lo, hi = self.limits[side]
            image = self.images[side]
            image.set_data(self.plane(side, slices[side], projection))
            image.set_clim(lo, lo + (hi - lo) * self.gain)
            ax.set_title(titles[side], color="white", fontsize=10)

            for layer in overlays[side]:
                self._draw_points(ax, slices[side], projection, show_ids, **layer)

        self._draw_links(links)
        self.canvas.draw_idle()

    def _in_view(self, ax, x, y):
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        return x0 <= x <= x1 and y0 <= y <= y1

    def _draw_links(self, links):
        """Lines across the two panels. A line whose end is zoomed out of view
        is left out, since it would run over the rest of the window."""
        ax0, ax1 = self.axes
        for (xa, ya), (xb, yb), highlighted in links:
            if not (self._in_view(ax0, xa, ya) and self._in_view(ax1, xb, yb)):
                continue
            line = ConnectionPatch(
                xyA=(xa, ya), coordsA=ax0.transData,
                xyB=(xb, yb), coordsB=ax1.transData,
                color=COLOR_SELECTED if highlighted else COLOR_LINK,
                alpha=0.95 if highlighted else 0.55,
                linewidth=1.6 if highlighted else 1.0, zorder=4)
            self.fig.add_artist(line)
            self._overlay.append(line)

    def _draw_points(self, ax, z_view, projection, show_ids, points, ids,
                     colors, selected):
        if points is None or len(points) == 0:
            return
        colors = np.asarray(colors)
        present = ~np.isnan(points[:, 0])       # NaN: no position at this timepoint
        distance = np.abs(np.round(points[:, 0]) - z_view)
        on_slice = present if projection else distance == 0
        near = present & ~on_slice & (distance <= NEAR_SLICES)

        # rings rather than discs: the structures underneath stay visible
        for mask, size, alpha in ((on_slice, 90, 1.0), (near, 40, 0.45)):
            if mask.any():
                self._overlay.append(ax.scatter(
                    points[mask, 2], points[mask, 1], s=size, marker="o",
                    facecolors="none", edgecolors=colors[mask],
                    linewidths=1.4, alpha=alpha, path_effects=RING_OUTLINE))

        if show_ids:
            for k in np.flatnonzero(on_slice):
                self._overlay.append(ax.annotate(
                    str(ids[k]), (points[k, 2], points[k, 1]),
                    xytext=(7, 7), textcoords="offset points",
                    color=colors[k], fontsize=9, fontweight="bold",
                    path_effects=TEXT_OUTLINE, clip_on=True))

        if selected is not None and (on_slice[selected] or near[selected]):
            self._overlay.append(ax.scatter(
                points[selected, 2], points[selected, 1], s=260, marker="o",
                facecolors="none", edgecolors=COLOR_SELECTED, linewidths=1.6,
                path_effects=RING_OUTLINE))

    # ---- mouse -------------------------------------------------------------

    def side_of(self, axes):
        if self.axes is None:
            return None
        for side, ax in enumerate(self.axes):
            if axes is ax:
                return side
        return None

    def nearest_point(self, side, x, y, points, candidates=None):
        """Index of the point closest to (x, y) on screen, or None if none is
        within ``PICK_RADIUS_PX``. ``points`` is (N, 3) in (z, y, x);
        ``candidates`` is an optional boolean mask of the points to consider."""
        if points is None or len(points) == 0:
            return None
        transform = self.axes[side].transData
        screen = transform.transform(points[:, [2, 1]])
        target = transform.transform((x, y))
        distance = np.hypot(*(screen - target).T)
        distance[np.isnan(distance)] = np.inf
        if candidates is not None:
            distance[~candidates] = np.inf
        k = int(np.argmin(distance))
        return k if distance[k] <= PICK_RADIUS_PX else None

    def _image_position(self, side, event):
        """Mouse position in the image coordinates of one panel, also when the
        cursor has left that panel."""
        x, y = self.axes[side].transData.inverted().transform((event.x, event.y))
        return float(x), float(y)

    def _on_press(self, event):
        side = self.side_of(event.inaxes)
        if side is None or event.xdata is None:
            return
        if self.toolbar.mode:            # zoom or pan tool is active
            self.blocked.emit()
            return
        shift = bool(QtWidgets.QApplication.keyboardModifiers()
                     & Qt.KeyboardModifier.ShiftModifier)
        button = int(event.button)
        self._press = (side, event.x, event.y) if button == 1 else None
        self._dragging = False
        self.pressed.emit(side, float(event.xdata), float(event.ydata), button, shift)

    def _on_motion(self, event):
        if self._press is None:
            return
        side, x0, y0 = self._press
        if not self._dragging and np.hypot(event.x - x0, event.y - y0) < DRAG_START_PX:
            return
        self._dragging = True
        self.dragged.emit(side, *self._image_position(side, event))

    def _on_release(self, event):
        if self._press is None:
            return
        side = self._press[0]
        was_drag, self._press, self._dragging = self._dragging, None, False
        self.released.emit(side, *self._image_position(side, event), was_drag)

    def _on_scroll(self, event):
        ctrl = bool(QtWidgets.QApplication.keyboardModifiers()
                    & Qt.KeyboardModifier.ControlModifier)
        if not ctrl:
            self.scrolled.emit(-1 if event.step > 0 else 1)
            return
        if self.side_of(event.inaxes) is None or event.xdata is None:
            return
        # zoom around the cursor; the axes are shared, so both panels follow
        ax = self.axes[0]
        factor = 0.8 if event.step > 0 else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata + (x0 - event.xdata) * factor,
                    event.xdata + (x1 - event.xdata) * factor)
        ax.set_ylim(event.ydata + (y0 - event.ydata) * factor,
                    event.ydata + (y1 - event.ydata) * factor)
        self.toolbar.push_current()
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------
# main window
# ----------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("STRIDES")
        self.resize(1500, 880)

        self.settings = settings
        self.model_cache = {}
        self.thread = None

        # ---- state ---------------------------------------------------------
        self.stack = None              # (T, Z, Y, X)
        self.stack_path = None
        self.t = 0                     # the transition shown is t -> t + 1
        self.ids = []                  # TRACK_ID per query point
        self.query = np.zeros((0, 3))  # query points at t, (z, y, x) pixels
        self.pred = None               # re-identified points at t + 1; a NaN
                                       # row is a point without a position there
        self.edited = None             # bool per point: result changed by hand
        self.at_edge = None            # bool per point: best match at the ROI border
        self.selected = None
        self._undo = []                # snapshots of the points before each edit
        self._drag_index = None        # point being dragged
        self._drag_recorded = False
        self.csv_labels = None         # DataFrame of the loaded label CSV
        self.tracks = {}               # timepoint -> (ids, points), what gets saved
        self.keep_saved_t = False      # see save_and_next
        self.save_path = None
        self.finished = False

        self._build_ui()
        self._update_controls()

    # ---- layout ------------------------------------------------------------

    def _build_ui(self):
        self.view = PairView()
        self.view.pressed.connect(self._on_view_pressed)
        self.view.dragged.connect(self._on_view_dragged)
        self.view.released.connect(self._on_view_released)
        self.view.blocked.connect(lambda: self.message(
            "The zoom/pan tool is active – click its toolbar button again to edit points."))
        self.view.scrolled.connect(self._on_view_scrolled)

        # slice controls under the images
        self.slice_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.refresh)
        self.slice_label = QtWidgets.QLabel("Slice –")
        self.slice_label.setMinimumWidth(90)

        self.offset_box = QtWidgets.QSpinBox()
        self.offset_box.setRange(-999, 999)
        self.offset_box.setToolTip(
            "The right stack shows this many slices further than the left one. "
            "Set automatically to the z-shift found by the inference.")
        self.offset_box.valueChanged.connect(self.refresh)

        self.projection_box = QtWidgets.QCheckBox("Max. projection")
        self.projection_box.toggled.connect(self.refresh)
        self.ids_box = QtWidgets.QCheckBox("IDs")
        self.ids_box.setChecked(True)
        self.ids_box.toggled.connect(self.refresh)
        self.links_box = QtWidgets.QCheckBox("Connection lines")
        self.links_box.setToolTip(
            "Connect each query point with its re-identified position "
            "(after the inference). On a single slice only the pairs with a "
            "point on or near the slices shown are connected.")
        self.links_box.setEnabled(False)
        self.links_box.toggled.connect(self.refresh)

        self.gain_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(5, 100)
        self.gain_slider.setValue(100)
        self.gain_slider.setFixedWidth(110)
        self.gain_slider.setToolTip("Upper end of the display range; lower it to brighten dim structures")
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        slice_row = QtWidgets.QHBoxLayout()
        slice_row.addWidget(self.slice_label)
        slice_row.addWidget(self.slice_slider, 1)
        slice_row.addWidget(QtWidgets.QLabel("Right stack offset"))
        slice_row.addWidget(self.offset_box)
        slice_row.addSpacing(12)
        slice_row.addWidget(self.projection_box)
        slice_row.addWidget(self.ids_box)
        slice_row.addWidget(self.links_box)
        slice_row.addSpacing(12)
        slice_row.addWidget(QtWidgets.QLabel("Brightness"))
        slice_row.addWidget(self.gain_slider)

        viewer = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(6, 6, 0, 6)
        viewer_layout.addWidget(self.view, 1)
        viewer_layout.addLayout(slice_row)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(viewer)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open a 4D TIFF stack to begin.")

        delete = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Delete), self)
        delete.activated.connect(self.delete_selected)
        undo = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo), self)
        undo.activated.connect(self.undo)

    def _build_side_panel(self):
        # ---- data ----------------------------------------------------------
        self.open_button = QtWidgets.QPushButton("Open stack …")
        self.open_button.clicked.connect(self.open_stack_dialog)
        self.settings_button = QtWidgets.QPushButton("Settings …")
        self.settings_button.clicked.connect(self.edit_settings)
        self.stack_label = QtWidgets.QLabel("No stack loaded")
        self.stack_label.setWordWrap(True)
        self.step_label = QtWidgets.QLabel("")
        font = self.step_label.font()
        font.setBold(True)
        self.step_label.setFont(font)

        data_buttons = QtWidgets.QHBoxLayout()
        data_buttons.addWidget(self.open_button)
        data_buttons.addWidget(self.settings_button)
        data_box = QtWidgets.QGroupBox("Stack")
        data_layout = QtWidgets.QVBoxLayout(data_box)
        data_layout.addLayout(data_buttons)
        data_layout.addWidget(self.stack_label)
        data_layout.addWidget(self.step_label)

        # ---- query points --------------------------------------------------
        self.add_button = QtWidgets.QPushButton("Add points")
        self.add_button.setCheckable(True)
        self.add_button.setToolTip(
            "While active: left click in the left stack adds a point on the "
            "current slice.")
        self.add_button.toggled.connect(self._on_add_mode_toggled)
        self.load_csv_button = QtWidgets.QPushButton("Load CSV …")
        self.load_csv_button.clicked.connect(self.load_labels_dialog)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setToolTip(
            "Delete the selected point in both stacks (Del)")
        self.delete_button.clicked.connect(self.delete_selected)
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setToolTip("Take back the last change to the points (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_points)

        self.csv_every_box = QtWidgets.QCheckBox("Use the CSV labels at every timepoint")
        self.csv_every_box.setToolTip(
            "Off: the re-identified points become the query points of the "
            "next timepoint.\nOn: the query points of each timepoint are read "
            "from the loaded CSV instead.")
        self.csv_every_box.setEnabled(False)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ID", "x", "y", "z", "→ x", "→ y", "→ z", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)

        legend = QtWidgets.QLabel(
            f'<span style="color:{COLOR_QUERY}">●</span> query &nbsp; '
            f'<span style="color:{COLOR_RESULT}">●</span> re-identified')
        legend.setToolTip(
            "The note 'ROI edge' in the table means that the best match lies "
            "at the border of the search ROI, so the true position may be "
            "outside it.")
        mouse_help = QtWidgets.QLabel(
            "<b>Click</b> selects a point · <b>drag</b> moves it · "
            "<b>Shift + click</b> places the selected point on the slice shown · "
            "<b>right click</b> removes it (left stack: the whole point, "
            "right stack: only its re-identified position).")
        mouse_help.setWordWrap(True)
        mouse_help.setStyleSheet("color: gray;")

        point_buttons = QtWidgets.QGridLayout()
        point_buttons.addWidget(self.add_button, 0, 0, 1, 3)
        point_buttons.addWidget(self.load_csv_button, 0, 3, 1, 3)
        point_buttons.addWidget(self.delete_button, 1, 0, 1, 2)
        point_buttons.addWidget(self.undo_button, 1, 2, 1, 2)
        point_buttons.addWidget(self.clear_button, 1, 4, 1, 2)
        points_box = QtWidgets.QGroupBox("Points")
        points_layout = QtWidgets.QVBoxLayout(points_box)
        points_layout.addLayout(point_buttons)
        points_layout.addWidget(self.csv_every_box)
        points_layout.addWidget(self.table, 1)
        points_layout.addWidget(legend)
        points_layout.addWidget(mouse_help)

        # ---- inference -----------------------------------------------------
        self.run_button = QtWidgets.QPushButton("Run inference")
        self.run_button.clicked.connect(self.run_inference)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_inference)

        self.z_bar = QtWidgets.QProgressBar()
        self.embedding_bar = QtWidgets.QProgressBar()
        for bar in (self.z_bar, self.embedding_bar):
            bar.setRange(0, 1)
            bar.setValue(0)
        self.stage_label = QtWidgets.QLabel("")
        self.stage_label.setWordWrap(True)

        run_buttons = QtWidgets.QHBoxLayout()
        run_buttons.addWidget(self.run_button, 1)
        run_buttons.addWidget(self.cancel_button)
        bars = QtWidgets.QFormLayout()
        bars.addRow("Z-registration", self.z_bar)
        bars.addRow("Embeddings", self.embedding_bar)
        inference_box = QtWidgets.QGroupBox("Inference")
        inference_layout = QtWidgets.QVBoxLayout(inference_box)
        inference_layout.addLayout(run_buttons)
        inference_layout.addLayout(bars)
        inference_layout.addWidget(self.stage_label)

        # ---- save ----------------------------------------------------------
        self.next_button = QtWidgets.QPushButton("Save && next timepoint  →")
        self.next_button.setMinimumHeight(38)
        self.next_button.clicked.connect(self.save_and_next)
        self.save_label = QtWidgets.QLabel("")
        self.save_label.setWordWrap(True)
        self.save_label.setStyleSheet("color: gray;")

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(460)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(data_box)
        layout.addWidget(points_box, 1)
        layout.addWidget(inference_box)
        layout.addWidget(self.next_button)
        layout.addWidget(self.save_label)
        return panel

    # ---- helpers -----------------------------------------------------------

    @property
    def running(self):
        return self.thread is not None

    @property
    def n_timepoints(self):
        return 0 if self.stack is None else self.stack.shape[0]

    @property
    def n_results(self):
        """Number of points with a re-identified position."""
        return 0 if self.pred is None else int((~np.isnan(self.pred[:, 0])).sum())

    @property
    def is_last_transition(self):
        return self.stack is not None and self.t + 2 >= self.n_timepoints

    def message(self, text):
        self.statusBar().showMessage(text)

    def confirm(self, title, text):
        buttons = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self, title, text, buttons.Yes | buttons.Cancel, buttons.Cancel)
        return answer == buttons.Yes

    def error(self, title, text, details=None):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _update_controls(self):
        loaded = self.stack is not None
        idle = not self.running
        editable = loaded and idle

        self.open_button.setEnabled(idle)
        self.settings_button.setEnabled(idle)
        self.add_button.setEnabled(editable)
        self.load_csv_button.setEnabled(editable)
        self.delete_button.setEnabled(editable and self.selected is not None)
        self.undo_button.setEnabled(editable and len(self._undo) > 0)
        self.clear_button.setEnabled(editable and len(self.ids) > 0)
        self.csv_every_box.setEnabled(
            idle and self.csv_labels is not None and "POSITION_T" in self.csv_labels)
        self.run_button.setEnabled(editable and len(self.ids) > 0)
        self.links_box.setEnabled(self.n_results > 0)
        self.cancel_button.setEnabled(self.running)
        self.next_button.setEnabled(editable and self.n_results > 0)
        self.next_button.setText("Save results" if self.is_last_transition
                                 else "Save && next timepoint  →")

        if loaded:
            self.step_label.setText(
                f"Timepoint {self.t} → {self.t + 1}   "
                f"(step {self.t + 1} of {self.n_timepoints - 1})"
                + ("   – saved" if self.finished else ""))
        else:
            self.step_label.setText("")

    # ---- drawing -----------------------------------------------------------

    def slices(self):
        """Slices shown on the left and on the right."""
        z = self.slice_slider.value()
        n = self.stack.shape[1]
        return z, int(np.clip(z + self.offset_box.value(), 0, n - 1))

    def refresh(self, *_):
        if self.stack is None:
            return
        z_left, z_right = self.slices()
        n = self.stack.shape[1]
        projection = self.projection_box.isChecked()
        self.slice_label.setText("Projection" if projection else f"Slice {z_left} / {n - 1}")

        where = ("max. projection", "max. projection") if projection else (
            f"slice {z_left}", f"slice {z_right}")
        titles = (f"Timepoint {self.t}  ·  {where[0]}",
                  f"Timepoint {self.t + 1}  ·  {where[1]}")

        left = [dict(points=self.query, ids=self.ids,
                     colors=[COLOR_QUERY] * len(self.ids), selected=self.selected)]
        right = []
        if self.pred is not None:
            right.append(dict(points=self.pred, ids=self.ids,
                              colors=[COLOR_RESULT] * len(self.ids),
                              selected=self.selected))
        self.view.draw((z_left, z_right), projection, titles, (left, right),
                       self.ids_box.isChecked(), self.links())

    def links(self):
        """Lines from each query point to its re-identified position, for the
        pairs that are drawn: all in the projection, otherwise those with the
        query point or the re-identified point on or near the slice shown."""
        if not self.links_box.isChecked() or self.pred is None:
            return []
        found = ~np.isnan(self.pred[:, 0])
        if not self.projection_box.isChecked():
            near_left = self.drawn(self.query, 0)
            with np.errstate(invalid="ignore"):
                near_right = self.drawn(self.pred, 1)
            found &= near_left | near_right
        return [((self.query[k, 2], self.query[k, 1]),
                 (self.pred[k, 2], self.pred[k, 1]), k == self.selected)
                for k in np.flatnonzero(found)]

    def _on_gain_changed(self, value):
        self.view.gain = value / 100.0
        self.refresh()

    def fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.ids))
        for row, track_id in enumerate(self.ids):
            z, y, x = self.query[row]
            cells = [str(track_id), f"{x:.1f}", f"{y:.1f}", f"{z:.0f}", "", "", "", ""]
            if self.pred is not None:
                pz, py, px = self.pred[row]
                if np.isnan(pz):
                    cells[4:7] = ["–", "–", "–"]
                    cells[7] = "removed" if self.edited[row] else "not set"
                else:
                    cells[4:7] = [f"{px:.1f}", f"{py:.1f}", f"{pz:.0f}"]
                    cells[7] = ("corrected" if self.edited[row]
                                else "ROI edge" if self.at_edge[row] else "")
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        if self.selected is not None:
            self.table.selectRow(self.selected)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)

    def points_changed(self):
        """Bring table, buttons and images in line with the points."""
        self.fill_table()
        self._update_controls()
        self.refresh()

    def discard_result(self):
        self.pred = self.edited = self.at_edge = None

    # ---- undo --------------------------------------------------------------

    def push_undo(self):
        """Remember the points as they are now; call before changing them."""
        self.finished = False          # whatever was saved is about to differ
        copy = lambda a: None if a is None else a.copy()
        self._undo.append((list(self.ids), self.query.copy(), copy(self.pred),
                           copy(self.edited), copy(self.at_edge), self.selected))
        del self._undo[:-UNDO_STEPS]

    def undo(self):
        if not self._undo or self.running or self.stack is None:
            return
        (self.ids, self.query, self.pred, self.edited, self.at_edge,
         self.selected) = self._undo.pop()
        self.points_changed()
        self.message("Last change undone.")

    # ---- stack -------------------------------------------------------------

    def open_stack_dialog(self):
        start = str(Path(self.stack_path).parent) if self.stack_path else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open 4D stack", start, "TIFF stacks (*.tif *.tiff);;All files (*)")
        if path:
            self.open_stack(path)

    def open_stack(self, path):
        try:
            stack = tiff.imread(path)
        except Exception as exc:
            self.error("Could not read the stack", f"{type(exc).__name__}: {exc}")
            return False
        if stack.ndim != 4:
            self.error("Not a 4D stack",
                       f"Expected a stack with the axes (T, Z, Y, X), but "
                       f"{Path(path).name} has the shape {stack.shape}.")
            return False
        if stack.shape[0] < 2:
            self.error("Only one timepoint",
                       "The stack needs at least two timepoints.")
            return False

        self.stack = stack
        self.stack_path = str(path)
        self.t = 0
        self.ids, self.query = [], np.zeros((0, 3))
        self.discard_result()
        self.selected = None
        self.csv_labels = None
        self.csv_every_box.setChecked(False)
        self.tracks = {}
        self.keep_saved_t = False
        self.save_path = None
        self.finished = False
        self.save_label.setText("")

        n_t, n_z, n_y, n_x = stack.shape
        self.stack_label.setText(
            f"{Path(path).name}\n{n_t} timepoints · {n_z} slices · {n_x} × {n_y} px")
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n_z - 1)
        self.slice_slider.setValue(n_z // 2)
        self.slice_slider.setEnabled(True)
        self.slice_slider.blockSignals(False)
        self.offset_box.blockSignals(True)
        self.offset_box.setValue(0)
        self.offset_box.blockSignals(False)

        self.view.reset()
        self.show_transition()
        self.message("Mark the query points in the left stack ('Add points') "
                     "or load them from a CSV.")
        return True

    def show_transition(self):
        """Display timepoints ``t`` and ``t + 1``."""
        self.view.set_volumes(self.stack[self.t], self.stack[self.t + 1])
        self._undo = []                # edits of another timepoint cannot be undone
        self.points_changed()

    # ---- query points ------------------------------------------------------

    def _on_add_mode_toggled(self, on):
        self.view.canvas.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        if on:
            self.message("Left click in the left stack adds a point on the current slice.")

    def next_id(self):
        numbers = [i for i in self.ids if isinstance(i, int)]
        return max(numbers, default=0) + 1

    def points_of(self, side):
        """Query points (left stack) or re-identified points (right stack)."""
        return self.query if side == 0 else self.pred

    def drawn(self, points, side):
        """Mask of the points currently drawn on one side, so that a click
        only hits what is visible."""
        if points is None or len(points) == 0 or self.projection_box.isChecked():
            return None
        return np.abs(np.round(points[:, 0]) - self.slices()[side]) <= NEAR_SLICES

    def point_at(self, side, x, y):
        """Index of the drawn point under the cursor, or None."""
        points = self.points_of(side)
        return self.view.nearest_point(side, x, y, points, self.drawn(points, side))

    # mouse: click selects, drag moves, Shift + click places the selected
    # point, right click removes; with "Add points" a click in the left stack
    # adds a point instead

    def _on_view_pressed(self, side, x, y, button, shift):
        self._drag_index = None
        if self.stack is None or self.running:
            return

        if button == 3:
            k = self.point_at(side, x, y)
            if k is not None and side == 0:
                self.delete_point(k)
            elif k is not None:
                self.remove_result(k)
        elif button != 1:
            return
        elif side == 0 and self.add_button.isChecked():
            self.add_point(x, y)
        elif shift:
            self.place_selected(side, x, y)
        else:
            k = self.point_at(side, x, y)
            self.select(k)
            if k is not None:
                self._drag_index = k
                self._drag_recorded = False
                self.message(
                    f"Point {self.ids[k]}: drag to move it, Shift + click to "
                    "place it on the slice shown, right click to remove it.")

    def _on_view_dragged(self, side, x, y):
        if self._drag_index is None or self.running:
            return
        if not self._drag_recorded:        # one undo step per drag
            self.push_undo()
            self._drag_recorded = True
        self.move_point(side, self._drag_index, x, y)
        self.refresh()

    def _on_view_released(self, side, x, y, was_drag):
        if self._drag_index is not None and was_drag:
            self.points_changed()
        self._drag_index = None

    def move_point(self, side, k, x, y):
        """Put point ``k`` of one stack at image position (x, y) on the slice
        shown there. In the projection the point keeps its slice."""
        volume = self.stack[self.t + side]
        n_z, n_y, n_x = volume.shape
        x = float(np.clip(x, 0, n_x - 1))
        y = float(np.clip(y, 0, n_y - 1))
        points = self.points_of(side)

        if not self.projection_box.isChecked():
            z = self.slices()[side]
        elif np.isnan(points[k, 0]):
            # no slice yet: the brightest one at this pixel
            z = int(np.argmax(volume[:, int(round(y)), int(round(x))]))
        else:
            z = points[k, 0]

        if side == 0:
            points[k] = (z, round(y), round(x))    # query points sit on voxels
        else:
            points[k] = (z, y, x)
            self.edited[k] = True

    def place_selected(self, side, x, y):
        if self.selected is None:
            self.message("Select a point first, then Shift + click where it belongs.")
            return
        if side == 1 and self.pred is None:
            self.message("Run the inference first.")
            return
        self.push_undo()
        self.move_point(side, self.selected, x, y)
        self.points_changed()
        self.message(f"Point {self.ids[self.selected]} placed.")

    def add_point(self, x, y):
        _, n_z, n_y, n_x = self.stack.shape
        col = int(np.clip(round(x), 0, n_x - 1))
        row = int(np.clip(round(y), 0, n_y - 1))
        if self.projection_box.isChecked():
            # in the projection the slice is the brightest one at this pixel
            z = int(np.argmax(self.stack[self.t][:, row, col]))
        else:
            z = self.slice_slider.value()

        self.push_undo()
        self.ids.append(self.next_id())
        self.query = np.vstack([self.query, [z, row, col]])
        if self.pred is not None:          # the new point has no result yet
            self.pred = np.vstack([self.pred, [np.nan] * 3])
            self.edited = np.append(self.edited, False)
            self.at_edge = np.append(self.at_edge, False)
        self.selected = len(self.ids) - 1
        self.points_changed()
        self.message(
            f"Point {self.ids[-1]} added."
            + ("" if self.pred is None else
               " It has no re-identified position: run the inference again, "
               "or Shift + click in the right stack to place it."))

    def delete_point(self, k):
        """Remove a point from both stacks."""
        self.push_undo()
        track_id = self.ids.pop(k)
        self.query = np.delete(self.query, k, axis=0)
        if self.pred is not None:
            self.pred = np.delete(self.pred, k, axis=0)
            self.edited = np.delete(self.edited, k)
            self.at_edge = np.delete(self.at_edge, k)
        self.selected = None
        self.points_changed()
        self.message(f"Point {track_id} deleted. {len(self.ids)} points left.")

    def remove_result(self, k):
        """Drop the re-identified position of a point: it is not saved for the
        next timepoint and not carried forward."""
        self.push_undo()
        self.pred[k] = np.nan
        self.edited[k] = True
        self.points_changed()
        self.message(
            f"Re-identified position of point {self.ids[k]} removed. "
            "Shift + click in the right stack places it again.")

    def delete_selected(self):
        if self.selected is not None and self.delete_button.isEnabled():
            self.delete_point(self.selected)

    def clear_points(self):
        self.push_undo()
        self.ids, self.query = [], np.zeros((0, 3))
        self.discard_result()
        self.selected = None
        self.points_changed()

    def select(self, k):
        self.selected = k
        self.table.blockSignals(True)
        if k is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(k)
        self.table.blockSignals(False)
        self._update_controls()
        self.refresh()

    def _on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        self.selected = rows[0].row() if rows else None
        if self.selected is not None and not self.projection_box.isChecked():
            # jump to the slice of the point
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(int(round(self.query[self.selected, 0])))
            self.slice_slider.blockSignals(False)
        self._update_controls()
        self.refresh()

    def _on_view_scrolled(self, step):
        if self.stack is not None:
            self.slice_slider.setValue(self.slice_slider.value() + step)

    # ---- labels from a CSV -------------------------------------------------

    def load_labels_dialog(self):
        start = str(Path(self.stack_path).parent) if self.stack_path else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load labels", start, "CSV files (*.csv);;All files (*)")
        if path:
            self.load_labels(path)

    def load_labels(self, path):
        try:
            labels = pd.read_csv(path)
        except Exception as exc:
            self.error("Could not read the CSV", f"{type(exc).__name__}: {exc}")
            return False
        missing = [c for c in ("POSITION_X", "POSITION_Y", "POSITION_Z")
                   if c not in labels.columns]
        if missing:
            self.error("Missing columns",
                       f"{Path(path).name} has no column {', '.join(missing)}.\n\n"
                       f"Expected: {', '.join(TRACK_COLUMNS)} "
                       "(TRACK_ID and POSITION_T are optional).")
            return False

        self.csv_labels = labels
        if not self.query_from_csv(self.t, report=True):
            self.csv_labels = None
            self._update_controls()
            return False
        return True

    def query_from_csv(self, t, report=False):
        """Replace the query points by the CSV labels of timepoint ``t``.
        Returns False, and leaves the points alone, if there are none."""
        labels = self.csv_labels
        if "POSITION_T" in labels.columns:
            rows = labels[labels["POSITION_T"] == t]
        else:
            rows = labels                       # one timepoint only
        if len(rows) == 0:
            if report and "POSITION_T" in labels.columns:
                present = sorted(labels["POSITION_T"].unique().tolist())
                self.error("No labels for this timepoint",
                           f"The CSV has no rows with POSITION_T = {t} "
                           f"(it has: {present}).")
            elif report:
                self.error("No labels", "The CSV is empty.")
            return False

        s = self.settings
        ids, points, dropped = pipeline.read_query_points(
            labels, t, s.voxel_size_xy, s.voxel_size_z, self.stack.shape[1:])
        if not ids:
            if report:
                self.error("Labels outside the stack",
                           "None of the labels lies inside the stack. Check "
                           "the pixel size and slice distance under Settings.")
            return False

        self.push_undo()
        self.ids = ids
        self.query = points
        self.discard_result()
        self.selected = None
        # labels replace points that were re-identified and saved for this
        # timepoint: the saved file keeps the re-identified ones
        self.keep_saved_t = t in self.tracks
        self.points_changed()
        self.message(f"Loaded {len(self.ids)} query points for timepoint {t}"
                     + (f"; {dropped} outside the stack were skipped." if dropped else "."))
        return True

    # ---- inference ---------------------------------------------------------

    def edit_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            dialog.apply_to(self.settings)
            self.settings.save()
            return True
        return False

    def run_inference(self):
        if self.stack is None or self.running:
            return
        if len(self.ids) < 2:
            self.error("Not enough query points",
                       "At least two query points are needed: the refinement "
                       "uses the distances between them.")
            return
        s = self.settings
        if not Path(s.model_path).is_file() or not Path(s.dinov3_repo).is_dir():
            self.message("Set the model checkpoint and the DINOv3 repository first.")
            if not self.edit_settings():
                return
            if not Path(s.model_path).is_file():
                self.error("Model not found", f"No checkpoint at:\n{s.model_path}")
                return
            if not Path(s.dinov3_repo).is_dir():
                self.error("DINOv3 repository not found", f"No folder at:\n{s.dinov3_repo}")
                return

        if self.edited is not None and self.edited.any() and not self.confirm(
                "Run the inference again?",
                f"This replaces the re-identified points, including "
                f"{int(self.edited.sum())} that you changed by hand."):
            return

        self.add_button.setChecked(False)
        for bar in (self.z_bar, self.embedding_bar):
            bar.setRange(0, 1)
            bar.setValue(0)
        self.stage_label.setText("Starting …")

        # same conversion as the command-line pipeline: truncate to voxels
        query_points = self.query.astype(int)
        self.thread = InferenceThread(self.stack[self.t], self.stack[self.t + 1],
                                      query_points, s, self.model_cache, self)
        self.thread.progress.connect(self._on_progress)
        self.thread.status.connect(self.stage_label.setText)
        self.thread.succeeded.connect(self._on_succeeded)
        self.thread.failed.connect(self._on_failed)
        self.thread.cancelled.connect(self._on_cancelled)
        self.thread.finished.connect(self._on_thread_finished)
        self._update_controls()
        self.thread.start()

    def cancel_inference(self):
        if self.running:
            self.thread.cancel()
            self.stage_label.setText("Cancelling …")
            self.cancel_button.setEnabled(False)

    def _on_progress(self, stage, done, total):
        self.stage_label.setText(STAGE_TEXT.get(stage, stage))
        bar = {"z_registration": self.z_bar, "embeddings": self.embedding_bar}.get(stage)
        if bar is not None:
            bar.setRange(0, max(total, 1))
            bar.setValue(done)

    def _on_thread_finished(self):
        thread, self.thread = self.thread, None
        thread.deleteLater()
        self._update_controls()

    def _on_cancelled(self):
        self.stage_label.setText("Cancelled.")

    def _on_failed(self, text, details):
        self.stage_label.setText("Failed.")
        hint = ""
        if "outside their own ROI" in text or "outside volume_shape" in text:
            hint = ("\n\nA coarse prediction landed outside the stack. This "
                    "happens for points close to the first or last slice when "
                    "the stacks are shifted in z; remove those points and run "
                    "again.")
        self.error("Inference failed", text + hint, details)

    def _on_succeeded(self, result):
        n_slices = self.stack.shape[1]
        self.pred = pipeline.snap_to_slices(result.points, n_slices)
        self.edited = np.zeros(len(self.pred), bool)
        self.at_edge = np.asarray(result.info["peak_at_edge"], bool).copy()
        self._undo = []

        for bar in (self.z_bar, self.embedding_bar):
            bar.setValue(bar.maximum())
        z_offset = int(result.z_offset)
        self.offset_box.setValue(z_offset)      # show corresponding slices
        self.stage_label.setText(
            f"Done – {len(self.ids)} points re-identified, z-shift {z_offset:+d} slices.")
        self.message("Review the result and correct it in the right stack if "
                     "needed, then save and move on.")
        self.points_changed()

    # ---- saving ------------------------------------------------------------

    def tracks_table(self):
        """All tracks so far in the format of the label CSV."""
        return pipeline.tracks_to_table(self.tracks, self.settings.voxel_size_xy,
                                        self.settings.voxel_size_z)

    def save_and_next(self):
        if self.n_results == 0 or self.running:
            return
        if self.save_path is None:
            default = Path(self.stack_path).with_name(Path(self.stack_path).stem + "_tracks.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save tracks as", str(default), "CSV files (*.csv)")
            if not path:
                return
            self.save_path = path

        # points without a re-identified position end at this timepoint
        found = ~np.isnan(self.pred[:, 0])
        found_ids = [i for i, keep in zip(self.ids, found) if keep]

        if not (self.keep_saved_t and self.t in self.tracks):
            self.tracks[self.t] = (list(self.ids), self.query.copy())
        self.tracks[self.t + 1] = (found_ids, self.pred[found].copy())
        try:
            self.tracks_table().to_csv(self.save_path, index=False)
        except OSError as exc:
            self.error("Could not save", f"{type(exc).__name__}: {exc}")
            self.save_path = None
            return
        self.save_label.setText(
            f"Saved timepoints 0–{max(self.tracks)} to\n{self.save_path}")

        if self.is_last_transition:
            # stay here: the result can still be corrected and saved again
            self.finished = True
            self.add_button.setChecked(False)
            self._update_controls()
            self.message(f"All timepoints processed. Tracks saved to {self.save_path}")
            return

        # the re-identified points are the query points of the next step
        self.t += 1
        self.ids = found_ids
        self.query = self.pred[found].copy()
        self.keep_saved_t = False
        self.discard_result()
        self.selected = None
        self.offset_box.blockSignals(True)
        self.offset_box.setValue(0)
        self.offset_box.blockSignals(False)
        for bar in (self.z_bar, self.embedding_bar):
            bar.setRange(0, 1)
            bar.setValue(0)
        self.stage_label.setText("")

        source = "the re-identified points"
        if self.csv_every_box.isChecked() and self.csv_labels is not None:
            if self.query_from_csv(self.t):
                source = "the CSV labels"
        self.show_transition()
        self.message(f"Timepoint {self.t} → {self.t + 1}: query points are {source}.")

    # ---- window ------------------------------------------------------------

    def closeEvent(self, event):
        if self.running:
            self.thread.cancel()
            self.thread.wait()
        event.accept()


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Graphical front end of the landmark-transfer pipeline. "
                    "All options can also be set in the window.")
    parser.add_argument("--stack", type=Path, help="4D TIFF (T, Z, Y, X) to open")
    parser.add_argument("--labels", type=Path,
                        help="CSV with query points to load (" + ", ".join(TRACK_COLUMNS) + ")")
    parser.add_argument("--model", type=Path, help="checkpoint of the embedding model (.pth)")
    parser.add_argument("--dinov3-repo", type=Path, help="local clone of the DINOv3 repository")
    parser.add_argument("--voxel-size-xy", type=float,
                        help="size of a pixel in the units of the label CSV")
    parser.add_argument("--voxel-size-z", type=float,
                        help="distance between slices in the units of the label CSV")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    settings = Settings.load()
    if args.model is not None:
        settings.model_path = str(args.model)
    if args.dinov3_repo is not None:
        settings.dinov3_repo = str(args.dinov3_repo)
    if args.voxel_size_xy is not None:
        settings.voxel_size_xy = args.voxel_size_xy
    if args.voxel_size_z is not None:
        settings.voxel_size_z = args.voxel_size_z

    window = MainWindow(settings)
    window.show()
    if args.stack is not None and window.open_stack(str(args.stack)):
        if args.labels is not None:
            window.load_labels(str(args.labels))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
