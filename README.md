# STRIDES

We provide STRIDES, a framework for **S**elf-supervised **T**emporal 
**R**e-**I**dentification of **DE**ndritic **S**pines.

Project website: [STRIDES](https://ankilab.github.io/STRIDES/)

<img width="1400" height="722" alt="Figure_1" src="https://github.com/user-attachments/assets/6681bfaa-c8ef-40de-a5a7-5875ed54d201" />

Given the dendritic spines at timepoint `t`, the pipeline predicts where each of them
is at `t + 1` in longitudinal microscopic volumes:

1. **Coarse x/y.** SuperPoint + LightGlue match features between the
   maximum-intensity projections of the two stacks. Each landmark is moved by
   the interpolated displacement of its nearest matched features.
2. **Coarse z.** Every plane of one stack is matched against every plane of the
   other. The plane offset with the highest total match score is the z-shift.
3. **Similarity maps.** A DINOv3-based encoder embeds a small crop around each
   landmark at `t`, and around every voxel of a ROI centred on the coarse
   prediction at `t + 1`. The cosine similarity gives one 3D map per landmark.
4. **Refinement.** All landmarks are optimised jointly: each is pulled towards
   the peak of its similarity map while the pairwise distances of the point set
   are preserved and the result stays near the coarse prediction.

The encoder is trained self-supervised: a crop and a randomly rotated, noisy
copy of the same crop should map to the same embedding (`training/`).

## Repository layout

```
inference/
    gui.py              graphical user interface: mark or load query points, run the
                        transfer, correct it, step through the timepoints
    run_inference.py    the pipeline (transfer_landmarks) and a command-line
                        script that runs it on stacks and scores it against labels
    utils.py            embedding model, coarse registration, ROIs, export, figures
    optimize.py         joint refinement of the landmark positions
training/
    train.py            generates training pairs, trains and evaluates the encoder
    models.py           embedding model, triplet loss, training loop
environment.yml         conda environment
requirements.txt        pip requirements (also used by environment.yml)
.gitignore
LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate strides
git clone https://github.com/facebookresearch/dinov3.git
```

The pinned versions are those STRIDES was developed and evaluated with
(Python 3.10, PyTorch 2.9.1). PyTorch is installed for CUDA 12.6; for another
CUDA version change the PyTorch index URL at the top of `requirements.txt`
(see [pytorch.org](https://pytorch.org/get-started/locally/)). Without conda:
`pip install -r requirements.txt` in a Python 3.10 environment.

The encoder loads its backbone through `torch.hub` from a local clone of the
[DINOv3 repository](https://github.com/facebookresearch/dinov3). By default the
clone is expected at the top level of this repository (`./dinov3`); point to
another location with `--dinov3-repo` or the `DINOV3_REPO` environment
variable. The packages DINOv3 needs for this are part of `requirements.txt`
(checked with DINOv3 commit `6876159`). See the DINOv3 README for access to
the pretrained backbone weights.

The scripts are run from their own folder, e.g. `cd inference` first.

Before running the GUI or run_inference.py, download our fine-tuned DINOv3 model
and store it at `./checkpoints/strides_embedder.pth`: 
[STRIDES Embedder](https://zenodo.org/records/23467782)

## Hardware

STRIDES was developed and tested on NVIDIA GPUs: the embedding model was
trained on an A100 (40 GB), and inference, including the GUI, was run on an
RTX 3090 (24 GB). Smaller GPUs have not been tested, but are not expected to work well.

## Graphical user interface

```bash
cd inference
python gui.py
python gui.py --stack ../data/stack.tif --model ../checkpoints/strides_embedder.pth
```

1. **Open stack.** Timepoints 0 and 1 of the 4D TIFF are shown side by side.
   The slider (or the mouse wheel) scrolls through the slices of both; Ctrl +
   mouse wheel zooms, and the two views zoom and pan together.
2. **Query points.** Either switch on *Add points* and click in the left stack
   to add points on the current slice, or load a label CSV. From a CSV the rows
   of the timepoint on the left are taken. The CSV requires columns 
   *POSITION_X, POSITION_Y, POSITION_Z* and optionally *TRACK_ID, POSITION_T*.
3. **Run inference.** Two progress bars follow the z-registration and the
   embedding computation; *Cancel* stops the run. The re-identified points are
   then drawn in the right stack, and *Right stack offset* is set to the
   z-shift that was found so that both views show corresponding slices.
   *Connection lines* (below the images) draws a line from each query point
   to its re-identified position.
4. **Correct the result** where needed, see the table below.
5. **Save & next timepoint** writes the tracks so far to a CSV and moves on to
   the next pair of timepoints. The re-identified points become the new query
   points. With *Use the CSV labels at every timepoint* the query points are
   read from the loaded CSV instead.

Points are edited with the mouse, in the same way in both stacks:

| Action | Effect |
| --- | --- |
| Click | Select a point (also by clicking its row in the table) |
| Drag | Move the point; it lands on the slice shown |
| Shift + click | Place the selected point at the cursor on the slice shown, e.g. to move it to another slice |
| Right click, left stack | Delete the point from both stacks |
| Right click, right stack | Remove only the re-identified position: the point is not saved for the next timepoint and not carried forward |
| Ctrl + Z | Undo the last change |

A point added after the inference has no re-identified position until the
inference is run again or the point is placed with Shift + click in the right
stack. While the zoom or pan tool of the toolbar is active, clicks belong to
that tool.

The saved file has the format of the label CSV (`TRACK_ID`, `POSITION_X`,
`POSITION_Y`, `POSITION_Z`, `POSITION_T`): the query points of the first
timepoint, and the re-identified points of every later one, including manual
corrections. It is rewritten after each step.

The model checkpoint, the DINOv3 repository, the voxel size and the ROI sizes
are set under *Settings* and remembered between sessions. Set the voxel size
before loading labels that are in physical units.

## Command-line inference

`run_inference.py` runs the same pipeline without interaction, in one of two
modes:

- **`--mode track`** does what the GUI does: it starts from the labelled
  points of one timepoint, predicts where they are at the next one, uses the
  predictions as the query points of the step after, and so on to the last
  timepoint. Only the starting timepoint needs labels. With
  `--labels-every-timepoint` the query points of every step are read from the
  CSV instead, like the GUI's *Use the CSV labels at every timepoint*.
- **`--mode evaluate`** (the default) transfers the labels of every timepoint
  to the next one and scores the predictions against the labels there, so it
  needs labels at every timepoint.

### Input

- **Stack:** a 4D TIFF with axes `(T, Z, Y, X)`.
- **Labels:** a CSV with one row per landmark and timepoint, and the columns
  `TRACK_ID`, `POSITION_X`, `POSITION_Y`, `POSITION_Z`, `POSITION_T`.
  `POSITION_T` is the index of the timepoint.
  - *track:* the rows of the starting timepoint are used: the first timepoint
    in the CSV, or the one given with `--start-timepoint`. `TRACK_ID` and
    `POSITION_T` are optional, as in the GUI. With `--labels-every-timepoint`
    the rows of each later timepoint are used too (this needs `POSITION_T`);
    a timepoint without rows falls back to the predictions.
  - *evaluate:* a landmark is transferred from `t` to `t + 1` only if its
    `TRACK_ID` is labelled at both, because the labels at `t + 1` are used to
    score the prediction.

If the positions are in physical units rather than pixels, pass the voxel size
with `--voxel-size-xy` and `--voxel-size-z`.

### Usage

Track the labelled points of one stack (from `inference/`):

```bash
python run_inference.py --mode track \
    --stacks ../data/tiffs/stack.tif \
    --labels ../data/csvs/stack.csv \
    --model ../checkpoints/strides_embedder.pth \
    --output ../results
```

Evaluate STRIDES predictions of one stack against given labels:

```bash
python run_inference.py \
    --stacks ../data/tiffs/stack.tif \
    --labels ../data/csvs/stack.csv \
    --model ../checkpoints/strides_embedder.pth \
    --output ../results
```

A folder of stacks, in either mode. Each `<name>.tif` is paired with
`<name>.csv`; files without a partner are reported and skipped:

```bash
python run_inference.py \
    --stacks ../data/tiffs \
    --labels ../data/csvs \
    --model ../checkpoints/strides_embedder.pth \
    --output ../results/my_dataset \
    --skip-done
```

`--skip-done` skips stacks that already finished in an earlier run. A stack
that raises an error is reported and the batch continues with the next one.

| Option | Default | Meaning |
| --- | --- | --- |
| `--crop-size-z` | 3 | z-slices in each embedding crop |
| `--roi-size-xy` | 32 | x/y size of the similarity ROI, in pixels |
| `--roi-size-z` | 8 | z size of the similarity ROI, in slices |
| `--voxel-size-xy` | 1.0 | size of a pixel in the units of the label CSV |
| `--voxel-size-z` | 1.0 | distance between slices in the units of the label CSV |

Run `python run_inference.py --help` for the full list. The weights of the
refinement step and the remaining fixed settings are constants at the top of
`run_inference.py`.

In evaluate mode the script opens two overview figures per transition (coarse
transfer, and predictions against labels) and waits until they are closed.

### Output

Everything for a stack goes to `<output>/<name>/`, plus a `.done` file once
the stack finished without error.

**track:** `<name>_tracks.csv`, in the format of the label CSV and the same
as the file the GUI saves: the query points at the starting timepoint and
the predicted positions at every later one (with `--labels-every-timepoint`,
each predicted from the labels of the timepoint before). It is rewritten
after each timepoint, so a run that stops half-way keeps the timepoints already done.
No figures are shown, so this mode also runs unattended.

**evaluate:**

| File | Content |
| --- | --- |
| `<name>_t{t}_t{t+1}.csv` | One row per landmark for the transition `t -> t+1` |

Columns of the evaluation CSV, all in the units of the label CSV:

| Column | Content |
| --- | --- |
| `TRACK_ID` | Landmark identifier |
| `T{t}_X/Y/Z` | Label at `t` |
| `T{t+1}_X/Y/Z` | Label at `t + 1` |
| `LG_X/Y/Z` | Coarse prediction (steps 1 and 2) |
| `PRED_X/Y/Z` | Final prediction (step 4) |
| `DISTANCE` | Euclidean distance between `PRED` and the label at `t + 1` |

## Training

`training/train.py` trains the encoder without labels. From every training
stack it takes random foreground voxels (Otsu mask); a crop around each voxel
and a copy rotated in 3D around the crop centre (up to ±3° tilt, any in-plane
angle) form a positive pair, both independently augmented with noise, and blur. 
A semi-hard triplet loss pulls the two embeddings of
a pair together and pushes the other pairs in the batch away.

After training, the encoder is evaluated on a separate benchmark stack: crops
at every n-th foreground voxel are matched against the rotated crops of all of
them, and the script prints how often the best match is the right one and the
95th percentile of the distance to the best match.

We trained and evaluated our model using the DeepD3 data, which can be found here:
[DeepD3 data](https://zenodo.org/records/8428849)

```bash
cd training
python train.py \
    --benchmark ../data/DeepD3_Benchmark.tif \
    --training-dir ../data/DeepD3_Training \
    --output-dir ../results/training \
    --cache-dir /scratch/strides_cache
```

- **Benchmark:** one stack `(Z, Y, X)`.
- **Training stacks:** every `.tif` in `--training-dir`, axes `(Z, C, Y, X)`;
  channel 0 is used.
- **Cache:** the crop pairs are first written to memory-mapped files in
  `--cache-dir`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--crop-size-xy` / `--crop-size-z` | 128 / 7 | crop size |
| `--points-per-stack` | 3000 | training crops per training stack |
| `--offset` | 5 | every n-th foreground voxel of the benchmark is evaluated |
| `--voxel-size` | 500 94 94 | benchmark voxel size (z y x) in nm, for the distances |
| `--epochs` | 100 | maximum; training stops early when the loss stalls for 7 epochs |
| `--batch-size`, `--lr` | 64, 1e-4 | |
| `--seed` | 42 | NumPy seed for point sampling and augmentation |

Outputs in `--output-dir`, with `<tag>` = `<name>_<crop xy>_<crop z>`
(`<name>` defaults to `deepd3_seed<seed>`):

| File | Content |
| --- | --- |
| `strides_embedder_ssl_<tag>.pth` | Model weights; pass to `--model` for inference |
| `loss_<tag>.npy` | Mean training loss per epoch |
| `distances_<tag>.npy` | Per benchmark point, distance in nm to its best match |

## License

The code in this repository is released under the Apache-2.0 License, see `LICENSE`.
It uses third-party models under their own terms: the DINOv3 backbone (and
therefore checkpoints fine-tuned from it) is covered by the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license),
and the SuperPoint weights used through LightGlue have a separate, restrictive
license (see the [LightGlue README](https://github.com/cvg/LightGlue)).
