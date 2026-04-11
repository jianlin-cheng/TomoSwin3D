# TomoSwin3D: A Swin3D Transformer for the Identification and Classification of Macromolecules in 3D Cryo-ET Tomograms.

TomoSwin3D leverages a multi-channel input
representation that augments raw tomogram densities with complementary 3D feature maps capturing edge strength (Sobel
gradients), local contrast enhancement (morphological top-hat), and multiscale blob responses (Difference-of-Gaussians),
improving detectability of small and low-contrast targets. To better preserve particle geometry and avoid hand-crafted
shape assumptions, it adopts occupancy-preserving supervision that directly uses available 3D instance masks rather than
heuristic Gaussian/spherical labels and applies scalable patch-wise inference followed by lightweight post-processing
(connected-component analysis, size filtering, centroid extraction) for robust coordinate extraction. Across diverse
simulated and experimental cryo-ET tomogram benchmarks including SHREC 2020 and 2021 test datasets, EMPIAR
dataset, and CryoET data portal dataset, TomoSwin3D achieves strong and consistent performance in detecting protein and
other particles, outperforming existing methods, with a pronounced advantage in picking hard, small protein particles.
These results establish TomoSwin3D as a scalable and accurate solution for high-throughput cryo-ET macromolecule
particle picking and downstream subtomogram averaging.

## Setup

#### Clone project
```
git clone https://github.com/jianlin-cheng/TomoSwin3D.git
cd TomoSwin3D/
```

### Download model weights


```bash
curl -L "https://zenodo.org/records/19500440/files/pretrained_models.zip?download=1" -o pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
```

`pretrained_models/` directory contains several checkpoints; see `miscellaneous/which_model_to_use.txt` for for more details.

### Download sample input test data

```bash
curl -L "https://zenodo.org/records/19500440/files/sample_input_data.zip?download=1" -o sample_input_data.zip
unzip sample_input_data.zip
rm sample_input_data.zip
```

### Create conda environment

```bash
conda remove --name TomoSwin3D --all
conda env create -f environment.yml
conda activate TomoSwin3D
```

## Prepare test data for inference

Run the full pipeline with `prepare_test_data.py` (default input: `sample_input_data/tomogram_collection`, one subfolder per tomogram with `reconstruction.mrc`).

### Processing pipeline steps

**Step 1: Tomogram Normalization**

**Purpose:** Normalize raw tomogram intensity values for consistent processing.

**Step 2: Feature Map Generation**

**Step 2a: DoG Blob Features**

**Purpose:** Highlight blob-like particle structures across multiple scales.

**Step 2b: Sobel Gradient Features**

**Purpose:** Extract edge and boundary information.

**Step 2c: Top-hat Features**

**Purpose:** Enhance local contrast and remove background trends.

**Step 3: Tomogram Splitting (Base Volume)**

**Purpose:** Divide normalized tomograms into smaller 3D sub-volumes (grids).

**Step 4: Feature-Based Grid Splitting**

**Step 4a: Sobel Feature Grids**

**Purpose:** Split Sobel gradient feature maps into grids.

**Step 4b: Top-hat Feature Grids**

**Purpose:** Split top-hat feature maps into grids.

**Step 4c: DoG Feature Grids**

**Purpose:** Split DoG blob feature maps into grids.

```bash
python prepare_test_data.py
```

Optional flags include `--input-path`, `--grid-size` (default 48), `--padding` (default 8), and top-hat options; see `python prepare_test_data.py --help`.



## Prediction on Test data

`predict.py` runs inference, saves per-grid predictions, reconstructs a 3D volume, and writes an MRC under a timestamped folder in `output/results/`.

```bash
python predict.py
```

### Configuring a run

`predict.py` is driven by variables at the top of the script (not the command line). Adjust as needed:

- **`data_ids`**: list of tomogram IDs under `sample_input_data/test_data/Grids_64_normalized/tomograms/<id>/`.
- **`model_checkpoint`**: path to a `.pth` file under `pretrained_models/` (default: `pretrained_models/TomoSwin3D_model_1.pth`).
- **`threshold`**: confidence threshold for multiclass or binary masks (typical multiclass: `0.7`; binary: often `0.9` per `miscellaneous/which_model_to_use.txt`).
- **`comment`**: optional string appended to output MRC filenames.

Outputs are written under `output/results/DATETIME_<timestamp>/<data_id>/`, including predicted grids and the reconstructed MRC.

### Post-processing: centroids and CSV

To turn a predicted MRC into connected components and centroid coordinates, use:

```bash
python get_coordinates_and_postprocessed_volume.py --directory "output/results/DATETIME_2026-03-24_11:18:36"
```

Use --directory (or -d) with the path to the timestamped folder that predict.py created.

## Rights and permissions

Open Access

This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made.

The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.

To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

## Cite this work

If you use the code, pretrained models, or sample data associated with this research, please cite:

TODO
