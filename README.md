# TomoSwin3D

TomoSwin3D is a 3D Swin U–Net style model for cryo-electron tomography: it predicts macromolecule segmentation or classification on tomogram grids and writes reconstructed prediction volumes (MRC) for downstream centroid extraction and visualization.

## Setup

After you clone the repository, create the conda environment, download the pretrained weights and sample inputs, and run prediction.

### Create conda environment

```bash
conda remove --name TomoSwin3D --all
conda env create -f environment.yml
conda activate TomoSwin3D
```

For a CPU-only machine, edit `environment.yml` and remove the `pytorch-cuda=12.4` line before `conda env create`, then install the PyTorch build appropriate for your platform.

### Download pretrained models

Weights are hosted on Zenodo ([record 19500440](https://zenodo.org/records/19500440)).

```bash
curl -L "https://zenodo.org/records/19500440/files/pretrained_models.zip?download=1" -o pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
```

Unpack so this repository contains a top-level `pretrained_models/` directory (for example `pretrained_models/TomoSwin3D_model_1.pth`). Several checkpoints are included; see `miscellaneous/which_model_to_use.txt` for which file to use (SHREC multiclass, binary centroid detection, CryoET Portal, unified multiclass, etc.).

### Download sample input data

```bash
curl -L "https://zenodo.org/records/19500440/files/sample_input_data.zip?download=1" -o sample_input_data.zip
unzip sample_input_data.zip
rm sample_input_data.zip
```

Unpack so you have `sample_input_data/` at the repository root, matching the paths expected by `predict.py` (grid NPZs under `sample_input_data/test_data/...` and reference tomograms under `sample_input_data/tomogram_collection/...`).

## Prediction on sample data

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
python get_coordinates_and_postprocessed_volume.py --help
```

See the script’s arguments for input MRC path or directory, minimum blob size, connectivity, and output CSV location.

## Rights and permissions

Open Access

This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made.

The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.

To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

## Cite this work

If you use the code, pretrained models, or sample data associated with this research, please cite:

TODO
