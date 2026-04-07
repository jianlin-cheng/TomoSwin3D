## Setup

After you clone the repository, download the pretrained models and test data, create the conda environment, and run prediction.

### Download trained models

```bash
curl https://calla.rnet.missouri.edu/TomoSwin3D/pretrained_model.tar.gz --output pretrained_model.tar.gz
tar -xvf pretrained_model.tar.gz
rm pretrained_model.tar.gz
```

### Download test data

```bash
curl https://calla.rnet.missouri.edu/TomoSwin3D/test_data.tar.gz --output test_data.tar.gz
tar -xvf test_data.tar.gz
rm test_data.tar.gz
```

### Create conda environment

```bash
conda remove --name TomoSwin3D --all
conda env create -f environment.yml
conda activate TomoSwin3D
```

## Prediction on test data

This code generates predicted macromolecules' centroid coordinates and the visualization.

```bash
python predict.py
```

### Optional arguments

TODO
Example:

```bash
python predict.py --TODO

## Rights and permissions

Open Access

This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made.

The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.

To view a copy of this license, visit `http://creativecommons.org/licenses/by/4.0/`.

## Cite this work

If you use the code or data associated with this research work or otherwise find this data useful, please cite:

TODO

