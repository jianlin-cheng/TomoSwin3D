# TomoSwin3D

A Swin3D UNET Transformer model for the identification and classification of macromolecules in 3D cellular Cryo-Electron tomograms.

## Clone the repository

```bash
git clone https://github.com/jianlin-cheng/TomoSwin3D.git
cd TomoSwin3D
```

## Create and activate the environment (recommended)

This repo provides a Conda environment file at `environment.yml` (env name: `CryoETPick`).


```bash
conda env create -f environment.yml
conda activate CryoETPick
```

## Verify the install

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import mrcfile; print('mrcfile ok')"
```

