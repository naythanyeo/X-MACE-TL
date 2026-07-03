# Nicer X-MACE Tutorials

Tiny notebook examples that use the library instead of the CLI interface. Abit more flexibility when implementing frozen layers etc

Current X-MACE-SOC branch uses the original source code. But for LORA likely additional layers need to be added so likely need to run with an editable install instead.

Environment is the same from the original config file + the missing dependencies + xarray and netCDF4 (for reading .nc files)

## Layout

- `00_data_conversion.ipynb` converts raw static-grid NetCDF files into X-MACE extended XYZ files.
- `01_minimum_autoencoder.ipynb` trains a minimal `AutoencoderExcitedMACE` model.
- `02_transfer_learning.ipynb` runs a shared transfer-learning pipeline with naive and frozen-GNN fine-tuning.
- `data/raw_static_grid/` contains the small NetCDF demo inputs.
- `data/xyz/` contains the extended XYZ files used by the notebooks.
- `_outputs/` is for generated checkpoints, plots, logs, and run artifacts. It is ignored by git.

The notebooks locate `nicer_tutorials/data/` automatically from their current working directory, so they can be opened from any folder inside this repository.
