# Fire_PM2.5

This repository contains Python scripts for:

* **Analyzing the quantile concentration trends of global PM2.5 from 2004 to 2023** (Script: `PM25_con_change.py`)
* **Quantifying the impact of fires on PM2.5 pollution** (Script: `QR_Fire_PM_95th.py`)

## Software Environment

All analyses were conducted in Python (version 3.10).

The main Python packages required to run the scripts are:
- `numpy`
- `pandas`
- `rasterio`
- `statsmodels`
- `patsy`
- `tqdm`

The packages `os`, `glob`, `re`, `gc`, `datetime`, and `multiprocessing` are part of the Python standard library.

A conda environment is recommended because `rasterio` depends on GDAL.

```bash
conda create -n fire_pm25 python=3.10
conda activate fire_pm25
conda install -c conda-forge numpy pandas rasterio statsmodels patsy tqdm
```

## Input Data Requirements

The analysis period is 2004–2023, corresponding to 240 monthly observations. All input raster files should be preprocessed to the same spatial resolution, spatial extent, projection, and grid alignment before running the scripts.

The input files should follow the naming rules used in the scripts:

```text
PM2.5 data:       PM25_YYYYMM_*.tif
Fire data:        Firedays_YYYYMM_*.tif
ERA5 variables:   *_YYYY_MM.tif
```

The files should be sorted in chronological order from January 2004 to December 2023.

## Output Format

All outputs are saved as GeoTIFF files with LZW compression. The output rasters use the spatial metadata of the input PM2.5 raster files.
