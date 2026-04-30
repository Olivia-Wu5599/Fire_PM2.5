import os
import glob
import re
import gc
import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf
from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm

INPUT_ROOT = r"./data/input"
OUTPUT_ROOT = r"./data/output"

DIR_PM25 = "PM25"
DIR_FIRE = "Fire"
DIR_BLH = "BLH"
DIR_PRECIP = "Precipitation"
DIR_TEMP = "Temperature"
DIR_WIND = "Windspeed"

TAU = 0.95
RESERVED_CORES = 1
MIN_VALID_MONTHS = 24
MIN_FIRE_MONTHS = 3

OUTPUT_SUBDIR = f"QR_{int(TAU * 100)}th_results"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

START_YEAR, END_YEAR = 2004, 2023
MONTHS_TOTAL = 240
TARGET_SHAPE = (720, 1440)

def get_file_list(folder_path, file_pattern_type):

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found:{folder_path}")

    all_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    valid_files = []

    patterns = {
        "pm25": re.compile(r'PM25_(\d{4})(\d{2})_'),
        "fire": re.compile(r'Firedays_(\d{4})(\d{2})_'),
        "era5": re.compile(r'_(\d{4})_(\d{2})\.tif$')
    }

    pattern = patterns.get(file_pattern_type)
    if not pattern:
        raise ValueError(f"Unknown file pattern type:{file_pattern_type}")

    for f in all_files:
        filename = os.path.basename(f)
        match = pattern.search(filename)
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                valid_files.append(f)

    return valid_files


def read_raster_stack(folder_path, file_type, target_shape, needs_crop=False):
    files = get_file_list(folder_path, file_type)

    if len(files) != MONTHS_TOTAL:
        raise ValueError(f"Expected {MONTHS_TOTAL} file in {folder_path}, but found {len(files)}.")

    with rasterio.open(files[0]) as src:
        profile = src.profile

    h, w = target_shape
    stack = np.zeros((MONTHS_TOTAL, h, w), dtype=np.float32)

    print(f"Reading: {os.path.basename(folder_path)} ...")

    for i, f in enumerate(tqdm(files, unit="file", leave=False)):
        with rasterio.open(f) as src:
            data = src.read(1)

            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            else:
                data[data < -1e30] = np.nan

            if needs_crop:
                if data.shape[0] >= h and data.shape[1] >= w:
                    stack[i] = data[:h, :w]
                else:
                    stack[i] = np.nan
            else:
                if data.shape == target_shape:
                    stack[i] = data
                else:
                    try:
                        stack[i] = data[:h, :w]
                    except:
                        stack[i] = np.nan

    return stack, profile

def solve_pixel(args):
    y, fire, blh, precip, temp, wind, time_idx, sin_t, cos_t = args

    if np.isnan(y).all():
        return np.nan, np.nan, np.nan

    df = pd.DataFrame({
        'PM25': y,
        'Fire': fire,
        'BLH': blh,
        'Precip': precip,
        'Temp': temp,
        'Wind': wind,
        'Time': time_idx,
        'Sin_T': sin_t,
        'Cos_T': cos_t
    })

    df.dropna(inplace=True)

    if len(df) < MIN_VALID_MONTHS:
        return np.nan, np.nan, np.nan

    if (df['Fire'] > 0).sum() < MIN_FIRE_MONTHS:
        return np.nan, np.nan, np.nan

    try:
        formula = 'PM25 ~ Fire + BLH + Precip + Temp + Wind + Sin_T + Cos_T + cr(Time, df=4)'

        mod = smf.quantreg(formula, df)
        res = mod.fit(q=TAU, max_iter=2000)

        return res.params['Fire'], res.pvalues['Fire'], res.prsquared

    except Exception as e:
        return np.nan, np.nan, np.nan

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Starting {int(TAU * 100)}th quantile regression model")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Loading and preprocessing data")

    path = os.path.join(INPUT_ROOT, DIR_PM25)
    stack_data, profile = read_raster_stack(path, "pm25", TARGET_SHAPE, needs_crop=False)
    Y_flat = stack_data.reshape(MONTHS_TOTAL, -1).T
    del stack_data
    gc.collect()
    print("PM2.5 data loaded")

    path = os.path.join(INPUT_ROOT, DIR_FIRE)
    stack_data, _ = read_raster_stack(path, "fire", TARGET_SHAPE, needs_crop=False)
    X_fire_flat = stack_data.reshape(MONTHS_TOTAL, -1).T
    del stack_data
    gc.collect()
    print("Fire data loaded")

    controls = [
        ("BLH", DIR_BLH),
        ("Precip", DIR_PRECIP),
        ("Temp", DIR_TEMP),
        ("Wind", DIR_WIND)
    ]

    control_data_flat = {}
    for name, folder in controls:
        path = os.path.join(INPUT_ROOT, folder)
        stack_data, _ = read_raster_stack(path, "era5", TARGET_SHAPE, needs_crop=True)
        control_data_flat[name] = stack_data.reshape(MONTHS_TOTAL, -1).T
        del stack_data
        gc.collect()
        print(f"{name} data loaded")

    time_idx = np.arange(MONTHS_TOTAL)
    sin_season = np.sin(2 * np.pi * time_idx / 12)
    cos_season = np.cos(2 * np.pi * time_idx / 12)

    print("\nStarting multiprocessing calculation...")

    n_pixels = Y_flat.shape[0]

    input_args = []
    for i in range(n_pixels):
        input_args.append((
            Y_flat[i],
            X_fire_flat[i],
            control_data_flat["BLH"][i],
            control_data_flat["Precip"][i],
            control_data_flat["Temp"][i],
            control_data_flat["Wind"][i],
            time_idx, sin_season, cos_season
        ))

    del Y_flat, X_fire_flat, control_data_flat
    gc.collect()

    total_cores = cpu_count()
    num_cores = max(1, total_cores - RESERVED_CORES)
    print(f"Allocating {num_cores} out of {total_cores} CPU cores for parallel processing.")

    results = []
    with Pool(processes=num_cores) as pool:
        results_iter = pool.imap(solve_pixel, input_args, chunksize=2000)
        results = list(tqdm(results_iter, total=n_pixels, unit="px"))

    print("\nSaving results to disk...")

    results = np.array(results)

    coef_map = results[:, 0].reshape(TARGET_SHAPE)
    pval_map = results[:, 1].reshape(TARGET_SHAPE)
    r2_map = results[:, 2].reshape(TARGET_SHAPE)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    suffix = f"{int(TAU * 100)}th"

    out_path = os.path.join(OUTPUT_DIR, f'QR_{suffix}_Fire_Coef.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(coef_map.astype(rasterio.float32), 1)
    print(f"Saved coefficients: {os.path.basename(out_path)}")

    out_path = os.path.join(OUTPUT_DIR, f'QR_{suffix}_Model_R2.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(r2_map.astype(rasterio.float32), 1)
    print(f"Saved R2: {os.path.basename(out_path)}")

    out_path = os.path.join(OUTPUT_DIR, f'QR_{suffix}_Fire_Pvalue.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pval_map.astype(rasterio.float32), 1)
    print(f"Saved P value: {os.path.basename(out_path)}")

    out_path = os.path.join(OUTPUT_DIR, f'QR_{suffix}_Fire_Coef_Sig05.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        sig_map_05 = coef_map.copy()
        sig_map_05[pval_map > 0.05] = np.nan
        dst.write(sig_map_05.astype(rasterio.float32), 1)
    print(f"Saved Sig < 0.05: {os.path.basename(out_path)}")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nDone")