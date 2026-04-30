import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf
from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm

DATA_ROOT = r"./data/input"
path_pm25 = os.path.join(DATA_ROOT, "PM25")

OUTPUT_ROOT = r"./data/output"
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

START_YEAR, END_YEAR = 2004, 2023
MONTHS_TOTAL = 240
TARGET_SHAPE = (720, 1440)
QUANTILES = [0.50, 0.90, 0.95, 0.98]
Q_NAMES = ['50th', '90th', '95th', '98th']

MIN_VALID_MONTHS = 120
MAX_ITERATIONS = 2000
SEASONAL_PERIOD = 12

def read_raster_stack(folder_path, file_pattern, expected_shape):
    search_path = os.path.join(folder_path, file_pattern)
    all_files = sorted(glob.glob(search_path))

    valid_files = []
    for f in all_files:
        filename = os.path.basename(f)
        match = re.search(r'(\d{4})', filename)
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                valid_files.append(f)

    if len(valid_files) != MONTHS_TOTAL:
        raise ValueError(f"Expected {MONTHS_TOTAL} files, found {len(valid_files)}")

    with rasterio.open(valid_files[0]) as src:
        profile = src.profile

    h, w = expected_shape
    stack = np.zeros((len(valid_files), h, w), dtype=np.float32)

    print(f"Loading {len(valid_files)} PM2.5 files...")
    for i, f in enumerate(tqdm(valid_files, unit="file", leave=False)):
        with rasterio.open(f) as src:
            data = src.read(1)
            data[data < -1e30] = np.nan
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            stack[i] = data

    return stack, profile

def solve_pixel_trend(args):
    y, time_idx, sin_t, cos_t = args

    if np.all(np.isnan(y)):
        return [np.nan] * (len(QUANTILES) * 3)

    df = pd.DataFrame({
        'PM25': y,
        'Time': time_idx,
        'Sin_T': sin_t,
        'Cos_T': cos_t
    })

    df.dropna(inplace=True)

    if len(df) < MIN_VALID_MONTHS:
        return [np.nan] * (len(QUANTILES) * 3)

    results = []
    formula = 'PM25 ~ Time + Sin_T + Cos_T'

    try:
        for q in QUANTILES:
            mod = smf.quantreg(formula, df)
            res = mod.fit(q=q, max_iter=MAX_ITERATIONS)

            coef = res.params['Time']
            pval = res.pvalues['Time']
            r2 = res.prsquared

            results.extend([coef, pval, r2])

        return results

    except Exception:
        return [np.nan] * (len(QUANTILES) * 3)

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"[{start_time}] Starting Analysis v3.0 (With Pseudo R2)...")

    time_idx = np.arange(MONTHS_TOTAL)
    sin_season = np.sin(2 * np.pi * time_idx / SEASONAL_PERIOD)
    cos_season = np.cos(2 * np.pi * time_idx / SEASONAL_PERIOD)

    pm25_stack, profile = read_raster_stack(path_pm25, "PM25_*.tif", TARGET_SHAPE)

    print("Reshaping...")
    Y_flat = pm25_stack.reshape(MONTHS_TOTAL, -1).T
    del pm25_stack

    n_pixels = Y_flat.shape[0]
    input_args = []
    for i in range(n_pixels):
        input_args.append((Y_flat[i], time_idx, sin_season, cos_season))
    del Y_flat

    print(f"Starting multiprocessing for {n_pixels} pixels...")
    num_cores = max(1, cpu_count() - 2)

    with Pool(processes=num_cores) as pool:
        results_iter = pool.imap(solve_pixel_trend, input_args, chunksize=2000)
        results = list(tqdm(results_iter, total=n_pixels, unit="px"))

    results = np.array(results)

    print("\nSaving results...")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    for idx, q_name in enumerate(Q_NAMES):
        col_coef = idx * 3
        col_pval = idx * 3 + 1
        col_r2 = idx * 3 + 2

        coef_map = results[:, col_coef].reshape(TARGET_SHAPE)
        pval_map = results[:, col_pval].reshape(TARGET_SHAPE)
        r2_map = results[:, col_r2].reshape(TARGET_SHAPE)

        base_name = f"PM25_concentration_{q_name}"

        path_coef = os.path.join(OUTPUT_ROOT, f"{base_name}_Trend.tif")
        with rasterio.open(path_coef, 'w', **profile) as dst:
            dst.write(coef_map.astype(rasterio.float32), 1)

        path_pval = os.path.join(OUTPUT_ROOT, f"{base_name}_Pvalue.tif")
        with rasterio.open(path_pval, 'w', **profile) as dst:
            dst.write(pval_map.astype(rasterio.float32), 1)

        path_r2 = os.path.join(OUTPUT_ROOT, f"{base_name}_PseudoR2.tif")
        with rasterio.open(path_r2, 'w', **profile) as dst:
            dst.write(r2_map.astype(rasterio.float32), 1)

            path_sig05 = os.path.join(OUTPUT_ROOT, f"{base_name}_Trend_Sig05.tif")
        sig05_map = coef_map.copy()
        sig05_map[pval_map >= 0.05] = np.nan
        sig05_map[np.isnan(coef_map)] = np.nan
        with rasterio.open(path_sig05, 'w', **profile) as dst:
            dst.write(sig05_map.astype(rasterio.float32), 1)

        print(f"Successfully processed and saved layers for: {q_name}")

    print(f"\nAll Done. Results saved to: {OUTPUT_ROOT}")
    print(f"Duration: {datetime.now() - start_time}")