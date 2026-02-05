#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import os
from pathlib import Path
import glob
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
# --- Storm Surge Peak Validation ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree


# In[ ]:


nconf=10
# Basin selection (change this as needed)
Basin = 'NA'  # Options: 'EP', 'NA', 'SP', 'NI', 'WP', 'SI'
# Paths and settings
folder = Path.cwd()
output_dir = folder / f'validation_ss_peak'
os.makedirs(output_dir, exist_ok=True)


if nconf>9:
    folder_path = folder
    name='storm_catalogue_ss_CONF'+str(nconf)

elif nconf<10:
    folder_path = folder
    name='storm_catalogue_ss_CONF0'+str(nconf)    


# Use glob to list files matching pattern
files= glob.glob(os.path.join(folder_path, '**', 'GESLA_hydrograph_all_0.png'), recursive=True)

tol       = 0.1
df_set=pd.DataFrame()
for  file in files:

    try: 
        path_file=file[:-26]
        merged=pd.DataFrame()
        file_gesla=glob.glob(os.path.join(path_file, '**', 'GESLA_hydrograph_all_*.csv'), recursive=True)
        file_gstm=glob.glob(os.path.join(path_file, '**', 'GSTM_hydrograph_all_*.csv'), recursive=True)

        gelsa=pd.read_csv(file_gesla[0])
        gstm=pd.read_csv(file_gstm[0])
        A=gstm.copy()
        B=gelsa.copy()

        A['REL_dist_TCmax']=A['TCdist_max']/A['TCrmw_max']
        B['REL_dist_TCmax']=B['TCdist_max']/B['TCrmw_max']

        tree = cKDTree(B[['SS_lon', 'SS_lat']].values)
        distances, indices = tree.query(A[['SS_lon', 'SS_lat']].values)

        mask      = distances <= tol
        A_matched = A[mask].reset_index(drop=True)
        B_matched = B.iloc[indices[mask]].reset_index(drop=True)
        A_matched = A_matched[['SID','SS_lon','SS_lat','ss_max_surge','ss_dur','REL_dist_TCmax','TCwind_max','TCpmin_max','TCsspeed_max']].copy()

        B_matched = B_matched[['SID','SS_lon','SS_lat','ss_max_surge','ss_dur','REL_dist_TCmax','TCwind_max','TCpmin_max','TCsspeed_max']].copy()
        merged    = pd.concat([A_matched, B_matched.add_suffix('_gelsa')], axis=1)
        df_set = pd.concat([df_set, merged], ignore_index=True)

    except:
        continue

# ===== 2. METRICS =====
df_set = df_set.dropna(subset=["ss_max_surge_gelsa"])
df_set = df_set.dropna(subset=["ss_max_surge"])

obs = df_set['ss_max_surge_gelsa']
mod = df_set['ss_max_surge']


# In[ ]:


# -------------------------------
# 1. Define wind categories
# -------------------------------

categories = [
    ("Tropical Storm", -np.inf, np.inf),
    ("Tropical Storm", -np.inf, 64),
    ("SSHW Category 1", 64, 82),
    ("SSHW Category 2", 82, 96),
    ("SSHW Category 3", 96, 113),
    ("SSHW Category 4", 113, 137),
    ("SSHW Category 5", 137, np.inf)
]

results = []

# -------------------------------
# 2. Loop through categories
# -------------------------------

for name, vmin, vmax in categories:

    df_cat = df_set[(df_set["TCwind_max"] >= vmin) & (df_set["TCwind_max"] < vmax)]
    df_cat = df_cat.dropna(subset=["ss_max_surge_gelsa", "ss_max_surge"])

    if len(df_cat) < 5:   # safety check
        print(f"Skipping {name} (too few points)")
        continue

    obs = df_cat["ss_max_surge_gelsa"]
    mod = df_cat["ss_max_surge"]

    MBE = np.mean(mod - obs)
    MAE = mean_absolute_error(obs, mod)
    RMSE = np.sqrt(mean_squared_error(obs, mod))
    R = np.corrcoef(obs, mod)[0,1]
    R2 = r2_score(obs, mod)

    results.append({
        "Category": name,
        "Storms": df_cat["SID"].nunique(),
        "Points": len(df_cat),
        "MBE": MBE,
        "MAE": MAE,
        "RMSE": RMSE,
        "R": R,
        "R2": R2
    })

    # -------------------------------
    # 3. Scatter plot
    # -------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(obs, mod, color='dodgerblue', alpha=0.7, edgecolor='silver')
    max_val = max(obs.max(), mod.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("GSTM Surge Peak (m)")
    plt.title(name)
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

    # -------------------------------
    # 4. Residual plot
    # -------------------------------
    residuals = mod - obs
    plt.figure(figsize=(8,4))
    plt.scatter(obs, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("Residual (m)")
    plt.title(f"Residuals – {name}")
    plt.xlim(0,4)
    plt.ylim(-1.5,1.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

# -------------------------------
# 5. Summary table
# -------------------------------
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv('validation_wind_confg10.csv')


# In[ ]:


# -------------------------------
# 1. Calcular cuantiles de TCpmin_max
# -------------------------------
q25, q50, q75 = df_set["TCpmin_max"].quantile([0.25, 0.5, 0.75])

# -------------------------------
# 2. Definir categorías por cuantiles
# -------------------------------
categories = [
    ("SLP high", q75, np.inf),
    ("SLP medium", q50, q75),
    ("SLP low", q25, q50),
    ("SLP very low", -np.inf, q25)
]

results = []

# -------------------------------
# 3. Loop a través de categorías
# -------------------------------
for name, vmin, vmax in categories:

    # Filtrar correctamente por TCpmin_max
    df_cat = df_set[(df_set["TCpmin_max"] >= vmin) & (df_set["TCpmin_max"] < vmax)]
    df_cat = df_cat.dropna(subset=["ss_max_surge_gelsa", "ss_max_surge"])

    if len(df_cat) < 5:
        print(f"Skipping {name} (too few points)")
        continue

    obs = df_cat["ss_max_surge_gelsa"]
    mod = df_cat["ss_max_surge"]

    MBE = np.mean(mod - obs)
    MAE = mean_absolute_error(obs, mod)
    RMSE = np.sqrt(mean_squared_error(obs, mod))
    R = np.corrcoef(obs, mod)[0,1]
    R2 = r2_score(obs, mod)

    results.append({
        "Category": name,
        "Storms": df_cat["SID"].nunique(),
        "Points": len(df_cat),
        "MBE": MBE,
        "MAE": MAE,
        "RMSE": RMSE,
        "R": R,
        "R2": R2
    })


    # -------------------------------
    # Scatter plot
    # -------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(obs, mod, color='dodgerblue', alpha=0.7, edgecolor='silver')
    max_val = max(obs.max(), mod.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("GSTM Surge Peak (m)")
    plt.title(name)
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

    # -------------------------------
    # Residual plot
    # -------------------------------
    residuals = mod - obs
    plt.figure(figsize=(8,4))
    plt.scatter(obs, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("Residual (m)")
    plt.title(f"Residuals – {name}")
    plt.xlim(0,4)
    plt.ylim(-1.5,1.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

# -------------------------------
# 4. Tabla resumen
# -------------------------------
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv('validation_SLP_config10.csv', index=False)


# In[ ]:


# 1. Calculate quantiles
q25, q50, q75 = df_set["REL_dist_TCmax"].quantile([0.25, 0.5, 0.75])

# 2. Define categories
categories = [
    ("Very low relative distance", -np.inf, q25),
    ("Low relative distance", q25, q50),
    ("Medium relative distance", q50, q75),
    ("High relative distance", q75, np.inf)
]

results = []

# 3. Loop through categories
for name, vmin, vmax in categories:

    if vmax == np.inf:
        df_cat = df_set[df_set["REL_dist_TCmax"] >= vmin]
    else:
        df_cat = df_set[(df_set["REL_dist_TCmax"] >= vmin) & (df_set["REL_dist_TCmax"] < vmax)]

    df_cat = df_cat.dropna(subset=["ss_max_surge_gelsa", "ss_max_surge"])

    if len(df_cat) < 5:
        print(f"Skipping {name} (too few points)")
        continue

    obs = df_cat["ss_max_surge_gelsa"]
    mod = df_cat["ss_max_surge"]

    MBE = np.mean(mod - obs)
    MAE = mean_absolute_error(obs, mod)
    RMSE = np.sqrt(mean_squared_error(obs, mod))
    R = np.corrcoef(obs, mod)[0,1]
    R2 = r2_score(obs, mod)

    results.append({
        "Category": name,
        "Storms": df_cat["SID"].nunique(),
        "Points": len(df_cat),
        "MBE": MBE,
        "MAE": MAE,
        "RMSE": RMSE,
        "R": R,
        "R2": R2
    })


    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(obs, mod, color='dodgerblue', alpha=0.7, edgecolor='silver')
    max_val = max(obs.max(), mod.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("GSTM Surge Peak (m)")
    plt.title(name)
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

    # Residual plot
    residuals = mod - obs
    plt.figure(figsize=(8,4))
    plt.scatter(obs, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("GESLA Surge Peak (m)")
    plt.ylabel("Residual (m)")
    plt.title(f"Residuals – {name}")
    plt.xlim(0,4)
    plt.ylim(-1.5,1.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_{name.replace(' ','_')}.png", dpi=300)
    plt.close()

# 4. Summary table
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv('validation_rel_distance_config10.csv', index=False)

