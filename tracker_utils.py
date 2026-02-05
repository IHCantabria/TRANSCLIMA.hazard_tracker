import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import xarray as xr
from sklearn.cluster import DBSCAN
from datetime import datetime
import os 
from general_utils import load_data,dynamic_kmeans, static_kmeans,plot_results,elbow_method,create_netCDF,sort_clusters,plot_results1, haversine,relative_angle_cyclone_shoreline
from pathlib import Path
from scipy.spatial import ConvexHull
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
import json
import warnings
warnings.filterwarnings('ignore')

def tracking_ss(basin, ii, folder, eps, min_samples, delta, threshold_peak,variable_name,pathout,ncentroids):
    # ======================================================
    # paths
    # ======================================================

    path_storm = folder / "storms"  
    path_storm.mkdir(parents=True, exist_ok=True)
    file_storm = path_storm / f'ibtracs.{basin}.list.v04r01_filtered_interpolate_v1.csv'

    if basin=='EP': #Eastern Pacific
        lat0,latf,lon0,lonf=0,60,180,-75
    elif basin=='NA': #North Atlantic
        lat0,latf,lon0,lonf=0,60,-105,0
    elif basin=='NI': #North Indian
        lat0,latf,lon0,lonf=0,60,30,100
    elif basin=='SI': #South Indian
        lat0,latf,lon0,lonf=-35,0,10,135
    elif basin=='SP': #South Pacific
        lat0,latf,lon0,lonf=-60,0,135,-120
    elif basin=='WP': #Western Pacific
        lat0,latf,lon0,lonf=0,60,100,180

    if variable_name=='ss':
        path_files = folder / "storm_surges_v3" / "shoreline"
        var_='surge'

    path_winds = folder / "winds" 
    path_slp = folder / "slp" 


    # ======================================================
    # read storm information
    # ======================================================
    df_list = pd.read_csv(file_storm,skiprows=[1]) 
    df_list=df_list[df_list['SEASON']>1979]
    SID_list=df_list['SID'].unique()

    SID=SID_list[ii]
    # test alogrithm with Katrina TC
    print(f"Selected storm ID: {SID}")

    path_output = pathout / SID  
    path_output.mkdir(parents=True, exist_ok=True)

    data_TC=df_list[df_list['SID']==SID]
    time0=data_TC.ISO_TIME.values[0]
    timef=data_TC.ISO_TIME.values[-1]

    lon0=np.nanmin(data_TC.LON)-delta
    lonf=np.nanmax(data_TC.LON)+delta
    lat0=np.nanmin(data_TC.LAT)-delta
    latf=np.nanmax(data_TC.LAT)+delta

    # identify the range to analayse the netcdf files 
    time0 = datetime.strptime(time0, '%Y-%m-%d %H:%M:%S')
    year0, month0 = time0.year, time0.month

    timef = datetime.strptime(timef, '%Y-%m-%d %H:%M:%S')
    yearf, monthf = timef.year, timef.month
    
    data_TC['datetime64']= pd.to_datetime(data_TC.ISO_TIME).copy()

    # ======================================================
    # READ files information of climatic variables, such as hs, wind, slp
    # ======================================================
    #dataset_waves=load_data(path_waves, variable_name,year0, month0, yearf, monthf)
    dataset_surge = load_data(path_files, variable_name,year0, month0, yearf, monthf,basin)
  
    
    # Extract station coordinates
    station_x = dataset_surge['station_x_coordinate'].values
    station_y = dataset_surge['station_y_coordinate'].values

    # filter data
    mask = (
    (dataset_surge['station_y_coordinate'] >= lat0) &
    (dataset_surge['station_y_coordinate'] <= latf) &
    (dataset_surge['station_x_coordinate'] >= lon0) &
    (dataset_surge['station_x_coordinate'] <= lonf)
    )

    # Apply the mask to filter the stations
    surge_basin = dataset_surge.where(mask, drop=True)

    # select the data for an specfici storm 
    times = data_TC['datetime64']

    time_extended = pd.date_range(
        start=pd.to_datetime(times.iloc[0]) ,
        end=pd.to_datetime(times.iloc[-1]) + pd.Timedelta(days=10),
        freq="H")

    surge_basin_1 = surge_basin.interp(time=time_extended, method='linear')
    surge_basin = surge_basin.interp(time=times, method='linear')

    #TC_surge=surge_basin.copy()

    if threshold_peak=='p99':
        path_per = folder / "percentile/reanalysis_surge_percentile99.nc"
        percentile=xr.open_dataset(path_per)
        mask = (surge_basin[var_] >= percentile.surge_p99) 
        TC_surge = surge_basin.where(mask)
    
    elif threshold_peak=='p95':
        path_per = folder / "percentile/reanalysis_surge_percentile95.nc"
        percentile=xr.open_dataset(path_per)
        mask = (surge_basin[var_] >= percentile.surge_p95) 
        TC_surge = surge_basin.where(mask)
    else:
        mask = (surge_basin[var_] >= float(threshold_peak)) 
        TC_surge = surge_basin.where(mask)

    nt,nx= TC_surge.surge.values.shape
    x = TC_surge['lon'].values
    y = TC_surge['station_y_coordinate'].values

    # ======================================================
    var1= getattr(TC_surge, var_).values
    var1 [np.isnan(var1)]=-999

    #optimal_k=elbow_method(dynamic_kmeans, var1, k_range=range(2, 11))
    idx1, centroids_norm, centroids,_=dynamic_kmeans(0, var1, var1, var1, ncentroids)

    TC_centroid=centroids[centroids>0.1]
    TC_centroid=np.unique(TC_centroid)
    TC_centroid=TC_centroid.tolist()
    mask = np.isin(centroids, TC_centroid).all(axis=1)
    TC_idx = np.where(mask)[0]

    image_filenames, xarray_time, xarray_z, z_output, track_lat, track_lon, track_storm_speed, track_storm_dir, track_wind_speed, track_slp,track_RMW,xarray_wind,xarray_slp = ([] for _ in range(13))
    
    for i in range(len(TC_surge.time)):

            try: 
                slp0 = TC_surge.isel(time=i)
                specific_time = TC_surge.time[i]
            except: 
                slp0 = TC_surge.isel(valid_time=i)
                specific_time = TC_surge.valid_time[i]

            data_TC1 = data_TC[data_TC['datetime64'] == specific_time.values]
            #z = getattr(slp0, var_).values
            z = slp0['surge'].values
            #n,m=z.shape
            # Extract indices and apply the mask to station coordinates
            idx0_1 = idx1[i, :]

            mask = np.isin(idx0_1, TC_idx)
            z0 = z.copy()
            #z0[~mask] = np.nan
            z0 =z0[mask] 
            station_x0 = x.copy()
            station_y0 = y.copy()

            station_y0 =station_y0[mask] 
            station_x0 =station_x0[mask] 
            # Perform DBSCAN clustering
            try: 
                xy_points = np.column_stack((station_x0, station_y0))
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                idx_ltln = dbscan.fit_predict(xy_points)

                # Find centroids of clusters
                idx_position = np.unique(idx_ltln[idx_ltln != -1])  # Exclude noise (-1)
                centroids = np.array([np.mean(xy_points[idx_ltln == cluster], axis=0) for cluster in idx_position])
                centroids[centroids[:, 0] > 180, 0] -= 360
                # Calculate the distance to each centroid
                distances = np.linalg.norm(centroids - np.array([data_TC1['LON'].values[0], data_TC1['LAT'].values[0]]), axis=1)
                closest_index = np.argmin(distances)
                closest_centroid = centroids[closest_index]

                # Get the corresponding cluster and points
                centroid_position = np.where(centroids == closest_centroid)[0][0]
                TC_idx2filter = idx_position[centroid_position]
                idx2 = idx_ltln[idx_ltln == TC_idx2filter]
                station_x2 = station_x0[idx_ltln == TC_idx2filter]
                station_y2 = station_y0[idx_ltln == TC_idx2filter]
                z2 = z0[idx_ltln == TC_idx2filter]

                # once I have filter a cluster, compare the distance and if is too large do not considered this cluster 
                xcenter=np.nanmean(station_x2)
                ycenter=np.nanmean(station_y2)
                if xcenter>180: 
                    xcenter=xcenter-360

                cluster_xy=[xcenter,ycenter]
                cluster_xy=np.array(cluster_xy)
                distances_0 = np.linalg.norm(cluster_xy - np.array([data_TC1['LON'].values[0], data_TC1['LAT'].values[0]]))

                if distances_0>10:
                    station_x2 = station_x0.copy()
                    station_y2 = station_y0.copy()
                    z2 = z0.copy() +np.nan

            except:
                station_x2 = station_x0.copy()
                station_y2 = station_y0.copy()
                z2 = z0.copy()

                                 
            z_output = np.full_like(z, np.nan)  # start all NaN
            
            for xi, yi, zi in zip(station_x2, station_y2, z2):
                # find the index in the original x,y grid
                idx = np.where((x == xi) & (y == yi))[0]
                z_output[idx] = zi

            #z_output=z_output.reshape(n,m) 
            z_output = z_output.flatten()
            z_output[z_output<0] = np.nan
            
            xarray_z.append(z_output)
            xarray_time.append(specific_time.time.values)
            track_lat.append(data_TC1['LAT'].values[0])
            track_lon.append(data_TC1['LON'].values[0])
            track_storm_speed.append(data_TC1['STORM_SPEED'].values[0])
            track_storm_dir.append(data_TC1['STORM_DIR'].values[0])
            track_RMW.append(data_TC1['RMW'].values[0])
            track_wind_speed.append(data_TC1['WIND_COMB'].values[0])
            track_slp.append(data_TC1['Pmin'].values[0])

            #plot_results1(data_TC, data_TC1, x, y, idx0_1.flatten(), 'hsv', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'idx',variable_name,0,5)
            #plot_results1(data_TC, data_TC1, station_x0, station_y0, z0, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'idx0',variable_name,0.1,1)
            #plot_results1(data_TC, data_TC1, station_x2, station_y2, z2, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'_GSTM',variable_name,0.5,1)
            #plot_results1(data_TC, data_TC1, x, y, z_output, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name,variable_name,0.1,1)
            #plot_results1(data_TC, data_TC1, x, y, z, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'_ALL',variable_name,0,1)

    c2 = np.sum(~np.isnan(xarray_z), axis=0)  # (station,)
    mask = c2 > 1
    z_stack = np.stack(xarray_z)
    xarray_z_filtered = z_stack[:, mask]
    xarray_z_filtered

    x_filtered = x[mask]
    y_filtered = y[mask]

   
    create_netCDF(xarray_z_filtered, xarray_time,y_filtered,x_filtered, track_lat, track_lon, track_wind_speed, track_storm_dir, track_storm_speed, track_slp,track_RMW,SID, path_output,variable_name)

    ds_filtered = surge_basin_1.sel(stations=surge_basin_1['stations'].values[mask])
    ds_extended = ds_filtered.reindex(time=time_extended, method=None)  # method=None deja NaN en los nuevos tiempos

    data_TC['time'] = pd.to_datetime(data_TC['ISO_TIME'])
    data_TC = data_TC.set_index('time')
    data_TC_extend = data_TC.reindex(time_extended)
    # Add storm variables (keep only the 'time' dimension)
    try:
        ds_extended = ds_extended.assign(
            storm_latitude=("time", data_TC_extend['LAT'].values),
            storm_longitude=("time", data_TC_extend['LON'].values),
            storm_wind_speed=("time", data_TC_extend['WIND_COMB'].values),
            storm_direction=("time", data_TC_extend['STORM_DIR'].values),
            storm_RMW=("time", data_TC_extend['RMW'].values),
            storm_Pmin=("time", data_TC_extend['Pmin'].values),
            storm_speed=("time", data_TC_extend['STORM_SPEED'].values)
        )

        output_file = path_output / f"{SID}_GSTM_all_time_series.nc"
        ds_extended.to_netcdf(output_file)


        print(f"File saved: {output_file}")

    except Exception as e:
        print(f'No storm surge over threshold {threshold_peak} was detected for SID {SID}')



def hydrograph(basin, ii, threshold, folder,variable_name,pathout):
    """
    Run storm hydrograph extraction for a given basin and storm index.

    Parameters
    ----------
    basin : str
        Basin code ('NA', 'EP', 'NI', 'SI', 'SP', 'WP')
    ii : int
        Index of storm in the filtered storm list
    folder : str or Path
        Base directory path. Defaults to current working directory.
    """
    # ======================================================
    # paths
    # ======================================================
    path_storm = folder / "storms"  
    path_storm.mkdir(parents=True, exist_ok=True)
    file_storm = path_storm / f'ibtracs.{basin}.list.v04r01_filtered_interpolate_v1.csv'

    # Basin bounds
    bounds = {
        'EP': (0, 60, 180, -75),
        'NA': (0, 60, -105, 0),
        'NI': (0, 60, 30, 100),
        'SI': (-35, 0, 10, 135),
        'SP': (-60, 0, 135, -120),
        'WP': (0, 60, 100, 180)
    }
    lat0, latf, lon0, lonf = bounds[basin]

    # ======================================================
    # read storm information
    # ======================================================
    df_list = pd.read_csv(file_storm, skiprows=[1]) 
    df_list = df_list[df_list['SEASON'] > 1979]
    SID_list = df_list['SID'].unique()
    SID = SID_list[ii]

    path_file = pathout / SID  
    path_out = path_file / "hydrograph"
    path_out.mkdir(parents=True, exist_ok=True)

    data = xr.load_dataset(path_file / f"{SID}_GSTM_ss.nc")

    data = data.assign_coords({"latitude": ("station", data.latitude.values),
                               "longitude": ("station", data.longitude.values)})
    
  
    data_all = xr.load_dataset(path_file / f"{SID}_GSTM_all_time_series.nc") 
    ds_clean = data.dropna(dim="station", how="all", subset=["ss"])

    # ======================================================
    # main processing loop
    # ======================================================
    columns = ['SID','IDss','SS_lon','SS_lat','ss_t0','ss_tf','ss_max_surge','ss_time_max',
            'ss_tasc','ss_tdes','ss_dur','TClat_max','TClon_max','TCwind_max','TCdir_max',
            'TCrmw_max','TCpmin_max','TCsspeed_max','TCdist_max','TCrel_dist_max','localwinds_max','localslp_max','rel_angle_max',
            'TCwind_dur','TCdir_dur',
            'TCrmw_dur','TCpmin_dur','TCsspeed_dur','localwinds_dur','localslp_dur']

    df_hydrograph = pd.DataFrame(columns=columns)
    valid_stations = []

    for i in range(len(ds_clean.station)):
        lon = ds_clean.longitude[i]
        lat = ds_clean.latitude[i]

        ss_series = ds_clean.ss.values[:, i]
        time_series = ds_clean.time.values
        valid_mask = ~np.isnan(ss_series)
        if np.sum(valid_mask) == 0:
            continue

        mask_not_nan = ~np.isnan(ds_clean.ss.values[:, i])
        time_clean = ds_clean.time.values[mask_not_nan]
        ss_clean   = ds_clean.ss.values[:, i][mask_not_nan]

        max_idx = np.argmax(ss_clean)
        max_val = ss_clean[max_idx]
        max_time = time_clean[max_idx]
        intervals = np.diff(time_clean)

        # check if any interval is larger than 1 hour
        if np.all(intervals > np.timedelta64(1, 'h')):
            continue
        
        elif np.any(intervals > np.timedelta64(1, 'h')):
            # keep only the maximum point
            ss_clean = np.array([max_val])
            time_clean = np.array([max_time])

        time_all   = data_all.time.values
        surge_all  = data_all.surge.values[:, i]
        # Step 1: find the range of indices in data_all that overlap with ds_clean
        mask_overlap = np.isin(time_all, time_clean)
        overlap_indices = np.where(mask_overlap)[0]

        if len(overlap_indices) == 0:
            continue

        start_idx = overlap_indices[0]
        end_idx   = overlap_indices[-1]

        if threshold=='std':
            path_std = folder / f"storm_surges_v3/reanalysis_surge_std.nc"
            std1=xr.open_dataset(path_std)
            dist = np.sqrt(
                (std1['station_x_coordinate'] - lon)**2 +
                (std1['station_y_coordinate'] - lat)**2
            )
            nearest_station = dist.argmin().item()

            # Extract the std value
            threshold = std1['ss_std'].isel(stations=nearest_station).item()

        # Step 2: extend backwards until surge == 0 (or close to 0)
        while start_idx > 0 and surge_all[start_idx] >= threshold:
            start_idx -= 1

        # Step 3: extend forwards until surge == 0
        while end_idx < len(surge_all)-1 and surge_all[end_idx] >= threshold:
            end_idx += 1

        # Step 4: slice the segment
        time_segment  = time_all[start_idx:end_idx+1]
        surge_segment = surge_all[start_idx:end_idx+1]

        if surge_segment[0] < threshold and surge_segment[-1] < threshold:
            valid_stations.append(ds_clean.station.values[i])

        storm_lon = data_all['storm_longitude'].sel(time=time_segment).values
        storm_lat = data_all['storm_latitude'].sel(time=time_segment).values

        ################################################################################################
            # Create figure with custom GridSpec (2/3 map, 1/3 hydrograph)
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # mapa = 1/3, hidrograma = 2/3

        # --- Map (left, 2/3 width) ---
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_map.set_extent([lon0, lonf, lat0, latf], crs=ccrs.PlateCarree())
        ax_map.coastlines(resolution="10m", color="black", linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

        # Storm position
        ax_map.plot(data.storm_longitude, data.storm_latitude, "-", color="silver",linewidth=2, transform=ccrs.PlateCarree())
        ax_map.plot(lon, lat, "r*", markersize=4, transform=ccrs.PlateCarree())
        ax_map.plot(storm_lon, storm_lat, "-", color="red",linewidth=2, transform=ccrs.PlateCarree())

        # Remove axis ticks/labels for a clean map
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_xlabel("")
        ax_map.set_ylabel("")
        ax_map.set_title("Hydrograph location and TC position")

        # --- Hydrograph (right, 1/3 width) ---
        ax_hydro = fig.add_subplot(gs[1])
        #ax_hydro.plot(ds_clean.time, ds_clean.ss.values[:, i],'-o', color="red", markersize=3, lw=1)

            # Plot series
        ax_hydro.plot(ds_clean.time, ds_clean.ss.values[:, i], 'o', color="red", markersize=3, lw=1, label="SS cleaned")
        ax_hydro.plot(data_all.time, data_all.surge.values[:, i], '-', color="black", lw=1, label="SS raw")
        ax_hydro.plot(time_segment, surge_segment, '-', color="blue", lw=2, label="SS hydrograph")
        ax_hydro.plot(time_segment[-1], surge_segment[-1], 'o', color="blue", markersize=5, lw=1)
        ax_hydro.plot(time_segment[0], surge_segment[0], 'o', color="blue", markersize=5, lw=1)
        ax_hydro.set_ylim(-1, 3)

        ax_hydro.set_title(f"Hydrograph at lon={lon:.2f}, lat={lat:.2f} (station {i})")
        ax_hydro.set_xlabel("Time")
        ax_hydro.set_ylabel("SS (m)")
        ax_hydro.grid(True, alpha=0.3)
        ax_hydro.legend()

        # Save figure
        image_filename = path_out / f"GSTM_hydrograph_all_{i}.png"
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        t0= time_segment[0]
        tf= time_segment[-1]

        max_surge = np.max(surge_segment)
        time_max = time_segment[np.argmax(surge_segment)]
        t_asc=(time_max-t0)/ np.timedelta64(1, 'h')
        t_des=(tf-time_max)/ np.timedelta64(1, 'h')
        dur=(tf-t0)/ np.timedelta64(1, 'h')

        lat_max=data_all['storm_latitude'].sel(time=time_max).values
        lon_max=data_all['storm_longitude'].sel(time=time_max).values
        wind_max=data_all['storm_wind_speed'].sel(time=time_max).values
        dir_max=data_all['storm_direction'].sel(time=time_max).values
        rmw_max=data_all['storm_RMW'].sel(time=time_max).values
        pmin_max=data_all['storm_Pmin'].sel(time=time_max).values
        sspeed_max=data_all['storm_speed'].sel(time=time_max).values
        dist_max = haversine(lat, lon, lat_max, lon_max).values

        rel_dist_max=dist_max/rmw_max
        relangle_max=relative_angle_cyclone_shoreline (lat_max,lon_max, lat.values, lon.values)

        wind_dur = np.nanmean(data_all['storm_wind_speed'].sel(time=time_segment).values)
        dir_dur= np.nanmean(data_all['storm_direction'].sel(time=time_segment).values)
        rmw_dur= np.nanmean(data_all['storm_RMW'].sel(time=time_segment).values)
        pmin_dur= np.nanmean(data_all['storm_Pmin'].sel(time=time_segment).values)
        sspeed_dur= np.nanmean(data_all['storm_speed'].sel(time=time_segment).values)

        # Add row to DataFrame
        #df_hydrograph.loc[i] = [SID,i,lon.values,lat.values,t0, tf, max_surge, time_max, t_asc, t_des, dur,
        #                lat_max, lon_max, wind_max, dir_max, rmw_max, pmin_max, sspeed_max,
        #                dist_max,rel_dist_max, relangle_max,
        #                wind_dur, dir_dur, rmw_dur, pmin_dur, sspeed_dur]
                        
                        
                        
        df_hydrograph.loc[i] = {
            "SID": SID,
            "i": i,
            "lon": float(lon.values),
            "lat": float(lat.values),
            "t0": t0,
            "tf": tf,
            "max_surge": max_surge,
            "time_max": time_max,
            "t_asc": t_asc,
            "t_des": t_des,
            "dur": dur,
            "lat_max": lat_max,
            "lon_max": lon_max,
            "wind_max": wind_max,
            "dir_max": dir_max,
            "rmw_max": rmw_max,
            "pmin_max": pmin_max,
            "sspeed_max": sspeed_max,
            "dist_max": dist_max,
            "rel_dist_max": rel_dist_max,
            "relangle_max": relangle_max,
            "wind_dur": wind_dur,
            "dir_dur": dir_dur,
            "rmw_dur": rmw_dur,
            "pmin_dur": pmin_dur,
            "sspeed_dur": sspeed_dur}               
    

        df_hydrograph_0 = pd.DataFrame({'time': time_segment, 'surge': surge_segment})
        df_hydrograph_0.to_csv(path_out / f"GSTM_hydrograph_{i}.csv")
    
    # ======================================================
    # save results
    # ======================================================
    
    
    ds_clean1 = ds_clean.sel(station=valid_stations)
    ds_clean1.to_netcdf(path_file / f"{SID}_GSTM_ss_clean.nc")
    df_hydrograph.to_csv(path_out / f"GSTM_hydrograph_all_{ii}.csv")

    print(f"Finished processing {SID} ({basin})")
    return path_file, df_hydrograph




def tracking_gesla(basin, ii, folder, eps, min_samples, delta, threshold_peak,variable_name,pathout,ncentroids):
    # ======================================================
    # paths
    # ======================================================

    path_storm = folder / "storms"  
    path_storm.mkdir(parents=True, exist_ok=True)
    file_storm = path_storm / f'ibtracs.{basin}.list.v04r01_filtered_interpolate_v1.csv'

    if basin=='EP': #Eastern Pacific
        lat0,latf,lon0,lonf=0,60,180,-75
    elif basin=='NA': #North Atlantic
        lat0,latf,lon0,lonf=0,60,-105,0
    elif basin=='NI': #North Indian
        lat0,latf,lon0,lonf=0,60,30,100
    elif basin=='SI': #South Indian
        lat0,latf,lon0,lonf=-35,0,10,135
    elif basin=='SP': #South Pacific
        lat0,latf,lon0,lonf=-60,0,135,-120
    elif basin=='WP': #Western Pacific
        lat0,latf,lon0,lonf=0,60,100,180


    if variable_name=='GESLA':
        path_files = folder / "GESLA"
        var_='surge'

    # ======================================================
    # read storm information
    # ======================================================
    df_list = pd.read_csv(file_storm,skiprows=[1]) 
    df_list=df_list[df_list['SEASON']>1979]
    SID_list=df_list['SID'].unique()

    SID=SID_list[ii]
    # test alogrithm with Katrina TC
    print(SID)
    path_output = pathout / SID  
    path_output.mkdir(parents=True, exist_ok=True)

    data_TC=df_list[df_list['SID']==SID]
    time0=data_TC.ISO_TIME.values[0]
    timef=data_TC.ISO_TIME.values[-1]

    lon0=np.nanmin(data_TC.LON)-delta
    lonf=np.nanmax(data_TC.LON)+delta
    lat0=np.nanmin(data_TC.LAT)-delta
    latf=np.nanmax(data_TC.LAT)+delta

    # identify the range to analayse the netcdf files 
    time0 = datetime.strptime(time0, '%Y-%m-%d %H:%M:%S')
    year0=time0.year
    month0=time0.month
    timef = datetime.strptime(timef, '%Y-%m-%d %H:%M:%S')
    yearf=timef.year
    monthf=timef.month
    data_TC['datetime64']= pd.to_datetime(data_TC.ISO_TIME).copy()

    # ======================================================
    # READ files information of climatic variables, such as hs, wind, slp
    # ======================================================
    #dataset_waves=load_data(path_waves, variable_name,year0, month0, yearf, monthf)
    dataset_surge=load_data(path_files, variable_name,year0, month0, yearf, monthf,basin)

    # Extract station coordinates
    station_x = dataset_surge['station_x_coordinate'].values
    station_y = dataset_surge['station_y_coordinate'].values

    # filter data
    mask = (dataset_surge['station_y_coordinate'] >= lat0) & (dataset_surge['station_y_coordinate'] <= latf) & \
        (dataset_surge['station_x_coordinate'] >= lon0) & (dataset_surge['station_x_coordinate'] <= lonf)

    # Apply the mask to filter the stations
    surge_basin = dataset_surge.where(mask, drop=True)

    # select the data for an specfici storm 
    times = data_TC['datetime64']

    time_extended = pd.date_range(
        start=pd.to_datetime(times.iloc[0]) ,
        end=pd.to_datetime(times.iloc[-1]) + pd.Timedelta(days=10), freq="H")
    surge_basin_1 = surge_basin.interp(time=time_extended, method='linear')

    surge_basin = surge_basin.interp(time=times, method='linear')

    #TC_surge=surge_basin.copy()

    if threshold_peak=='p99':
        path_per = folder / f"GESLA/GESLA_{basin}_p99.nc"
        percentile=xr.open_dataset(path_per)
        mask = (surge_basin[var_] >= percentile.surge_p99) 
        TC_surge = surge_basin.where(mask)
    
    elif threshold_peak=='p95':
        path_per = folder / f"GESLA/GESLA_{basin}_p95.nc"
        percentile=xr.open_dataset(path_per)
        mask = (surge_basin[var_] >= percentile.surge_p95) 
        TC_surge = surge_basin.where(mask)
    else:
        mask = (surge_basin[var_] >= float(threshold_peak)) 
        TC_surge = surge_basin.where(mask)

    nt,nx= TC_surge.surge.values.shape
    x = TC_surge['lon'].values
    y = TC_surge['station_y_coordinate'].values

    # ======================================================
    var1= getattr(TC_surge, var_).values
    var1 [np.isnan(var1)]=-999

    #optimal_k=elbow_method(dynamic_kmeans, var1, k_range=range(2, 11))

    idx1, centroids_norm, centroids,_=dynamic_kmeans(0, var1, var1, var1, ncentroids)

    TC_centroid=centroids[centroids>0.1]
    TC_centroid=np.unique(TC_centroid)
    TC_centroid=TC_centroid.tolist()
    mask = np.isin(centroids, TC_centroid).all(axis=1)
    TC_idx = np.where(mask)[0]

    image_filenames, xarray_time, xarray_z, z_output, track_lat, track_lon, track_storm_speed, track_storm_dir, track_wind_speed, track_slp,track_RMW = ([] for _ in range(11))

    for i in range(len(TC_surge.time)):

            try: 
                slp0 = TC_surge.isel(time=i)
                specific_time = TC_surge.time[i]
            except: 
                slp0 = TC_surge.isel(valid_time=i)
                specific_time = TC_surge.valid_time[i]

            data_TC1 = data_TC[data_TC['datetime64'] == specific_time.values]
            #z = getattr(slp0, var_).values
            z = slp0['surge'].values
            #n,m=z.shape
            # Extract indices and apply the mask to station coordinates
            idx0_1 = idx1[i, :]

            mask = np.isin(idx0_1, TC_idx)
            z0 = z.copy()
            #z0[~mask] = np.nan
            z0 =z0[mask] 
            station_x0 = x.copy()
            station_y0 = y.copy()

            station_y0 =station_y0[mask] 
            station_x0 =station_x0[mask] 
            # Perform DBSCAN clustering
            try: 
                xy_points = np.column_stack((station_x0, station_y0))
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                idx_ltln = dbscan.fit_predict(xy_points)

                # Find centroids of clusters
                idx_position = np.unique(idx_ltln[idx_ltln != -1])  # Exclude noise (-1)
                centroids = np.array([np.mean(xy_points[idx_ltln == cluster], axis=0) for cluster in idx_position])
                centroids[centroids[:, 0] > 180, 0] -= 360
                # Calculate the distance to each centroid
                distances = np.linalg.norm(centroids - np.array([data_TC1['LON'].values[0], data_TC1['LAT'].values[0]]), axis=1)
                closest_index = np.argmin(distances)
                closest_centroid = centroids[closest_index]

                # Get the corresponding cluster and points
                centroid_position = np.where(centroids == closest_centroid)[0][0]
                TC_idx2filter = idx_position[centroid_position]
                idx2 = idx_ltln[idx_ltln == TC_idx2filter]
                station_x2 = station_x0[idx_ltln == TC_idx2filter]
                station_y2 = station_y0[idx_ltln == TC_idx2filter]
                z2 = z0[idx_ltln == TC_idx2filter]

                # once I have filter a cluster, compare the distance and if is too large do not considered this cluster 
                xcenter=np.nanmean(station_x2)
                ycenter=np.nanmean(station_y2)
                if xcenter>180: 
                    xcenter=xcenter-360

                cluster_xy=[xcenter,ycenter]
                cluster_xy=np.array(cluster_xy)

                distances_0 = np.linalg.norm(cluster_xy - np.array([data_TC1['LON'].values[0], data_TC1['LAT'].values[0]]))

                if distances_0>10:
                    station_x2 = station_x0.copy()
                    station_y2 = station_y0.copy()
                    z2 = z0.copy() +np.nan

            except:
                station_x2 = station_x0.copy()
                station_y2 = station_y0.copy()
                z2 = z0.copy()
                        
            z_output = np.full_like(z, np.nan)  # start all NaN
            for xi, yi, zi in zip(station_x2, station_y2, z2):
                # find the index in the original x,y grid
                idx = np.where((x == xi) & (y == yi))[0]
                z_output[idx] = zi
            #z_output=z_output.reshape(n,m) 
            z_output = z_output.flatten()
            z_output[z_output<0] = np.nan

            xarray_z.append(z_output)
            xarray_time.append(specific_time.time.values)
            track_lat.append(data_TC1['LAT'].values[0])
            track_lon.append(data_TC1['LON'].values[0])
            track_storm_speed.append(data_TC1['STORM_SPEED'].values[0])
            track_storm_dir.append(data_TC1['STORM_DIR'].values[0])
            track_wind_speed.append(data_TC1['WIND_COMB'].values[0])
            track_slp.append(data_TC1['Pmin'].values[0])
            track_RMW.append(data_TC1['RMW'].values[0])

            #plot_results1(data_TC, data_TC1, x, y, idx0_1.flatten(), 'hsv', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'idx',variable_name,0,5)
            #plot_results1(data_TC, data_TC1, station_x0, station_y0, z0, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'idx0',variable_name,0.1,1)
            #plot_results1(data_TC, data_TC1, station_x2, station_y2, z2, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'_GESLA',variable_name,0.5,1)
            #plot_results1(data_TC, data_TC1, x, y, z_output, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name,variable_name,0.1,1)
            #plot_results1(data_TC, data_TC1, x, y, z, 'bwr', specific_time, lon0, lonf, lat0, latf, path_output, SID, i,variable_name+'_ALL',variable_name,0,1)

    # Step 3: filter x and y at the same positions
    print('slp:',track_slp)
    c2 = np.sum(~np.isnan(xarray_z), axis=0)  # (station,)
    mask = c2 > 1
    z_stack = np.stack(xarray_z)
    xarray_z_filtered = z_stack[:, mask]
    xarray_z_filtered

    x_filtered = x[mask]
    y_filtered = y[mask]


    create_netCDF(xarray_z_filtered, xarray_time,y_filtered,x_filtered, track_lat, track_lon, track_wind_speed, track_storm_dir, track_storm_speed, track_slp,track_RMW,SID, path_output,variable_name)

    ds_filtered = surge_basin_1.sel(stations=surge_basin_1['stations'].values[mask])
    ds_extended = ds_filtered.reindex(time=time_extended, method=None)  # method=None deja NaN en los nuevos tiempos

    data_TC['time'] = pd.to_datetime(data_TC['ISO_TIME'])
    data_TC = data_TC.set_index('time')
    data_TC_extend = data_TC.reindex(time_extended)
    # Add storm variables (keep only the 'time' dimension)
    try:
        ds_extended = ds_extended.assign(
            storm_latitude=("time", data_TC_extend['LAT'].values),
            storm_longitude=("time", data_TC_extend['LON'].values),
            storm_wind_speed=("time", data_TC_extend['WIND_COMB'].values),
            storm_direction=("time", data_TC_extend['STORM_DIR'].values),
            storm_RMW=("time", data_TC_extend['RMW'].values),
            storm_Pmin=("time", data_TC_extend['Pmin'].values),
            storm_speed=("time", data_TC_extend['STORM_SPEED'].values)
        )

        output_file = path_output / f"{SID}_GESLA_all_time_series.nc"
        ds_extended.to_netcdf(output_file)
    except:
        print(f'No storm surge over thersold {threshold_peak} was detected for SID {SID}')

def hydrograph_gesla(basin, ii, threshold, folder,variable_name,pathout):
    """
    Run storm hydrograph extraction for a given basin and storm index.

    Parameters
    ----------
    basin : str
        Basin code ('NA', 'EP', 'NI', 'SI', 'SP', 'WP')
    ii : int
        Index of storm in the filtered storm list
    folder : str or Path
        Base directory path. Defaults to current working directory.
    """
    # ======================================================
    # paths
    # ======================================================
    path_storm = folder / "storms"  
    path_storm.mkdir(parents=True, exist_ok=True)
    file_storm = path_storm / f'ibtracs.{basin}.list.v04r01_filtered_interpolate_v1.csv'

    # Basin bounds
    bounds = {
        'EP': (0, 60, 180, -75),
        'NA': (0, 60, -105, 0),
        'NI': (0, 60, 30, 100),
        'SI': (-35, 0, 10, 135),
        'SP': (-60, 0, 135, -120),
        'WP': (0, 60, 100, 180)
    }
    lat0, latf, lon0, lonf = bounds[basin]

    # ======================================================
    # read storm information
    # ======================================================
    df_list = pd.read_csv(file_storm, skiprows=[1]) 
    df_list = df_list[df_list['SEASON'] > 1979]
    SID_list = df_list['SID'].unique()
    SID = SID_list[ii]

    path_file = pathout / SID  
    path_out = path_file / "hydrograph"
    path_out.mkdir(parents=True, exist_ok=True)

    data = xr.load_dataset(path_file / f"{SID}_GESLA_ss.nc")
    data = data.assign_coords({"latitude": ("station", data.latitude.values),
                               "longitude": ("station", data.longitude.values)})
    data_all = xr.load_dataset(path_file / f"{SID}_GESLA_all_time_series.nc") 
    print(data)
    ds_clean = data.dropna(dim="station", how="all", subset=["ss"])

    # ======================================================
    # main processing loop
    # ======================================================
    columns = ['SID','IDss','SS_lon','SS_lat','ss_t0','ss_tf','ss_max_surge','ss_time_max',
               'ss_tasc','ss_tdes','ss_dur','TClat_max','TClon_max','TCwind_max','TCdir_max',
               'TCrmw_max','TCpmin_max','TCsspeed_max','TCdist_max','TCwind_dur','TCdir_dur',
               'TCrmw_dur','TCpmin_dur','TCsspeed_dur']

    df_hydrograph = pd.DataFrame(columns=columns)
    valid_stations = []

    
    for i in range(len(ds_clean.station)):
        lon = ds_clean.longitude[i]
        lat = ds_clean.latitude[i]

        ss_series = ds_clean.ss.values[:, i]
        time_series = ds_clean.time.values
        valid_mask = ~np.isnan(ss_series)
        if np.sum(valid_mask) == 0:
            continue

        mask_not_nan = ~np.isnan(ds_clean.ss.values[:, i])
        time_clean = ds_clean.time.values[mask_not_nan]
        ss_clean   = ds_clean.ss.values[:, i][mask_not_nan]

        max_idx = np.argmax(ss_clean)
        max_val = ss_clean[max_idx]
        max_time = time_clean[max_idx]
        intervals = np.diff(time_clean)

        # check if any interval is larger than 1 hour
        if np.all(intervals > np.timedelta64(1, 'h')):
            continue
        
        elif np.any(intervals > np.timedelta64(1, 'h')):
            # keep only the maximum point
            ss_clean = np.array([max_val])
            time_clean = np.array([max_time])

        time_all   = data_all.time.values
        surge_all  = data_all.surge.values[:, i]

        # Step 1: find the range of indices in data_all that overlap with ds_clean
        mask_overlap = np.isin(time_all, time_clean)
        overlap_indices = np.where(mask_overlap)[0]

        if len(overlap_indices) == 0:
            continue

        start_idx = overlap_indices[0]
        end_idx   = overlap_indices[-1]


        if threshold=='std':
            path_std = folder / f"GESLA/GESLA_{basin}_std.nc"
            std1=xr.open_dataset(path_std)
            dist = np.sqrt(
                (std1['station_x_coordinate'] - lon)**2 +
                (std1['station_y_coordinate'] - lat)**2
            )
            nearest_station = dist.argmin().item()

            # Extract the std value
            threshold = std1['surge_std'].isel(stations=nearest_station).item()


        # Step 2: extend backwards until surge == 0 (or close to 0)
        while start_idx > 0 and surge_all[start_idx] >= threshold:
            start_idx -= 1

        # Step 3: extend forwards until surge == 0
        while end_idx < len(surge_all)-1 and surge_all[end_idx] >= threshold:
            end_idx += 1

        # Step 4: slice the segment
        time_segment  = time_all[start_idx:end_idx+1]
        surge_segment = surge_all[start_idx:end_idx+1]

        if surge_segment[0] < threshold and surge_segment[-1] < threshold:
            valid_stations.append(ds_clean.station.values[i])

        storm_lon = data_all['storm_longitude'].sel(time=time_segment).values
        storm_lat = data_all['storm_latitude'].sel(time=time_segment).values

        ################################################################################################
            # Create figure with custom GridSpec (2/3 map, 1/3 hydrograph)
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # mapa = 1/3, hidrograma = 2/3

        # --- Map (left, 2/3 width) ---
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_map.set_extent([lon0, lonf, lat0, latf], crs=ccrs.PlateCarree())
        ax_map.coastlines(resolution="10m", color="black", linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

        # Storm position
        ax_map.plot(data.storm_longitude, data.storm_latitude, "-", color="silver",linewidth=2, transform=ccrs.PlateCarree())
        ax_map.plot(lon, lat, "r*", markersize=4, transform=ccrs.PlateCarree())
        ax_map.plot(storm_lon, storm_lat, "-", color="red",linewidth=2, transform=ccrs.PlateCarree())

        # Remove axis ticks/labels for a clean map
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_xlabel("")
        ax_map.set_ylabel("")
        ax_map.set_title("Hydrograph location and TC position")

        # --- Hydrograph (right, 1/3 width) ---
        ax_hydro = fig.add_subplot(gs[1])
        #ax_hydro.plot(ds_clean.time, ds_clean.ss.values[:, i],'-o', color="red", markersize=3, lw=1)

            # Plot series
        ax_hydro.plot(ds_clean.time, ds_clean.ss.values[:, i], 'o', color="red", markersize=3, lw=1, label="SS cleaned")
        ax_hydro.plot(data_all.time, data_all.surge.values[:, i], '-', color="black", lw=1, label="SS raw")
        ax_hydro.plot(time_segment, surge_segment, '-', color="blue", lw=2, label="SS hydrograph")
        ax_hydro.plot(time_segment[-1], surge_segment[-1], 'o', color="blue", markersize=5, lw=1)
        ax_hydro.plot(time_segment[0], surge_segment[0], 'o', color="blue", markersize=5, lw=1)
        ax_hydro.set_ylim(-1, 3)

        ax_hydro.set_title(f"Hydrograph at lon={lon:.2f}, lat={lat:.2f} (station {i})")
        ax_hydro.set_xlabel("Time")
        ax_hydro.set_ylabel("SS (m)")
        ax_hydro.grid(True, alpha=0.3)
        ax_hydro.legend()

        # Save figure
        image_filename = path_out / f"GESLA_hydrograph_all_{i}.png"
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        t0= time_segment[0]
        tf= time_segment[-1]

        max_surge = np.max(surge_segment)
        time_max = time_segment[np.argmax(surge_segment)]
        t_asc=(time_max-t0)/ np.timedelta64(1, 'h')
        t_des=(tf-time_max)/ np.timedelta64(1, 'h')
        dur=(tf-t0)/ np.timedelta64(1, 'h')

        lat_max=data_all['storm_latitude'].sel(time=time_max).values
        lon_max=data_all['storm_longitude'].sel(time=time_max).values
        wind_max=data_all['storm_wind_speed'].sel(time=time_max).values
        dir_max=data_all['storm_direction'].sel(time=time_max).values
        rmw_max=data_all['storm_RMW'].sel(time=time_max).values
        pmin_max=data_all['storm_Pmin'].sel(time=time_max).values
        sspeed_max=data_all['storm_speed'].sel(time=time_max).values
        dist_max = haversine(lat, lon, lat_max, lon_max).values

        wind_dur = np.nanmean(data_all['storm_wind_speed'].sel(time=time_segment).values)
        dir_dur= np.nanmean(data_all['storm_direction'].sel(time=time_segment).values)
        rmw_dur= np.nanmean(data_all['storm_RMW'].sel(time=time_segment).values)
        pmin_dur= np.nanmean(data_all['storm_Pmin'].sel(time=time_segment).values)
        sspeed_dur= np.nanmean(data_all['storm_speed'].sel(time=time_segment).values)

        # Add row to DataFrame
        df_hydrograph.loc[i] = [SID,i,lon.values,lat.values,t0, tf, max_surge, time_max, t_asc, t_des, dur,
                            lat_max, lon_max, wind_max, dir_max, rmw_max, pmin_max, sspeed_max,
                            dist_max, wind_dur, dir_dur, rmw_dur, pmin_dur, sspeed_dur]
        
        df_hydrograph_0 = pd.DataFrame({'time': time_segment, 'surge': surge_segment})

        df_hydrograph_0.to_csv(path_out / f"GESLA_hydrograph_{i}.csv")

    # ======================================================
    # save results
    # ======================================================
    df_hydrograph.to_csv(path_out / f"GESLA_hydrograph_all_{ii}.csv")

    ds_clean1 = ds_clean.sel(station=valid_stations)

    print(ds_clean)
    print(ds_clean1)

    ds_clean1.to_netcdf(path_file / f"{SID}_GESLA_ss_clean.nc")

    print(f"Finished processing {SID} ({basin})")
    return path_file, df_hydrograph

def Random_forest_feature_importance(basin, var, folder,pathout):

    #all_files = glob.glob(os.path.join(root_folder, '**', 'hydrograph/GSTM_hydrograph_all_*.csv'), recursive=True)
    all_files = list(folder.rglob("hydrograph/GSTM_hydrograph_all_*.csv"))  # busca recursivamente

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    print(f"Found {len(all_files)} CSV files. Loading...")

    # =========================
    # 2️ Load CSVs into DataFrame
    # =========================
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        df['folder'] = file.parent.name
        df['id'] = file.stem.split('_')[-1]
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna()
    print(f"Total rows after concatenation and dropping NaNs: {len(data)}")

    if var == 'ssmax':
        # Define feature sets
        feat1 = ['TClat_max', 'TClon_max', 'TCwind_max', 'TCdir_max', 'TCrmw_max', 'TCpmin_max', 'TCsspeed_max', 'TCdist_max']
        feat2 = ['TCwind_max', 'TCdir_max', 'TCrmw_max','TCpmin_max', 'TCsspeed_max', 'TCdist_max']
        feat3 = ['TCdir_max', 'TCrmw_max','TCpmin_max', 'TCsspeed_max', 'TCdist_max']
        feat4 = ['TCwind_max','TCdir_max', 'TCpmin_max', 'TCsspeed_max', 'TCdist_max']
        feat5 = ['TCwind_max', 'TCpmin_max']
        feature_sets = [feat1, feat2, feat3,feat4,feat5]
        feature_sets = [feat1, feat2]
        target = 'ss_max_surge'

    elif var == 'ssdur':
        # Define feature sets
        feat1 = ['TCwind_dur', 'TCdir_dur','TCrmw_dur', 'TCpmin_dur', 'TCsspeed_dur']
        feat2 = ['SS_lon','SS_lat','TCwind_dur', 'TCdir_dur','TCrmw_dur', 'TCpmin_dur', 'TCsspeed_dur']
        feat3 = ['TCwind_max', 'TCpmin_max']
        feature_sets = [feat1, feat2, feat3]
        feature_sets = [feat1, feat2]

        target = 'ss_dur'

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    cluster_data = data.copy()

    # Loop through each predefined feature set
    for set_idx, feats in enumerate(feature_sets, start=1):
        X = cluster_data[feats].values
        y = cluster_data[target].values
        corr = cluster_data[feats].corr()

        # --- Compute correlation + p-values ---
        corr_matrix = pd.DataFrame(np.zeros((len(feats), len(feats))),
                                columns=feats, index=feats)
        pval_matrix = corr_matrix.copy()

        for var1 in feats:
            for var2 in feats:
                r, p = pearsonr(cluster_data[var1], cluster_data[var2])
                corr_matrix.loc[var1, var2] = r
                pval_matrix.loc[var1, var2] = p

        corr_flat = (corr_matrix.stack().rename("correlation")
        .reset_index().merge(pval_matrix.stack().rename("p_value").reset_index(),
            on=["level_0", "level_1"]).rename(columns={"level_0": "var1", "level_1": "var2"}))
        corr_flat["significant"] = corr_flat["p_value"] < 0.05
        corr_flat = corr_flat[corr_flat["var1"] != corr_flat["var2"]]

        corr_flat.to_csv(pathout / f'{var}_correlation_matrix_set{set_idx}.csv')

        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        scores = cross_val_score(rf, X, y, cv=cv, scoring='r2', n_jobs=-1)

        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
        feature_importances = dict(zip(feats, perm.importances_mean))

        results.append({
            'Set': f'feat{set_idx-1}',
            'n_features': len(feats),
            'R2_mean': scores.mean(),
            'R2_std': scores.std(),
            'features': json.dumps(feats),  # list → JSON string
            'feature_importances': json.dumps(feature_importances)  # dict → JSON string
        })

        print(f"\nSet {set_idx}: R² = {scores.mean():.3f} ± {scores.std():.3f}")
        for f, imp in feature_importances.items():
            print(f"  {f}: {imp:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values('R2_mean', ascending=False)
    print("\nTop results:")
    print(results_df[['Set', 'n_features', 'R2_mean']])

    # Optional: save results
    results_df.to_csv(pathout /f'{var}_RF_set_comparison.csv', index=False)

    