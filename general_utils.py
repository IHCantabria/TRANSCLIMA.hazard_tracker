import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.cluster import KMeans,DBSCAN
import glob
import geopandas as gpd
from pathlib import Path
from scipy import spatial
import scipy.stats
import math 
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import nearest_points


def get_engine(filename):

    filename = str(filename)  # convierto a string

    if filename.endswith('.nc'):
        return 'netcdf4'  # or None, depending on your setup
    elif filename.endswith('.grib') or filename.endswith('.grb'):
        return 'cfgrib'
    else:
        raise ValueError(f"Unsupported file format for file: {filename}")


def load_data0(path_files,variable_name, year0, month0, yearf, monthf,basin):
    """
    Load ERA5 wave height data (`Hswv`) from GRIB files based on start and end year/month.
    
    Parameters:
    - path_waves (str): Path to the directory containing the ERA5 wave files.
    - year0 (int): Start year.
    - month0 (int): Start month.
    - yearf (int): End year.
    - monthf (int): End month.

    Returns:
    - dataset_waves (xarray.Dataset): Combined dataset containing wave height data.
    """
    if variable_name=='waves':
        # Construct file names with proper formatting
        file_name1 = f"{path_files}era5.Hswv.{year0}{month0:02d}.grb"
        file_name2 = f"{path_files}era5.Hswv.{yearf}{monthf:02d}.grb"

    elif variable_name=='swell':
        # Construct file names with proper formatting
        file_name1 = f"{path_files}era5.hsswell.{year0}{month0:02d}.grib"
        file_name2 = f"{path_files}era5.hsswell.{yearf}{monthf:02d}.grib"

    elif variable_name=='swell1':
        # Construct file names with proper formatting
        file_name1 = f"{path_files}era5.hs1swell.{year0}{month0:02d}.grib"
        file_name2 = f"{path_files}era5.hs1swell.{yearf}{monthf:02d}.grib"    

    elif variable_name=='winds':
        # Construct file names with proper formatting
        file_name1 = path_files /f"Global_10m_wind_speed_{year0}.{month0:02d}.nc"
        file_name2 = path_files /f"Global_10m_wind_speed_{yearf}.{monthf:02d}.nc"

    elif variable_name=='slp':
        # Construct file names with proper formatting
        file_name1 = path_files /f"era5.slp.{year0}{month0:02d}.nc"
        file_name2 = path_files /f"era5.slp.{yearf}{monthf:02d}.nc"

    elif variable_name=='ss':
        # Construct file names with proper formatting
        #file_name1 = f"{path_files}reanalysis_surge_hourly_{year0}_{month0:02d}_v3_shoreline.nc"
        #file_name2 = f"{path_files}reanalysis_surge_hourly_{yearf}_{monthf:02d}_v3_shoreline.nc" 
        file_name1 = path_files / f"reanalysis_surge_hourly_{year0}_{month0:02d}_v3_shoreline.nc"
        file_name2 = path_files / f"reanalysis_surge_hourly_{yearf}_{monthf:02d}_v3_shoreline.nc"

    elif variable_name=='GESLA':
        file_name1 = path_files / f"GESLA_{basin}.nc"
        file_name2 = path_files / f"GESLA_{basin}.nc"

          
    # Load the dataset(s)
    if file_name1 == file_name2:
        engine = get_engine(file_name1)
        dataset = xr.open_dataset(file_name1, engine=engine)
    else:
        engine1 = get_engine(file_name1)
        engine2 = get_engine(file_name2)

        dataset1 = xr.open_dataset(file_name1, engine=engine1)
        dataset2 = xr.open_dataset(file_name2, engine=engine2)
        dataset = xr.concat([dataset1, dataset2], dim="time")

    # Convert longitude from 0-360 to -180 to 180
    if variable_name=='ss' or variable_name=='GESLA' :
        dataset = dataset.assign_coords(
            lon=((dataset.station_x_coordinate + 180) % 360) - 180
        )
    else:
        dataset = dataset.assign_coords(
            lon=((dataset.longitude + 180) % 360) - 180)


    return dataset


def load_data(path_files, variable_name, year0, month0, yearf, monthf, basin):
    """
    Load data for waves, swell, wind, SLP, storm surge, or GESLA using ERA5 or other files.

    Parameters:
    - path_files (Path or str): Directory containing data files.
    - variable_name (str): waves, swell, swell1, winds, slp, ss, GESLA
    - year0, month0, yearf, monthf (int): Start/end year and month
    - basin (str): Basin name (only for GESLA)

    Returns:
    - xarray.Dataset
    """

    # Ensure path is a Path object
    path_files = Path(path_files)

    # ---------------------------
    # 1. Build filenames
    # ---------------------------
    if variable_name == 'waves':
        f1 = path_files / f"era5.Hswv.{year0}{month0:02d}.grb"
        f2 = path_files / f"era5.Hswv.{yearf}{monthf:02d}.grb"

    elif variable_name == 'swell':
        f1 = path_files / f"era5.hsswell.{year0}{month0:02d}.grib"
        f2 = path_files / f"era5.hsswell.{yearf}{monthf:02d}.grib"

    elif variable_name == 'swell1':
        f1 = path_files / f"era5.hs1swell.{year0}{month0:02d}.grib"
        f2 = path_files / f"era5.hs1swell.{yearf}{monthf:02d}.grib"

    elif variable_name == 'winds':
        f1 = path_files / f"Global_10m_wind_speed_{year0}.{month0:02d}.nc"
        f2 = path_files / f"Global_10m_wind_speed_{yearf}.{monthf:02d}.nc"

    elif variable_name == 'slp':
        f1 = path_files / f"era5.slp.{year0}{month0:02d}.nc"
        f2 = path_files / f"era5.slp.{yearf}{monthf:02d}.nc"

    elif variable_name == 'ss':
        f1 = path_files / f"reanalysis_surge_hourly_{year0}_{month0:02d}_v3_shoreline.nc"
        f2 = path_files / f"reanalysis_surge_hourly_{yearf}_{monthf:02d}_v3_shoreline.nc"

    elif variable_name == 'GESLA':
        f1 = path_files / f"GESLA_{basin}.nc"
        f2 = f1  # Same file

    else:
        raise ValueError(f"Unknown variable_name: {variable_name}")

    # ---------------------------
    # 2. Load dataset(s)
    # ---------------------------
    if f1 == f2:
        eng = get_engine(f1)
        ds = xr.open_dataset(f1, engine=eng)
    else:
        e1 = get_engine(f1)
        e2 = get_engine(f2)

        ds1 = xr.open_dataset(f1, engine=e1)
        ds2 = xr.open_dataset(f2, engine=e2)

        # concat only along time
        ds = xr.concat([ds1, ds2], dim="time")

    # ---------------------------
    # 3. Fix longitude naming per dataset type
    # ---------------------------
    if variable_name in ['ss', 'GESLA']:
        lon_raw = ds.get("station_x_coordinate")
    else:
        lon_raw = ds.get("longitude")

    if lon_raw is not None:
        ds = ds.assign_coords(
            lon=((lon_raw + 180) % 360) - 180
        )

    return ds


def sort_clusters(idx2, centroids):
    """
    Reorders clusters based on the first column of centroids and updates idx2 accordingly.

    Parameters:
    - idx2 (numpy.ndarray): Array containing cluster indices.
    - centroids (numpy.ndarray): Array of cluster centroids.

    Returns:
    - idx2 (numpy.ndarray): Updated array with reordered cluster indices.
    - centroids (numpy.ndarray): Sorted centroids array.
    """

    # Generate original indices
    original_indices = np.arange(len(centroids))

    # Sort centroids based on the first column
    sorted_indices = np.argsort(centroids[:, 0])
    centroids = centroids[sorted_indices]

    # Update idx2 based on sorted indices
    idx2 = np.where(idx2 >= 0, sorted_indices[idx2] + 10, idx2)  # Ensure only valid indices are updated
    idx2 -= 10  # Restore original range

    return idx2, centroids


def static_kmeans(var1,var2, ik):
    # Get dimensions of the input 3D matrix
    nr, nc = var1.shape

    # Flatten the 3D arrays into 1D
    var1 = var1.flatten()
    var2 = var2.flatten()

    # Normalize var1
    Tmin, Tmax = np.min(var1), np.max(var1)
    VAR1 = (var1 - Tmin) / (Tmax - Tmin)

    # Normalize var2
    dcmin, dcmax = np.min(var2), np.max(var2)
    VAR2 = (var2 - dcmin) / (dcmax - dcmin)

    # Combine the variables into a single array for clustering
    VAR = np.column_stack((VAR1, VAR2))

    # Perform k-means clustering with computed centroids
    kmeans = KMeans(n_clusters=ik, random_state=42)
    kmeans.fit(VAR)
    idx = kmeans.labels_
    inertia=kmeans.inertia_

    # Denormalize centroids
    C = kmeans.cluster_centers_
    CC = np.zeros_like(C)
    CC[:, 0] = C[:, 0] * (Tmax - Tmin) + Tmin
    CC[:, 1] = C[:, 1] * (dcmax - dcmin) + dcmin

    # Reshape the cluster assignments back to the original 3D shape
    idx = idx.reshape((nr, nc))

    return idx, CC,inertia


def dynamic_kmeans(centknowed, var1, var2, var3, ik):
    """
    Apply dynamic k-means clustering.

    Parameters:
    centknowed (int): 1 if centroids are known, 0 otherwise.
    var1 (numpy.ndarray): 3D array (time, lat, lon) for variable 1.
    var2 (numpy.ndarray): 3D array (time, lat, lon) for variable 2.
    var3 (numpy.ndarray): 3D array (time, lat, lon) for variable 3.
    ik (int): Optimal number of clusters.

    Returns:
    numpy.ndarray: Cluster assignments in the original 3D shape.
    """
    # Get dimensions of the input 3D matrix

    dim=len (var1.shape)
    
    if dim==2:
        nr, nc = var1.shape
    elif dim==3:
        nr, nc, nt = var1.shape

    

    # Flatten the 3D arrays into 1D
    var1 = var1.flatten()
    var2 = var2.flatten()
    var3 = var3.flatten()

    # Normalize var1
    Tmin, Tmax = np.nanmin(var1), np.nanmax(var1)
    VAR1 = (var1 - Tmin) / (Tmax - Tmin)

    # Normalize var2
    dcmin, dcmax = np.nanmin(var2), np.nanmax(var2)
    VAR2 = (var2 - dcmin) / (dcmax - dcmin)

    # Normalize var3
    dsmin, dsmax = np.nanmin(var3), np.nanmax(var3)
    VAR3 = (var3 - dsmin) / (dsmax - dsmin)

    # Combine the variables into a single array for clustering
    VAR = np.column_stack((VAR1, VAR2, VAR3))
    VAR = np.nan_to_num(VAR, nan=0, posinf=0, neginf=0)

 
    if centknowed == 1:
        # Load centroids from file
        CC1 = np.loadtxt('centroids.txt')

        # Normalize centroids
        CC1[:, 0] = (CC1[:, 0] - Tmin) / (Tmax - Tmin)
        CC1[:, 1] = (CC1[:, 1] - dcmin) / (dcmax - dcmin)
        CC1[:, 2] = (CC1[:, 2] - dsmin) / (dsmax - dsmin)

        # Perform k-means clustering with known centroids
        kmeans = KMeans(n_clusters=ik, init=CC1, n_init=1, random_state=42)
        kmeans.fit(VAR)
        idx = kmeans.labels_

    elif centknowed == 0:
        # Perform k-means clustering with computed centroids
        kmeans = KMeans(n_clusters=ik, random_state=42)
        kmeans.fit(VAR)
        idx = kmeans.labels_
        inertia=kmeans.inertia_

        # Denormalize centroids
        C = kmeans.cluster_centers_
        CC = np.zeros_like(C)
        CC[:, 0] = C[:, 0] * (Tmax - Tmin) + Tmin
        CC[:, 1] = C[:, 1] * (dcmax - dcmin) + dcmin
        CC[:, 2] = C[:, 2] * (dsmax - dsmin) + dsmin

        #score = silhouette_score(VAR, kmeans.labels_)
        # Save centroids
        np.savetxt('centroids.txt', CC, fmt='%.6f')
        np.savetxt('centroidsNorm.txt', C, fmt='%.6f')

    # Reshape the cluster assignments back to the original 3D shape

    if dim==2:
        idx = idx.reshape((nr, nc))
    elif dim==3:
        idx = idx.reshape((nr, nc, nt))


    # Save the cluster assignments
    np.save('idx.npy', idx)
    return idx,C, CC,inertia


def  plot_results(data_TC,lon0, lonf,lat0, latf, x0, y0, z,cmap,SID,path_output,variable_name):

    if variable_name=='ss':
        set_label1='ss (m), '
    elif variable_name=='slp':
        set_label1='slp (hpa), '
    elif variable_name=='winds':
        set_label1='wind speed (m/s)'
    elif variable_name=='waves':
        set_label1='wave height (m), '
    
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('gainsboro')  # Background color for map

    ax.coastlines(resolution='10m', color='white', facecolor='gray', linewidth=1.2) 
    ax.add_feature(cfeature.LAND, facecolor='silver')

    ax.plot(data_TC['LON'], data_TC['LAT'], color='black', linewidth=1) 
    ax.plot(data_TC['LON'], data_TC['LAT'], '.', color='black',markersize=5)

    # Scatter plot of the selected stations
    #norm = plt.Normalize(5, 15)
    scat = ax.scatter(x0, y0, c=z, cmap=cmap, s=10, marker='o', transform=ccrs.PlateCarree())

    # Set the title and color bar
    ax.set_title(SID)

    cbar = plt.colorbar(plt.cm.ScalarMappable( cmap=cmap), ax=ax)
    cbar.set_label(f"{set_label1}")

    # Set the axis limits and save the plot
    ax.set_xlim([lon0, lonf])
    ax.set_ylim([lat0, latf])
    ax.gridlines(draw_labels=True, color='white', linestyle='--', alpha=0.7, linewidth=1)
    image_filename = f"{path_output}/{SID}_footprint_{variable_name}.png"
    plt.savefig(image_filename)
    plt.close() 


def elbow_method(dynamic_kmeans, var2, k_range=range(2, 11)):
    """
    Finds the optimal number of clusters (k) using the elbow method with the maximum perpendicular distance approach.
    
    Parameters:
        dynamic_kmeans (function): The k-means function that returns inertia.
        var2: The input data used for clustering.
        k_range (range): Range of k values to test (default is 2 to 10).
    
    Returns:
        int: Optimal number of clusters (k).
    """
    inertia_array = []
    
    # Compute inertia for each k
    for k in k_range:
        _, _,_, inertia = dynamic_kmeans(0, var2, var2, var2, k)
        inertia_array.append(inertia)
    
    points = np.array(list(zip(k_range, inertia_array)))
    line_start, line_end = points[0], points[-1]
    
    # Calculate perpendicular distances from each point to the line
    distances = []
    for point in points:
        distance = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        distances.append(distance)
    
    # Find k with the maximum distance (optimal k)
    optimal_k = k_range[np.argmax(distances)]
    
    return optimal_k


def plot_results1(data_TC, data_TC1, station_x2, station_y2, z2, cmap, specific_time, lon0, lonf, lat0, latf, output_dir, SID, i,variable_name,var_name,vmin,vmax):
    """
    Plots wave data on a map using Cartopy and saves the figure.

    Parameters:
    - data_TC (DataFrame): Track data containing 'LON' and 'LAT' for the storm track.
    - data_TC1 (DataFrame): Selected storm points containing 'LON' and 'LAT'.
    - station_x2, station_y2 (array-like): Station coordinates for scatter plot.
    - z2 (array-like): Color-coded variable for stations.
    - cmap (colormap): Colormap for the scatter plot.
    - specific_time (datetime): Time reference for the title.
    - lon0, lonf (float): Longitude limits for the plot.
    - lat0, latf (float): Latitude limits for the plot.
    - output_dir (str): Directory to save the image.
    - SID (str): Storm ID for filename.
    - i (int): Index for saving multiple images.

    Returns:
    - image_filename (str): The path of the saved image.
    """

    if var_name=='ss' or var_name=='GESLA':
        set_label1='ss (m)'
    elif var_name=='slp':
        set_label1='slp (hpa)'
    elif var_name=='winds':
        set_label1='wind speed (m/s)'
    elif var_name=='waves':
        set_label1='wave height (m)'


    # Create figure and axis with Cartopy projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('gainsboro')  # Background color for the map

    # Add coastlines and land features
    #ax.coastlines(resolution='10m', color='white', linewidth=1.2)
    folder = os.getcwd()
    folder = Path(folder)
    path_coast = folder / "ne_10m_coastline/ne_10m_coastline.shp" 
    path_coast=str(path_coast)
    gdfc = gpd.read_file(path_coast)
    gdfc.plot(ax=ax, color='white',  linewidth=1.2)


    path_land = folder / "ne_110m_land/ne_110m_land.shp" 
    path_land=str(path_land)
    gdf = gpd.read_file(path_land)
    gdf.plot(ax=ax, facecolor='silver')

    # Plot storm track
    ax.plot(data_TC['LON'], data_TC['LAT'], color='black', linewidth=0.7)
    ax.plot(data_TC['LON'], data_TC['LAT'], '.', color='black', markersize=2)
    ax.plot(data_TC1['LON'], data_TC1['LAT'], 'o', markersize=8, color='black')

    # Scatter plot of selected stations
    norm = plt.Normalize(vmin, vmax)
    scat = ax.scatter(station_x2, station_y2, c=z2, cmap=cmap, norm=norm, s=10, marker='o', transform=ccrs.PlateCarree())

    # Set title and colorbar
    ax.set_title(f'{var_name} at Time: {specific_time.values}')

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(f"{set_label1}")
    # Set axis limits and gridlines
    ax.set_xlim([lon0, lonf])
    ax.set_ylim([lat0, latf])
    ax.gridlines(draw_labels=True, color='white', linestyle='--', alpha=0.7, linewidth=1)

    # Save and close the plot
    image_filename = f"{output_dir}/{SID}_{variable_name}_{i}.png"
    #print(image_filename)
    plt.savefig(image_filename)
    plt.close()

    return image_filename


def function_DBSCAN_XX(station_x, station_y, eps=2.5, min_samples=2):
    """
    Perform DBSCAN clustering on (x, y) coordinates and compute centroids of clusters.
    """
    xy_points = np.column_stack((station_x, station_y))  # Combine x and y coordinates
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    idx_ltln = dbscan.fit_predict(xy_points)
    print(idx_ltln)

    # Find unique clusters (excluding noise labeled as -1)
    idx_position = np.unique(idx_ltln[idx_ltln != -1])
    print(idx_position)

    # Compute centroids for each cluster
    centroids = np.array([np.mean(xy_points[idx_ltln == cluster], axis=0) for cluster in idx_position])
    print(centroids)

    return idx_ltln, idx_position, centroids  # Ensure all three values are returned


def find_closest_centroid(centroids, data_TC1, idx_position):
    """
    Find the closest centroid to a given storm location and return its cluster index.

    Parameters:
        centroids (numpy array): Array of centroid coordinates (shape: [n_clusters, 2]).
        data_lon (float): Longitude of the target point (e.g., storm location).
        data_lat (float): Latitude of the target point (e.g., storm location).
        idx_position (numpy array): Indices of clusters.

    Returns:
        closest_centroid (numpy array): Coordinates of the closest centroid.
        TC_idx2filter (int): Corresponding cluster index.
    """
    # Compute distances from the given point to all centroids
    distances = np.linalg.norm(centroids - np.array([data_TC1['LON'].values[0], data_TC1['LAT'].values[0]]), axis=1)

    # Find the index of the closest centroid
    closest_index = np.argmin(distances)
    closest_centroid = centroids[closest_index]

    # Get the corresponding cluster and points
    centroid_position = np.where((centroids == closest_centroid).all(axis=1))[0][0]
    TC_idx2filter = idx_position[centroid_position]
    

    return closest_centroid, TC_idx2filter


def create_netCDF(xarray_z, xarray_time, x, y, track_lat, track_lon, track_wind_speed, track_storm_dir, track_storm_speed, track_slp,track_RMW,SID, output_dir,variable_name):
    """
    Creates a storm track dataset with wave data and exports it to a NetCDF file.
    
    Parameters:
        xarray_z (array-like): Wave height data.
        xarray_time (array-like): Time coordinates.
        x (array-like): Latitude coordinates.
        y (array-like): Longitude coordinates.
        track_lat (array-like): Storm latitude values.
        track_lon (array-like): Storm longitude values.
        track_wind_speed (array-like): Storm wind speeds.
        track_storm_dir (array-like): Storm directions.
        track_storm_speed (array-like): Storm speeds.
        SID (str): Storm identifier.
        output_dir (str): Directory to save the NetCDF file.
    
    Returns:
        str: Path to the saved NetCDF file.
    """

    if variable_name=='ss':
        waves_data = xr.DataArray(
        data=np.array(xarray_z),
        dims=["time", "station"],
        coords={
            "time": np.array(xarray_time),
            "station": np.arange(len(x)),  # station index (0, 1, 2, ...)
            "latitude": ("station", y),    # latitude for each station
            "longitude": ("station", x),   # longitude for each station
        },
        name=variable_name,
        attrs={
            "units": "m",
            "description": f"Storm surge elevation for storm: {SID}"
        })

    elif variable_name=='GESLA':
        waves_data = xr.DataArray(
        data=np.array(xarray_z),
        dims=["time", "station"],
        coords={
            "time": np.array(xarray_time),
            "station": np.arange(len(x)),  # station index (0, 1, 2, ...)
            "latitude": ("station", y),    # latitude for each station
            "longitude": ("station", x),   # longitude for each station
        },
        name='ss',
        attrs={
            "units": "m",
            "description": f"Storm surge elevation for storm: {SID}"
        })

    elif variable_name=='slp':
        # Create the wind gust DataArray
        waves_data = xr.DataArray(
            data=np.array(xarray_z),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": np.array(xarray_time),
                "latitude": y,
                "longitude": x,
            },
            name=variable_name,
            attrs={"units": "hPa", "description": f'Sea Level Pressure for storm: {SID}'}
        )

    elif variable_name=='winds':
        # Create the wind gust DataArray
        waves_data = xr.DataArray(
            data=np.array(xarray_z),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": np.array(xarray_time),
                "latitude": y,
                "longitude": x,
            },
            name=variable_name,
            attrs={"units": "m/s", "description": f'wind gust speed at 10 m for storm: {SID}'}
        )

    elif variable_name=='waves':
        # Create the wind gust DataArray
        waves_data = xr.DataArray(
            data=np.array(xarray_z),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": np.array(xarray_time),
                "latitude": y,
                "longitude": x,
            },
            name=variable_name,
            attrs={"units": "m", "description": f'Significant Wave Height for storm: {SID}'}
        )


    # Create the storm track Dataset
    storm_track = xr.Dataset(
        data_vars={
            'ss': waves_data,
            "storm_latitude": ("time", track_lat),
            "storm_longitude": ("time", track_lon),
            "storm_wind_speed": ("time", track_wind_speed),
            "storm_direction": ("time", track_storm_dir),
            "storm_speed": ("time", track_storm_speed),
            "storm_slp": ("time", track_slp),
            "storm_RMW": ("time", track_RMW),           
            
        },
        coords={
            "time": np.array(xarray_time),
            "latitude": y,
            "longitude": x,
        },
        attrs={"description": f'Storm track and {variable_name} for: {SID}'}
    )

    # Save to NetCDF
    if variable_name=='GESLA':
        ncfile = f"{output_dir}/{SID}_GESLA_ss.nc"
    else:
        ncfile = f"{output_dir}/{SID}_ERA5_{variable_name}.nc"

    storm_track.to_netcdf(ncfile)
    
    return ncfile


def create_video_from_images(folder_path,SID, variable_name, frame_width=2000, frame_height=1250, fps=10, clean_up=False):
    """
    Creates a video from a list of image filenames.
    
    Parameters:
        image_filenames (list): List of image file paths.
        video_filename (str): Name of the output video file.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
        fps (int): Frames per second for the video.
        clean_up (bool): If True, deletes the images after creating the video.
    
    Returns:
        str: Path to the saved video file.
    """

    pattern = os.path.join(folder_path, SID+'_'+variable_name+'_*.png')

    # List all matching filenames
    image_filenames = glob.glob(pattern)
    image_filenames = natsort.natsorted(image_filenames)



    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_filename=os.path.join(folder_path, SID+'_'+variable_name+'.mp4')

    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
    
    for image_filename in image_filenames:
        img = cv2.imread(image_filename)
        img_resized = cv2.resize(img, (frame_width, frame_height))
        out.write(img_resized)
    
    out.release()
    
    if clean_up:
        for image_filename in image_filenames:
            os.remove(image_filename)
    
    return video_filename


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (specified in decimal degrees).

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point in degrees.
    lat2, lon2 : float
        Latitude and longitude of the second point in degrees.

    Returns
    -------
    distance : float
        Distance in kilometers.
    """
    # Earth radius in km
    R = 6371.0

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance



#==============================================================================
# Extend a line 
#============================================================================== 
def extend_line (TC_line,extension_distance):
   
    # Create the original LineString with lat/lon coordinates
    line = TC_line
    
    
    # Extract the first and last points
    start_point = Point(line.coords[0])
    end_point = Point(line.coords[-1])
    
    # Calculate the angle between the first and second points
    angle = math.atan2(line.coords[1][1] - line.coords[0][1], line.coords[1][0] - line.coords[0][0])
    
    # Calculate the new coordinates for the extended points
    new_start_x = start_point.x - extension_distance * math.cos(angle)
    new_start_y = start_point.y - extension_distance * math.sin(angle)
    
    new_end_x = end_point.x + extension_distance * math.cos(angle)
    new_end_y = end_point.y + extension_distance * math.sin(angle)
    
    # Create the new LineString with extended points
    extended_line = LineString([(new_start_x, new_start_y)] + list(line.coords[1:-1]) + [(new_end_x, new_end_y)])

    return extended_line

#==============================================================================
# Track complexity index that results from the accumulated bearing angle of the
# simplied track previosly computed with  douglas_peucker algorithm
#==============================================================================      
def track_complexity_index (simplified_points):
    #this function calculate an accumulated bearing angle for the track
    x, y = zip(*simplified_points)
    result=0
    for i in range(1,len(simplified_points)):
        point1=[y[i],x[i]]
        point2=[y[i-1],x[i-1]]
        ang=bearing_angle(point1, point2)
        result=result+ang
    return result

#==============================================================================
# Function to compute the angle between lines
# angle convection(anticlockwise=positive, clockwise=negative) 
#==============================================================================   
def angle_between_lines(m1,m2):

    if m1 is not None and m2 is not None:
        angle_rad = np.arctan2(m2 - m1, 1 + m1 * m2)
    else:
        angle_rad = np.pi / 2  # 90 degrees
    
    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)
    
    # For angles higher than +-90 return the complementary angle

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    
    return angle_deg
    
#==============================================================================
# Calculate the tangent of the shoreline on the intersection point
#==============================================================================  
def tangent_shoreline(intersection, intersection_point,coastline_gdf):

    if intersection_point.any():
        # If there are multiple intersections, you can loop through them
        
        for geom in intersection_point:
            if geom.geom_type == 'Point':
                x, y = geom.xy
                target_longitude=x[0] 
                target_latitude=y[0] 
            elif geom.geom_type == 'MultiPoint':
                #extract the first point
                x, y = point.xy  
                target_longitude=x[0] 
                target_latitude=y[0] 

    # Extract the geometry of the shoreline where the intersectoin occurrs
    print(intersection[intersection==True])
    geometry =coastline_gdf['geometry'][coastline_gdf['geometry'][intersection==True].index[0]]
        
    # Convert the shoreline to multipoint
    mp_shoreline = MultiPoint(list(geometry.coords))

    # Obtaint the nearest point of the shoreline to the intersecction 
    nearest = nearest_points(Point(target_longitude, target_latitude), mp_shoreline)
    nearest_point, target_point = nearest
    
    # Extract the nearest point and the two sourranded points        
    list_points=list(geometry.coords)
    distance=[sqrt(pow(list_points[i][0]-nearest_point.x,2)+pow(list_points[i][1]-nearest_point.y,2))  for i in range(len(list_points))]
    pos=distance.index(min(distance))
    pos0=pos-1
    posf=pos+1

    # gnerate a line with this three points      
    short_shoreline=list_points[pos0:posf]
    short_shoreline_y=[short_shoreline[i][1]  for i in range(len(short_shoreline))]
    short_shoreline_x=[short_shoreline[i][0] for i in range(len(short_shoreline))]
    
    # Create the tangent with a polyfit of the short_shoreline
    slope_tangent, intercept = np.polyfit(short_shoreline_x,short_shoreline_y,1)
    
    # Calculate the perpendicular of the short_shoreline tangent
    slope_perpendicular= -1 / slope

    return slope_tangent,slope_perpendicular

#==============================================================================
# Plot the lines to check if angles are ok  
#============================================================================== 
def plot_intersections(slope1,slope2,nearest,geometry,short_shoreline_x,short_shoreline_y):
    # Plot to check if all lines are generated correctly 
    
    # Generate x-values for the first line
    x1 = np.linspace(-1, 1, 10)
    
    # Calculate the corresponding y-values for the first line
    y1 = slope1 * x1
   
    # Generate x-values for the second line
    x2 = np.linspace(-1, 1, 10)
    
    # Calculate the corresponding y-values for the second line
    y2 = slope2 * x2
    
    # Displace the corresponding y-values for the second line    
    y1 = y1 + nearest[0].y
    y2 = y2 + nearest[0].y
    x2 = x2 + nearest[0].x
    x1 = x1 + nearest[0].x

    # Plot tangent and perpendicular lines  
    fig, ax = plt.subplots()
    
    plt.plot(x1,y1,'black') # perpendicular
    plt.plot(x2,y2,'gray') # tangent
    
    # Extract the coordinates of the LINESTRING geometry
    coords = list(geometry.coords)
    
    # Separate latitude and longitude into separate lists
    longitudes,latitudes = zip(*coords)
    
    # Plot shoreline   
    plt.plot(longitudes,latitudes)
    # plt.plot(short_shoreline_x,short_shoreline_y,'+', markersize=20)

    # Plot intersection 
    x, y = TC_line.xy
    plt.plot(x, y, color='r')

#==============================================================================
# Calculate the relative angle between the coast and the tC 
#==============================================================================   
def relative_angle (Lat_water,Lon_water,Lat_land,Lon_land):

    # Change tropical cyclone coordinats from (0,360) to (-180, 180)
    if Lon_water >180: Lon_water=Lon_water-360
    if Lon_land >180: Lon_land=Lon_land-360
    
    # Create the line for the TC, with the consecutive points water land
    TC_line = LineString([(Lon_water,Lat_water), (Lon_land,Lat_land)])
    
    # Calculate the slope of the TC_line
    TC_x, TC_y = TC_line.xy
    slope_track_shoreline=  (TC_y[1] - TC_y[0])/ (TC_x[1] - TC_x[0])
    
    # Load the shapefile that containes the global coastline save in a foder with name (ne_10m_coastline) in the path where this code is executed 
    current_directory = os.getcwd()
    shapefile_path = current_directory+'/ne_10m_coastline/ne_10m_coastline.shp'
    coastline_gdf = gpd.read_file(shapefile_path)
    
    # Generate intersection between coastline and TC_line
    intersection = coastline_gdf['geometry'].intersects(TC_line)
    
    # Get the point fo intersection between coastline and TC_line
    intersection_point = coastline_gdf['geometry'].intersection(TC_line)

    # Calculate the slopes (tangent and perpendicular of the shoreline postion stline and TC_line
    slope_tangent,slope_perpendicular=tangent_shoreline(intersection_point, intersection_point,coastline_gdf)

    # x2 = np.linspace(-1, 1, 10)
    # y2 = slope_perpendicular * x2
    # y2 = y2 + nearest[0].y
    # x2 = x2 + nearest[0].x
    # _, intercept = np.polyfit(x2,y2,1)

    # Calculate the angle between tropical cyclone and perpendicular of the shoreline
    #           The angle is calculate from the perpendicular of the shoreline in the intersection point between shoreline and tropical cyclone
    #           Angle convection clockwise (negative) and anticlockwise (positive)
    angle_Between_TC_shoreline=angle_between_lines(slope_perpendicular,slope_track_shoreline)

    # Plot to check if all lines are generated correctly (activate this line if you want to check it) 
    # plot_intersections(slope_perpendicular,slope_tangent,nearest,geometry,short_shoreline_x,short_shoreline_y)
    return angle_Between_TC_shoreline

#==============================================================================
# Bearing angle of point 2 from point 1
#============================================================================== 
def bearing_angle(point1, point2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

    # Calculate the longitude difference
    delta_lon = lon2 - lon1

    # Calculate the bearing angle using the arctan2 function
    y = math.sin(delta_lon)
    x = math.cos(lat1) * math.tan(lat2) - math.sin(lat1) * math.cos(delta_lon)
    angle = math.atan2(y, x)

    # Convert the angle from radians to degrees
    bearing = math.degrees(angle)

    # Ensure the result is between 0 and 360 degrees
    bearing = (bearing + 360) % 360

    return bearing

#==============================================================================
# Bearing angle of point 2 from point 1
#============================================================================== 
def douglas_peucker(points, epsilon):
    if len(points) <= 2:
        return [points[0], points[-1]]

    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = np.abs(np.linalg.norm(np.cross(points[i] - points[0], points[end] - points[0])) / np.linalg.norm(points[end] - points[0]))
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        rec_results1 = douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)
        results = rec_results1[:-1] + rec_results2
    else:
        results = [points[0], points[end]]

    return results


#==============================================================================
# Calculate the relative angle between tropical cyclone and shoreline
#============================================================================== 
def relative_angle_cyclone_shoreline (TClat,TClon, LocalLat, LocalLon):
        
    # Change tropical cyclone coordinats from (0,360) to (-180, 180)
    if LocalLon >180: LocalLon=LocalLon-360
    if TClon >180: TClon=TClon-360
    
    # Create the line for the TC, with the consecutive points water land
    TC_line = LineString([(TClon,TClat), (LocalLon,LocalLat)])
    
    # Calculate the slope of the TC_line
    TC_x, TC_y = TC_line.xy

    # angle between local point (hydropgraph location) and TC position
    slope_track_shoreline = (TC_y[1] - TC_y[0]) / (TC_x[1] - TC_x[0]) 

    # Load the shapefile that containes the global coastline save in a foder with name (ne_10m_coastline) in the path where this code is executed 
    current_directory = os.getcwd()
    shapefile_path = current_directory+'/ne_10m_coastline/ne_10m_coastline.shp'
    coastline_gdf = gpd.read_file(shapefile_path)
    
    # Generate intersection between coastline and TC_line
    intersection = coastline_gdf['geometry'].intersects(TC_line)
    
    # Get the point fo intersection between coastline and TC_line
    intersection_point = coastline_gdf['geometry'].intersection(TC_line)

    # extend the line until TC_line intersects the shoreline
    while not intersection_point.any():
        extended_line=extend_line(TC_line,0.3)
        TC_x, TC_y=extended_line.xy

        # Generate intersection between coastline and TC_line
        intersection = coastline_gdf['geometry'].intersects(extended_line)
        
        # Get the point fo intersection between coastline and TC_line
        intersection_point = coastline_gdf['geometry'].intersection(extended_line)
        TC_line=extended_line

    # If there are multiple intersections, you can loop through them
    geom=intersection_point[intersection==True].iloc[0]
    if geom.geom_type == 'Point':
        x, y = geom.xy
        target_longitude=x[0] 
        target_latitude=y[0] 
        
    elif geom.geom_type == 'MultiPoint':
        # multi_point_geometries = intersection_point[intersection==True]
        # for multi_point in multi_point_geometries:
        for point in geom.geoms:
            x, y = point.xy
            target_longitude=x[0] 
            target_latitude=y[0] 

    # Extract the geometry of the shoreline where the intersectoin occurrs
    geometry =coastline_gdf['geometry'][coastline_gdf['geometry'][intersection==True].index[0]]
        
    # Convert the shoreline to multipoint
    mp_shoreline = MultiPoint(list(geometry.coords))

    # Obtaint the nearest point of the shoreline to the intersecction 
    nearest = nearest_points(Point(target_longitude, target_latitude), mp_shoreline)
    nearest_point, target_point = nearest
    
    # Extract the nearest point and the two sourranded points        
    list_points=list(geometry.coords)
    
    distance=[math.sqrt(pow(list_points[i][0]-nearest_point.x,2)+pow(list_points[i][1]-nearest_point.y,2))  for i in range(len(list_points))]
    pos=distance.index(min(distance))
    pos0=pos-1
    posf=pos+1
    pos0, pos, posf = (0, 1, 2) if pos == 0 else (pos0, pos, posf)
    # plt.plot(pos)

    # gnerate a line with this three points      
    short_shoreline=list_points[pos0:posf]
    short_shoreline_y=[short_shoreline[i][1]  for i in range(len(short_shoreline))]
    short_shoreline_x=[short_shoreline[i][0] for i in range(len(short_shoreline))]
    
    # Create the tangent with a polyfit of the short_shoreline
    slope_tangent, intercept = np.polyfit(short_shoreline_x,short_shoreline_y,1)
    
    # Calculate the perpendicular of the short_shoreline tangent
    slope_perpendicular= -1 / slope_tangent

    # Calculate the angle between tropical cyclone and perpendicular of the shoreline
    #           The angle is calculate from the perpendicular of the shoreline in the intersection point between shoreline and tropical cyclone
    #           Angle convection clockwise (negative) and anticlockwise (positive)
    angle_Between_TC_shoreline=angle_between_lines(slope_perpendicular,slope_track_shoreline)
        
    return angle_Between_TC_shoreline
