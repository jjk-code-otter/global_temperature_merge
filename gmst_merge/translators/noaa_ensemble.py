from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os
import struct
import requests
from tqdm import tqdm
import gzip
import shutil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def plot_map(df):

    variable = df["tas_mean"].isel(time=1776)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)

    _ = variable.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=-3, vmax=3,
        cmap="RdBu_r",
        cbar_kwargs={"shrink": 0.8, "label": "Temperature anomaly (°C)"},
    )

    ax.set_title("Map of Temperature Anomalies")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()


def read_bin(filename):
    data_list = []

    rows = 72
    cols = 36
    iterations = 2008
    ngrid = 2594 # The actual grid 2592 but the files have an extra float at the start and end.

    # Open the binary file in read mode
    with open(filename, 'rb') as f:
        for n in range(1, iterations + 1):  # Loop from 1 to 2008
            # Read the entire 2D array (72x36) of real numbers (big-endian floats)
            data_bytes = f.read(ngrid * 4)  # 4 bytes per float
            if not data_bytes:
                break  # Stop if we reach the end of the file

            # Unpack the binary data into a flat list of floats
            temp = struct.unpack(f'>{ngrid}f', data_bytes)  # Big-endian floats

            # Strip out the first and last elements and convert the flat list into a 2D NumPy array
            temp_2d = np.array(temp[1:-1], dtype=np.float32).reshape(cols, rows)

            # Roll the array to get the dateline where I want it.
            temp_2d = np.roll(temp_2d, 36, axis=1)
            # Append the transposed 2D array to the list
            data_list.append(temp_2d)

    # Convert the list of 2D arrays into a single 3D NumPy array
    data_array = np.array(data_list, dtype=np.float32)  # Shape: (2008, 36, 72)

    return data_array


def make_xarray(target_grid, times, latitudes, longitudes, variable: str = 'tas_mean') -> xa.Dataset:
    """
    Make an xarray DataFrame from a numpy array

    :param target_grid: numpy array
        3-d array shape (ntime, nlat, nlon) containing the temperature anomalies. Missing data should be np.nan
    :param times:
        1-d array (ntime) containing the times of each element
    :param latitudes: numpy array
        1-d array (nlat) containing latitudes
    :param longitudes: numpy array
        1-d array (nlon) containing longitudes
    :param variable: str
        Variable name
    :return:
    """
    ds = xa.Dataset({
        variable: xa.DataArray(
            data=target_grid,
            dims=['time', 'latitude', 'longitude'],
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
            attrs={'long_name': '2m air temperature', 'units': 'K'}
        )
    },
        attrs={'project': 'NA'}
    )

    return ds


def download_file(url, output_path):
    """
    Download a file from the specified URL and save it to the specified output path.

    Parameters:
        url (str): The URL of the file to download.
        output_path (str): The local file path to save the downloaded file.
    """
    try:
        # Send a GET request to the URL with stream enabled
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get the total file size from headers (if available)
            total_size = int(response.headers.get('content-length', 0))

            # Open the output file in binary write mode
            with open(output_path, 'wb') as file:
                # Use tqdm to display a progress bar
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress:
                    # Write data to file in chunks
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
                            progress.update(len(chunk))

        print(f"File downloaded successfully: {output_path}")
        decompress_path = Path(str(output_path).rstrip('.gz'))
        # Decompress the file if it is GZIP
        with gzip.open(output_path, 'rb') as gz_file:
            with open(decompress_path, 'wb') as decompressed_file:
                shutil.copyfileobj(gz_file, decompressed_file)
        os.remove(output_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")


data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'NOAA_ensemble'

# File details
rows = 72
cols = 36
iterations = 2008
nyears = 167

n_ensemble = 1000

output = np.zeros((nyears, n_ensemble + 1))

for i in range(n_ensemble):

    filename = data_file_dir / f'temp.ano.merg53.dat.{i + 1:04d}.gz'
    url = f'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/tmp/2019.ngt.par.ensemble/temp.ano.merg53.dat.{i + 1:04d}.gz'

    if not (data_file_dir / f'temp.ano.merg53.dat.{i + 1:04d}').exists():
        download_file(url, filename)
    filename = data_file_dir / f'temp.ano.merg53.dat.{i + 1:04d}'

    print(filename)

    # Only want whole years
    data_array = read_bin(filename)
    data_array = data_array[0:167 * 12, :, :]
    data_array[data_array < -900] = np.nan

    latitudes = np.linspace(-87.5, 87.5, 36)
    longitudes = np.linspace(-177.5, 177.5, 72)
    times = pd.date_range(start=f'1850-01-01', freq='1MS', periods=nyears * 12)

    df = make_xarray(data_array, times, latitudes, longitudes)

    # plot_map(df)

    # Open file get area weights
    weights = np.cos(np.deg2rad(df.tas_mean.latitude))

    # Calculate the area-weighted average, then the annual average
    regional_ts = df.tas_mean.weighted(weights).mean(dim=("latitude", "longitude"))
    regional_ts = regional_ts.data
    regional_ts = np.mean(regional_ts.reshape(nyears, 12), axis=1)

    # Make a time axis
    time = np.arange(1850, 1850 + nyears, 1)

    output[:, 0] = time[:]
    output[:, i + 1] = regional_ts[:]

    os.remove(filename)

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output[:, 0:i + 2], delimiter=",")
