from pathlib import Path
import numpy as np
import os
import struct
import requests
import gzip
import shutil
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries

def read_bin(filename):
    data_list = []

    rows = 72
    cols = 36
    iterations = 2008 # Number of months
    ngrid = 2594 # The actual grid 2592 but the files have an extra float at the start and end

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

data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'NOAAGlobalTempv5.0'

output = np.arange(1850,2017).reshape((-1,1))
for i in range(1000):
    output_path = data_file_dir / f'temp.ano.merg5.dat.{i + 1:04d}.gz'
    decompress_path = Path(str(output_path).rstrip('.gz'))
    url = f'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/tmp/2019.ngt.par.ensemble/temp.ano.merg5.dat.{i + 1:04d}.gz'
    if not decompress_path.exists():
        try:
            # Send a GET request to the URL with stream enabled
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Open the output file in binary write mode
                with open(output_path, 'wb') as file:
                    # Write data to file in chunks
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
            # print(f"File downloaded successfully: {output_path}")
            # Decompress the file if it is GZIP
            with gzip.open(output_path, 'rb') as gz_file:
                with open(decompress_path, 'wb') as decompressed_file:
                    shutil.copyfileobj(gz_file, decompressed_file)
            os.remove(output_path)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download file: {e}")

    data_array = read_bin(decompress_path)
    data_array[data_array < -900] = np.nan # Relic from past code; there should be no missing values if using unmasked ensembles
    data_array = monthly_to_annual_timeseries(gridded_to_timeseries(np.transpose(data_array)),1850)
    output = np.append(output,data_array,axis=1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.16f', delimiter=",")
