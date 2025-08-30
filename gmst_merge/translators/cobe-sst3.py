from pathlib import Path
import netCDF4
import numpy as np
import os
import requests
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries

data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'COBE-SST3'

n_ensemble = 300
delete_large_data_files = True # storing COBE-SST3 gridded ensembles requires 12 TB of data
start_year = 1890
end_year = 2020
np.savetxt(data_file_dir / "ensemble_time_series.csv", np.arange(1850,end_year+1).reshape(-1,1), fmt='%.16f', delimiter=",")
for ensemble_member in range(n_ensemble):
    output = np.loadtxt(data_file_dir / "ensemble_time_series.csv", delimiter=",").reshape(end_year-1850+1,-1)
    new_ensemble_values = np.zeros((end_year-start_year+1,1))
    for year in range(start_year,end_year+1):
        output_path = data_file_dir / f'cobe-sst3.m{ensemble_member+1:03d}.{year}.nc'
        if not Path(output_path).exists():
            url = f'https://climate.mri-jma.go.jp/pub/archives/Ishii-et-al_COBE-SST3/cobe-sst3/perturb/m{ensemble_member+1:03d}/cobe-sst3.m{ensemble_member+1:03d}.{year}.nc'
            try:
                # Send a GET request to the URL with stream enabled
                with requests.get(url, stream=True) as response:
                    response.raise_for_status() # Raise an exception for HTTP errors

                    # Open the output file in binary write mode
                    with open(output_path, 'wb') as file:
                        # Write data to file in chunks
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:# Filter out keep-alive chunks
                                file.write(chunk)
                print(f"File downloaded successfully: {output_path}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download file: {e}")
        data_file = netCDF4.Dataset(output_path)
        data = np.transpose(np.ma.getdata(data_file.variables['sst']).data)
        data_file.close()
        if delete_large_data_files:
            os.remove(output_path)
        data[data > 999] = np.nan
        new_ensemble_values[year-start_year,0] = np.mean(gridded_to_timeseries(data),axis=0) #ensemble provides daily data
    #COBE-SST3 uses 1890 perturbations for years prior to 1890
    new_ensemble_values = np.append(np.kron(np.ones((40,1)),new_ensemble_values[0,0]),new_ensemble_values).reshape(-1,1)
    np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((output,new_ensemble_values),axis=1), fmt='%.16f', delimiter=",")
    print(f"Completed ensemble {ensemble_member+1} out of 300")
