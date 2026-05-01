from pathlib import Path
import netCDF4
import numpy as np
import sys
import requests
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries
from useful_functions import days_in_month

data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'JRA-3Q'

mean = np.zeros((0,1))
current_year = 1981
current_month = 1
while True:
    date_string = str(current_year)+str(int(current_month//10))+str(int(current_month%10))
    file_name = "jra3q-ms-mn.anl_surf125.0_0_0.tmp2m-hgt-an-ll125-mn."+date_string+"0100_"+date_string+str(days_in_month(current_year)[current_month-1]).lstrip('[').rstrip(']')+"18.nc"
    file_location = data_file_dir / file_name;
    if not file_location.is_file():
        url = "https://mghpcc-cache.nationalresearchplatform.org:8443/ncar/gdex/d640002/anl_surf/"+date_string+"/"+file_name
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(file_location, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
        except requests.exceptions.RequestException as e:
                print(f"Failed to download file: {e}")
                break
        break
    data_file = netCDF4.Dataset(file_name)
    month_of_data = np.transpose(np.ma.getdata(data_file.variables['tmp2m-hgt-an-ll125-mn']).data)
    month_of_data = np.kron(month_of_data,np.ones((1,2,1)))[:,1:-2,:] # convert to 288 latitudinal bands with equal thickness
    month_of_data = gridded_to_timeseries(month_of_data.reshape(month_of_data.shape[0],month_of_data.shape[1],1)).reshape(-1,1)
    mean = np.append(mean,month_of_data,axis=0)
    current_month += 1
    if current_month == 13:
        current_month = 1
        current_year += 1
mean = monthly_to_annual_timeseries(mean,1981)
years = np.arange(1981,1981+mean.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
