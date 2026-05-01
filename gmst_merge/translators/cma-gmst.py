from pathlib import Path
import netCDF4
import numpy as np
import os
import zipfile
from CMDCapi import CMDCClient
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'CMA-GMST'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'CMA-GMST'

client = CMDCClient(user_id = "I36qmCaRBufW3y3Il7djpHirUXcdJpR9Fz40DmiawNP%2BeQYQvYHRNw%3D%3D", output_dir = data_file_dir)  # This is Bruce's user ID
mean = np.zeros((0,1))
current_year = 1850
current_month = 1;
while True:
    if current_month != 1:
        break
    file_location = data_file_dir / ("SURF_CLI_GLB_MST_MON_GRID_2DEG_"+str(current_year)+"01.nc")
    if not file_location.is_file():
        params = {
            'isZip': '1',
            'day': '1',
            'month': '1,2,3,4,5,6,7,8,9,10,11,12',
            'year': current_year,
            'productId': '16',
            'source': '1',
        }
        try:
            file_name = client.retrieve(params)
            zipfile.ZipFile(file_name,"r").extractall(path=data_file_dir)
            os.remove(file_name)
            
            for current_month in range(1,13):
                date_string = str(current_year)+str(int(current_month//10))+str(int(current_month%10))
                file_name = data_file_dir / ("SURF_CLI_GLB_MST_MON_GRID_2DEG_"+date_string+".nc")
                if not file_name.is_file():
                    break
                data_file = netCDF4.Dataset(file_name)
                month_data = np.transpose(np.ma.getdata(data_file.variables['anomaly']).data)
                month_data = np.concatenate((month_data[:,0,:],np.kron(month_data,np.ones((1,2,1))),month_data[:,-1,:]),'axis=1') # convert to 180 latitudinal bands with equal thickness
                month_data = gridded_to_timeseries(month_data.reshape(month_data.shape[0],month_data.shape[1],1)).reshape(-1,1)
                mean = np.append(mean,month_data,axis=0)
        except:
            break
mean = monthly_to_annual_timeseries(mean,1850)
years = np.arange(1850,1850+mean.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
