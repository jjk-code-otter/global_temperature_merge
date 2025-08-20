# Code written by Bruce T. T. Calvert - August 2025
# Obtains a 100 representative member subset of a 1000 member radiative forcing timeseries ensemble using balanced k-means algorithm

import numpy as np
from pathlib import Path
import netCDF4
import pandas as pd
from openpyxl import Workbook
from useful_functions import balanced_kmeans

if __name__ == '__main__':
    # Set Seed of Random Number Generator
    rng = np.random.default_rng(seed=0)
    variable_names = [
        'total', 'co2', 'ch4', 'n2o', 'halogen', 'o3', 'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
        'contrails', 'land_use', 'bc_snow', 'h2o_strat', 'solar', 'volcanic'
    ]
    wb = Workbook()
    (wb.active).title = 'total'
    wb.save(
        Path(__file__).resolve().parent / r'Data\Radiative Forcing\representative_ERF_ensemble_1750_to_present.xlsx')
    writer = pd.ExcelWriter(
        Path(__file__).resolve().parent / r'Data\Radiative Forcing\representative_ERF_ensemble_1750_to_present.xlsx',
        engine='openpyxl', mode='a', if_sheet_exists='replace')

    for i in range(len(variable_names)):

        if i == 0:
            data_file = netCDF4.Dataset(Path(__file__).resolve().parent / r'Data\Radiative Forcing\ERF_DAMIP_1000.nc')
        else:
            data_file = netCDF4.Dataset(
                Path(__file__).resolve().parent / r'Data\Radiative Forcing\ERF_DAMIP_1000_full.nc')

        data = np.ma.getdata(data_file.variables[variable_names[i]]).data

        if i == 0:
            clusters = balanced_kmeans(data, 100, rng)
            selected_members = np.array([])
            for j in range(100):
                selected_members = np.append(selected_members, np.arange(1000)[clusters == j][
                    np.random.permutation(np.arange(sum(clusters == j)))[0]])

        output_ensemble = np.ma.getdata(data_file.variables['time']).data.reshape(-1, 1)

        # Obtain one ensemble from each cluster
        for j in range(100):
            output_ensemble = np.append(output_ensemble, data[:, int(selected_members[j])].reshape((-1, 1)), axis=1)

        with pd.ExcelWriter(
                Path(__file__).resolve().parent / r'Output\representative_ERF_ensemble_1750_to_present.xlsx',
                engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df = pd.DataFrame(output_ensemble, columns=['time'] + ['ensemble' + str(j) for j in range(1, 101)])
            df.to_excel(writer, sheet_name=str(variable_names[i]), index=False)
