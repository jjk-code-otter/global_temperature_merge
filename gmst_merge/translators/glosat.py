from pathlib import Path
import numpy as np
import os


def convert_file():
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GloSAT'
    filename = data_file_dir / 'GloSATref.1.0.0.0.analysis.ensemble_series.global.annual.csv'

    with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
        with open(filename, 'r') as f:
            f.readline()
            for line in f:
                columns = line.split(',')

                if int(columns[0]) >= 1850:
                    year = [columns[0]]

                    # Convert the ensemble to a numpy array
                    ensemble = [float(x) for x in columns[3:]]
                    ensemble = np.array(ensemble)

                    coverage_unc = float(columns[2])  # Coverage uncertainty is one sigma according to the file

                    ensemble_std = np.std(ensemble, ddof=1)
                    ensemble_mean = np.mean(ensemble)

                    total_unc = np.sqrt(coverage_unc ** 2 + ensemble_std ** 2)

                    # Scale the ensemble deviations from the mean to include coverage uncertainty
                    ensemble = ensemble_mean + ((total_unc / ensemble_std) * (ensemble - ensemble_mean))
                    ensemble = ensemble.tolist()
                    ensemble = [f'{x:.4f}' for x in ensemble]

                    columns = year + ensemble

                    line = ','.join(columns)
                    line += '\n'
                    o.write(line)
