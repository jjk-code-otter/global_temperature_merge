# Global temperature merge

Code for merging estimates of annual global mean temperature into a single ensemble dataset of global mean temperature. Also produces ensembles of radiative forcing and ocean heat content.



Python modules required:
numpy scipy sklearn statsmodels torch openpyxl pandas pathlib matplotlib seaborn requests xarray netCDF4



Download instructions:

1. Download Land\_and\_Ocean\_summary.txt from https://berkeleyearth.org/data/ and put it into the Berkeley Earth folder.
2. Download https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global\_TAVG\_ensemble.txt and put it into the Berkeley Earth Hires folder.
3. Copy-paste China-MST3.0-Imax data http://www.gwpu.net/en/h-nd-166.html from into an excel file. Save the excel file in the CMST3 folder and name it China-MST3.0-Imax.xlsx.
4. The translator cma-gmst.py automatically downloads and processes CMA-GMST data.
5. Download the netcdf file from https://zenodo.org/records/19630991 and save it in the Kadow folder. Contact Chris Kadow (kadow@dkrz.de) for access to the climatereconstructionAI ensemble and then save it in the Kadow\_ensemble folder. This is similar to the non-ensemble version available from https://zenodo.org/records/11262704.
6. Download annual\_gm\_cobe-stemp3 from https://climate.mri-jma.go.jp/pub/archives/Ishii-et-al\_COBE-SST3/gm/ and put it in the COBE-STEMP3 folder.
7. Download the DCENT\_I annual time series ensemble member netcdf and annual time series text file from the https://dcent-i.github.io/ and put them in the DCENT-I folder.
8. Download DCENT\_MLE\_v1.2\_timeseries\_annual\_anomalies\_ensemble.nc from https://www.wdc-climate.de/ui/entry?acronym=DCENT\_MLE\_v1\_2 and put it in the DCENT\_MLE folder.
9. Download C3S\_Bulletin\_temp\_202507\_Fig2b\_timeseries\_anomalies\_ref1850-1900\_global\_allmonths\_data.csv from https://sites.ecmwf.int/data/c3sci/bulletin/202507/temperature/ and put it in the ERA5 folder.
Alternatively, you can download the ERA5 2m temperature monthly data from the Copernicus Climate Data Store (https://cds.climate.copernicus/eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download) as a netcdf file and save it in the ERA5 folder as ERA5.nc. Later, run the translator era5\_alternative.py instead of era5.py.
10. Also, email Adrian Simmons (adrian.Simmons@ecmwf.int) for the ERA5 ensemble timeseries and put it in the ERA5\_ensemble folder.
Alternatively, you can download the ERA5 2m temperature monthly ensemble data from the Copernicus Climate Data Store (https://cds.climate.copernicus/eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download) as a netcdf file and save it in the ERA5\_ensemble folder as ERA5\_ensemble.nc. Later, run the translator era5\_ensemble\_alternative.py instead of era5\_ensemble.py.
11. Download the global mean timeseries GLB.Ts+dSST.csv from https://data.giss.nasa.gov/gistemp/ and put it in the GISTEMPv4 folder.
12. Download the analysis.ensemble\_series.global.annual.nc and analysis.component\_series.global.annual.nc files from https://www.metoffice.gov.uk/hadobs/glosatref and put them in the GloSAT folder.
13. Download HadCRU\_MLE\_v1.4\_timeseries\_annual\_anomalies\_ensemble.nc from https://doi.org/10.26050/WDCC/HadCRU\_MLE\_v1.4 and put it in the HadCRU\_MLE folder.
14. Download the analysis.ensemble\_series.global.annual.nc and analysis.component\_series.global.annual.nc files from https://www.metoffice.gov.uk/hadobs/hadcrut5 and put them in the HadCRUT5 folder.
15. Contact the JMA (jra@met.kishou.go.jp) for the JRA-3Q\_tmp2m\_global\_ts\_125\_Clim9120.txt file and put it in the JRA-3Q folder.
Alternatively, run the translator jra3q\_alternative.py instead of jra3q.py to automatically download and process JRA-3Q data from the UCAR website.
16. Go to https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5.1/access/, download annual average NOAAGlobalTempv5.1 land-ocean anomalies time series from 90°S to 90°N as aravg.ann.land\_ocean.90S.90N.v5.1.0.asc data, and then put it in the NOAAGlobalTempv5.1 folder.
17. Go to https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/, download annual average NOAAGlobalTempv6 land-ocean anomalies time series from 90°S to 90°N as aravg.ann.land\_ocean.90S.90N.v6.0.0.asc data, and then put it in the NOAAGlobalTempv6 folder.
18. The translator noaa\_global\_tempv5.0\_ensemble.py automatically downloads and processes the NOAA GlobalTempv5.0 ensemble.
19. Either download and run the forcing time series available at https://github.com/ClimateIndicator/forcing-timeseries to get ERF\_DAMIP\_1000.nc and ERF\_DAMIP\_1000\_full.nc, or contact Chris Smith (chris.smith@vub.be) for these files. Put these files in the Radiative Forcing folder. This data is similar to non-ensemble data available from https://zenodo.org/records/15630666.



Set up

\------



The managed data will be stored in a directory specified by the environment variable DATADIR.



In Windows, do the following:

\* right-click on the Windows icon (bottom left of the screen usually) and select "System"

\* Click on "Advanced system settings"

\* Click on "Environment Variables..."

\* Under "User variables for Username" click "New..."

\* For "Variable name" type DATADIR

\* For "Variable value" type the pathname for the directory you want the data to go in or navigate to the approriate directory using browse.

\* Make sure that the directory exists.



In MacOS or Linux, input the following commands into terminal, modified to accomodate your preferred path:

nano \~/.zshrc

export DATADIR=/path/to/data

source \~/.zshrc



Running code instructions:

\--------------------------

1. Input datasets are converted into a common format using code in the `translators` directory. You can use `run\_translators.py` to run the code. By default, the code does not automatically run cobe-sst3.py, which uses COBE-SST3 to create pseudo-ensembles as part of a sensitivity test. Please feel free to run cobe-sst3.py, but it will take substantially longer than the rest of the code. Alternatively, email Bruce Calvert (brucetcalvert@gmail.com) for the output excel file.
2. Generate pseudo-ensembles using `pseudo\_ensembles.py`.
3. Run `make\_meta\_ensemble.py` to merge datasets based on dataset family trees, which are defined in json files such as `FamilyTrees/hierarchy\_ur.json`. Family trees are organised into experiments such as `Experiments/basic.json`.
4. Plots are generated by `make\_meta\_ensemble.py`. Additional plots can be generated by running `compare\_all\_ensembles.py` and `plot\_ensembles.py`.
5. Run `frechet\_distances.py` to perform Fréchet distance tests.
6. Run `radiative\_forcing\_ensemble.py` to generate radiative forcing ensemble.



Output data is saved in the `Output` directory and plots are saved in the `Figures` directory, with subdirectories in each one for each "experiment".

