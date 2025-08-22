Global temperature merge
========================

Code for merging estimates of annual global mean temperature into a single ensemble dataset of global mean temperature. 
Also produces ensembles of radiative forcing and ocean heat content.

Python modules required:
numpy scipy sklearn statsmodels torch openpyxl pandas pathlib matplotlib seaborn requests xarray netCDF4


Download instructions:
----------------------

Create a `data` directory and inside that create a `ManagedData` directory. All directories mentioned here in the 
download instructions should be subfolders of `ManagedData`

1. Download `Land_and_Ocean_summary.txt` from https://berkeleyearth.org/data/ and put it into the `Berkeley Earth` folder.

2. Download https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global_TAVG_ensemble.txt and put it in
   the `Berkeley Earth Hires` folder

4. Copy-paste China-MST3.0-Imax data from http://www.gwpu.net/en/h-nd-166.html from into an excel file and delete all the
   except year, Global and GMST uncertainty. Save the excel file in csv format in the `CMST3` folder and name it `CMST_3.0.csv`.

6. Contact Chris Kadow (kadow@dkrz.de) for access to the climatereconstructionAI ensemble and then save it in the 
   `Kadow_ensemble` folder. This is similar to the non-ensemble version available from https://zenodo.org/records/11262704.
   You want the files named `20crtaspadzens_tas_mon-gl-72x36_hadcrut5_observation_ens-3_1850-2022_image_XXX.nc` where XXX is
   the ensemble member number.

8. Download annual_gm_cobe-stemp3 from https://climate.mri-jma.go.jp/pub/archives/Ishii-et-al_COBE-SST3/gm/ and put it 
   in the `COBE-STEMP3` folder.

9. Download DCENT_MLE_v1.1_timeseries_annual_anomalies_ensemble.nc from 
   https://www.wdc-climate.de/ui/entry?acronym=DCENT_MLE_v1_1 and put it in the `DCENT_MLE_v1p1` folder.

10. Download `C3S_Bulletin_temp_202507_Fig2b_timeseries_anomalies_ref1850-1900_global_allmonths_data.csv` (or whichever is the
    latest equivalent) from https://sites.ecmwf.int/data/c3sci/bulletin/202507/temperature/ and put it in the `ERA5` folder.

11. Get the ERA ensemble from Adrian Simmons and put it in the `ERA5 ensemble` directory

12. Download the global mean timeseries `GLB.Ts+dSST.csv` from https://data.giss.nasa.gov/gistemp/ and put it in the 
   `GISTEMP` folder.

13. Download GloSATref-1-0-0-0_analysis_ensemble-series_global_annual.nc from 
   https://data.ceda.ac.uk/badc/deposited2025/GloSAT/GloSATref-1-0-0-0/analysis/diagnostics/ensemble-series or
   https://www.metoffice.gov.uk/hadobs/glosatref/download.html and put it in the `GloSAT` folder.

15. Download `HadCRU_MLE_v1.3_timeseries_annual_anomalies_ensemble.nc` from https://doi.org/10.26050/WDCC/HadCRU_MLE_v1.3 
   and put it in the `HadCRU_MLE` folder.

16. Download `HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.nc` and 
    `HadCRUT.5.0.2.0.analysis.component_series.global.annual.nc` from  
    https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/download.html and put it in the HadCRUT5 folder.

17. Go to https://rda.ucar.edu/datasets/d640002/dataaccess/#, click "Web File Listing" for anl_surf125, then "Faceted 
    Browse", then select "2m temperature (mean)", and then download all netcdf files since 1981 and save them in the 
    `JRA-3Q` folder. Or just email the JRA-3Q team to get the monthly global means.

18. Go to https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5.1/access/, download annual average 
    `NOAAGlobalTempv5.1` land-ocean anomalies time series from 90°S to 90°N as aravg.ann.land_ocean.90S.90N.v5.1.0.asc 
    data, and then put it in the `NOAA v5.1` folder.

19. Go to https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6/access/timeseries/, download annual 
    average NOAAGlobalTempv6.0 land-ocean anomalies time series from 90°S to 90°N as 
    aravg.ann.land_ocean.90S.90N.v6.0.0.asc data, and then put it in the `NOAA v6` folder.

20. Go to https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/tmp/2019.ngt.par.ensemble/, download and extract all 1000 
    ensemble members that start with temp.ano.merg5.dat, and then put them in the `NOAA_ensemble` folder. 
    Alternatively, the code automatically downloads and extracts these files for you if you have an internet
    connection.

Bonus:

21. Contact Chris Smith (chris.smith@vub.be) for access to ensemble radiative forcing data files of ERF_DAMIP_1000.nc 
    and ERF_DAMIP_1000_full.nc, and put it in the Radiative Forcing folder. This data is similar to non-ensemble data 
    available from https://zenodo.org/records/15630666.

Set up
------

The managed data will be stored in a directory specified by the environment variable DATADIR. This is easy to set in linux. In windows, do the following:

* right-click on the Windows icon (bottom left of the screen usually) and select "System"
* Click on "Advanced system settings"
* Click on "Environment Variables..."
* Under "User variables for Username" click "New..."
* For "Variable name" type DATADIR
* For "Variable value" type the pathname for the directory you want the data to go in or navigate to the approriate directory using browse.
* Make sure that the directory exists.


Running code instructions:
--------------------------

1. Input datasets are converted into a common format using code in the `translators` directory. You can use 
   `run_translators.py` to run the code. The first time you run it set `run_long_conversions` to `True` to
   make sure you convert the larger datasets. The default is `False` which only runs the quick conversions. 

2. Generate pseudo-ensembles using `pseudo_ensembles.py`.

3. Run `make_meta_ensemble.py` to merge datasets based on dataset family trees, which are defined in json files such 
   as `FamilyTrees/hierarchy_ur.json`. Family trees are organised into experiments such as `Experiments/basic.json`.

4. Plots are generated by `make_meta_ensemble.py`. Additional summary plots can be generated by running 
   `compare_all_ensembles.py` and `plot_ensembles.py`.

5. Run `frechet_distances.py` to perform Fréchet distance tests.

6. Run `radiative_forcing_ensemble.py` to generate radiative forcing ensemble.


Data goes into the `Output` directory and plots go into the `Figures` directory, with subdirectories in each one for 
each "experiment".
