import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime as dt
import time
import math
import re
import cartopy.feature as cf
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import matplotlib.patches as mpatches
import streamlit as st
from IPython.display import set_matplotlib_formats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import requests
from io import StringIO
import streamlit as st
import time

st.set_page_config(layout="wide")

st.write("# Wind Anomaly Forecast Tool")
st.subheader("Methdology: Ridge Regression")

# Extracts online data 

def ENSO_ONI_Obtain(NAO_index):
    url = "https://psl.noaa.gov/data/correlation/oni.data"
    response = requests.get(url)
    data = response.text

    length = len(data)
    data = data[0:length - 312]

    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[1::]]
    
    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 1:].apply(pd.to_numeric)

    # Drop any rows with missing values
    #df.dropna(inplace=True)

    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    ENSO_ONI_index = numpy_array.flatten()
    ENSO_ONI_index = ENSO_ONI_index[~np.isnan(ENSO_ONI_index)]

    if len(ENSO_ONI_index) < len(NAO_index):
        ENSO_ONI_index = np.append(ENSO_ONI_index, np.nan)

    return ENSO_ONI_index

def SOI_Obtain():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
    response = requests.get(url)
    data = response.text

    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[4:80]]
    data_lines = [re.findall(r'-?\d+\.\d+|-999.9', line) for line in lines[4:80]]
    
    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 0::].apply(pd.to_numeric, errors = 'coerce')
    
    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    SOI_index = numpy_array.flatten()

    mask = SOI_index >= -500
    SOI_index = SOI_index[mask]

    num_nan_add = 12
    nan_array = np.full(num_nan_add, np.nan)

    SOI_index = np.concatenate((nan_array, SOI_index))

    return SOI_index

def NAO_Obtain():
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
    response = requests.get(url)
    data = response.text

    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[1::]]

    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 1:].apply(pd.to_numeric)

    # Drop any rows with missing values
    #df.dropna(inplace=True)

    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    NAO_index = numpy_array.flatten()
    NAO_index = NAO_index[~np.isnan(NAO_index)]

    return NAO_index

def PDO_Obtain():
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    response = requests.get(url)
    data = response.text
    
    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[98::]]
    
    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 1:].apply(pd.to_numeric)

    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    PDO_index = numpy_array.flatten()

    mask = PDO_index <= 90
    PDO_index = PDO_index[mask]
    
    return PDO_index

def Very_Dry_Wet_Obtain():
    url = "https://www.ncei.noaa.gov/access/monitoring/uspa/wet-dry/0/data.csv"

    df = pd.read_csv(url, skiprows = 1)
    df.columns = df.columns.str.strip()
    df = df[660::]
    
    percent_very_dry = df['Very Dry'].str.replace('%', '').astype(float)
    percent_very_wet = df['Very Wet'].str.replace('%', '').astype(float)

    Very_Dry = percent_very_dry.to_numpy()
    Very_Wet = percent_very_wet.to_numpy()

    return Very_Dry, Very_Wet

def QBO_Obtain():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index"
    response = requests.get(url)
    data = response.text

    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[4:54]]
    data_lines = [re.findall(r'-?\d+\.\d+|-999.9', line) for line in lines[4:54]]
    
    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 1::].apply(pd.to_numeric, errors = 'coerce')

    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    QBO_index = numpy_array.flatten()
    QBO_index = QBO_index[~np.isnan(QBO_index)]

    mask = QBO_index >= -900
    QBO_index = QBO_index[mask]

    num_nan_add = 393
    nan_array = np.full(num_nan_add, np.nan)

    QBO_index = np.concatenate((nan_array, QBO_index))
    
    return QBO_index

def AMO_Obtain():
    url = "https://tropical.colostate.edu/Forecast/downloadable/csu_amo.csv"
    
    df = pd.read_csv(url)
    
    df = df.iloc[:, 1::].apply(pd.to_numeric, errors = 'coerce')

    AMO_index = df.to_numpy().flatten()

    mask = ~np.isnan(AMO_index)
    AMO_index = AMO_index[mask]

    return AMO_index 

def MJO_extract():
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii"
    response = requests.get(url)
    data = response.text

    # Remove the first line (INDEX headers) to make parsing easier
    data_text = data.strip().split('\n')[1:]

    # Convert the remaining text data to a pandas DataFrame
    MJO = pd.read_csv(StringIO('\n'.join(data_text)), delim_whitespace=True)

    # Reset the index to the first column (assumed to be the date)
    MJO.reset_index(inplace=True)
    MJO.rename(columns={'index': 'Date'}, inplace=True)
    MJO = MJO.drop('Date', axis = 1)
    MJO = MJO.rename(columns = {'PENTAD': 'Date'})
    MJO['Date'] = pd.to_datetime(MJO['Date'], format = '%Y%m%d')
    MJO.set_index('Date', inplace = True)
    MJO = MJO.apply(pd.to_numeric, errors = 'coerce')
    MJO = MJO.resample('M').mean()
    MJO = MJO.dropna()
    MJO = MJO[:-1]

    MJO_20 = MJO['20E'].to_numpy()
    MJO_70 = MJO['70E'].to_numpy()
    MJO_80 = MJO['80E'].to_numpy()
    MJO_100 = MJO['100E'].to_numpy()
    MJO_120 = MJO['120E'].to_numpy()
    MJO_140 = MJO['140E'].to_numpy()
    MJO_160 = MJO['160E'].to_numpy()
    MJO_120W = MJO['120W'].to_numpy()
    MJO_40W = MJO['40W'].to_numpy()
    MJO_10W = MJO['10W'].to_numpy()
    
    num_nan_add = 336
    nan_array = np.full(num_nan_add, np.nan)

    MJO_20 = np.concatenate((nan_array, MJO_20))
    MJO_70 = np.concatenate((nan_array, MJO_70))
    MJO_80 = np.concatenate((nan_array, MJO_80))
    MJO_100 = np.concatenate((nan_array, MJO_100))
    MJO_120 = np.concatenate((nan_array, MJO_120))
    MJO_140 = np.concatenate((nan_array, MJO_140))
    MJO_160 = np.concatenate((nan_array, MJO_160))
    MJO_120W = np.concatenate((nan_array, MJO_120W))
    MJO_40W = np.concatenate((nan_array, MJO_40W))
    MJO_10W = np.concatenate((nan_array, MJO_10W))

    return MJO, MJO_20, MJO_70, MJO_80, MJO_100, MJO_120, MJO_140, MJO_160, MJO_120W, MJO_40W, MJO_10W

def PNA_Obtain():
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table"
    response = requests.get(url)
    data = response.text

    lines = data.strip().split('\n')

    # Extract the data lines (ignoring header, footer, and empty lines)
    data_lines = [line.split() for line in lines[1::]]

    # Convert the data lines into a pandas DataFrame
    df = pd.DataFrame(data_lines)
    
    # Select only the numerical columns and drop the non-numeric columns
    df = df.iloc[:, 1:].apply(pd.to_numeric)
    
    # Drop any rows with missing values
    #df.dropna(inplace=True)

    # Convert the DataFrame to a NumPy array
    numpy_array = df.values
    
    PNA_index = numpy_array.flatten()
    PNA_index = PNA_index[~np.isnan(PNA_index)]

    return PNA_index

NAO_index = NAO_Obtain()
PNA_index = PNA_Obtain()
PDO_index = PDO_Obtain()
ENSO_ONI_index = ENSO_ONI_Obtain(NAO_index)
SOI_index = SOI_Obtain()
Dry_index, Wet_index = Very_Dry_Wet_Obtain()
QBO_index = QBO_Obtain()
AMO_index = AMO_Obtain()
MJO, MJO_20, MJO_70, MJO_80, MJO_100, MJO_120, MJO_140, MJO_160, MJO_120W, MJO_40W, MJO_10W = MJO_extract()

# Calculates normalized wind anomalies for lower 48

def wind_anom_calculate():

    wind_100m = xr.open_dataset('/Users/rshroff/SESCO Code/SESCO_PROJECT_3/100m_wind.nc')
    wind_100m = wind_100m.reduce(np.nansum, dim='expver',keep_attrs=True)
    
    wind_100m['ws'] = np.sqrt(wind_100m["u100"] ** 2 + wind_100m["v100"] ** 2)
    
    wind_100m = wind_100m.drop_vars('u100')
    wind_100m = wind_100m.drop_vars('v100')

    wind_100m_mean = wind_100m.groupby('time.month').mean(dim='time')
    wind_100m_std = wind_100m.groupby('time.month').std(dim='time')
    
    wind_100m_anom = (wind_100m.groupby('time.month') - wind_100m_mean)
    wind_100m_anom = wind_100m_anom.groupby('time.month') / wind_100m_std

    return wind_100m_anom

wind_100m_anom = wind_anom_calculate()

# Performs regression analysis

def reg_calc(df):
    
    leadtime = 1
    df['target'] = df.shift(-leadtime)['City_Wind_Anom'] # setting lead time for model (-8 = lead time of 5)
    #df = df.iloc[:-leadtime,:].copy()

    from sklearn.linear_model import Ridge # importing regression model

    reg = Ridge(alpha=0.005)

    predictors = ['Prev_Wind_Anom', 'NAO', 'PDO', 'PDO','PNA', 'AMO', 'SOI', 'ENSO_ONI', 'Season', 'MJO_20', 
              'MJO_70', 'MJO_80', 'MJO_100', 'MJO_120', 'MJO_140', 'MJO_160', 'MJO_120W', 'MJO_40W', 'MJO_10W',
              ]
    
    train = df.loc['1950-01-01':'2018-12-01'] # setting training period
    test = df.loc['2019-01-01':] # setting testing period
    reg.fit(train[predictors], train['target']) # training model
    predictions = reg.predict(test[predictors]) # using model to predict ENSO for years 2001-2020

    combined = pd.concat([test['target'], pd.Series(predictions, index = test.index)], axis = 1)
    combined.columns = ['actual', 'predictions'] # combining actual vs predicted ENSO values
    error = abs(combined['predictions']).mean()
    forval = combined['predictions'][53]

    return forval, error

# Creates pandas dataframe with predictors, wind anom, and target

def df_create(wind_100m_anom, lat, lon):

    date_range = pd.date_range(start = '1950-01', end = '2023-7', freq='M')
    df = pd.DataFrame(index = date_range)
    df.index = df.index.strftime('%Y-%m')

    latitude = lat
    longitude = lon

    City_wind_anom = wind_100m_anom['ws'].sel(latitude = latitude, longitude = longitude, method = 'nearest')

    df['City_Wind_Anom'] = City_wind_anom

    def Prev_Wind_Anom(City_wind_anom):
        
        City_wind_anom = City_wind_anom[:-1]
        prev_wind_anom = np.insert(City_wind_anom, 0, np.nan)
        
        df['Prev_Wind_Anom'] = prev_wind_anom
        
        return prev_wind_anom

    Prev_Wind_Anom = Prev_Wind_Anom(City_wind_anom)

    def add_variable_columns():
       
        df['NAO'] = NAO_index
        df['PNA'] = PNA_index
        df['ENSO_ONI'] = ENSO_ONI_index
        df['PDO'] = PDO_index
        df['SOI'] = SOI_index
        df['%Dry'] = Dry_index
        df['%Wet'] = Wet_index
        df['AMO'] = AMO_index
        df['QBO'] = QBO_index
        df['MJO_20'] = MJO_20
        df['MJO_70'] = MJO_70
        df['MJO_80'] = MJO_80
        df['MJO_100'] = MJO_100
        df['MJO_120'] = MJO_120
        df['MJO_140'] = MJO_140
        df['MJO_160'] = MJO_160
        df['MJO_120W'] = MJO_120W
        df['MJO_40W'] = MJO_40W
        df['MJO_10W'] = MJO_10W
    
    add_variable_columns()

    df.fillna(0, inplace = True)

    df.index = pd.to_datetime(df.index)

    # Define a function to map months to seasons
    def get_season(month):
        if 3 <= month <= 5:  # Spring: March (3) to May (5)
            return 1
        elif 6 <= month <= 8:  # Summer: June (6) to August (8)
            return 2
        elif 9 <= month <= 11:  # Fall/Autumn: September (9) to November (11)
            return 3
        else:  # Winter: December (12), January (1), and February (2)
            return 0

    # Extract the month from the index and map to the corresponding season
    df['Season'] = df.index.month.map(get_season)

    return df

with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

@st.cache_data
def run_analysis():

    lats = np.arange(25,50,.5)
    lons = np.arange(-110,-75,.5)

    lat_list = []
    lon_list = []
    forval_list = []
    error_list = []

    for a in range(len(lats)):
        for b in range(len(lons)):

            
            lat = lats[a]
            
            lon = lons[b]

            df = df_create(wind_100m_anom, lat, lon)
            
            forval, error = reg_calc(df)

            lat_list.append(lat)
            lon_list.append(lon)
            forval_list.append(forval)
            error_list.append(error)

    return forval_list, error_list, lat_list, lon_list

forval_list, error_list, lat_list, lon_list = run_analysis()

data = {
'latitude': lat_list,
'longitude': lon_list,
'temperature': forval_list,
'error': error_list
}

wind_forecast_data = pd.DataFrame(data)

# Convert pandas DataFrame to xarray dataset

ds = wind_forecast_data.set_index(['latitude', 'longitude']).to_xarray()

# Creates figure showing wind anomaly forecast


def create_fig(ds):

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.contourf(ds['longitude'], ds['latitude'], ds['temperature'], transform=ccrs.PlateCarree(),levels=np.linspace(-3, 3, 20), cmap = 'BrBG')
    plt.colorbar(pad = 0)
    #ds['error'].plot()


    ax.coastlines(resolution='50m')
    ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor=[.1, .1, .1], linewidth=1)
    ax.add_feature(cf.STATES.with_scale('50m'), edgecolor=[.1, .1, .1], linewidth=.5)

    ax.set_title('July 100 m Wind Anomaly Forecast (m/s)', fontsize = 12, fontweight = 'bold', loc = 'left')

    return fig

fig = create_fig(ds)

st.pyplot(fig)