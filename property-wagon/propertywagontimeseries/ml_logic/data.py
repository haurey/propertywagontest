import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import pathlib




def query_data():
    data_folder = pathlib.Path().absolute().parent.joinpath('property-wagon','propertywagontimeseries','raw_data')
    data_path = data_folder.joinpath('resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv').absolute()
    data_path2 = data_folder.joinpath('resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv').absolute()
    data_path3 = data_folder.joinpath('resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv').absolute()
    data_path4 = data_folder.joinpath('resale-flat-prices-based-on-approval-date-2000-feb-2012.csv').absolute()


    df = pd.read_csv(data_path)
    df2 = pd.read_csv(data_path2)
    df3 = pd.read_csv(data_path3)
    df4 = pd.read_csv(data_path4)
    #ALL DATA
    df4 = df4.sort_values(by=['month'],ascending=True)
    df3 = df3.sort_values(by=['month'],ascending=True)
    df2 = df2.sort_values(by=['month'],ascending=True)
    df = df.sort_values(by=['month'],ascending=False)
    df = pd.concat([df,df2],ignore_index=True,sort=False)
    df = pd.concat([df,df3],ignore_index=True,sort=False)
    df = pd.concat([df,df4],ignore_index=True,sort=False)
    df = df.sort_values(by=['month'],ascending=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['town']=='WOODLANDS') & (df['flat_type']=='4 ROOM')]
    df = df.groupby(by='month').mean()
    df.drop(columns=['floor_area_sqm','lease_commence_date'],inplace=True)
    df = df.reset_index()
    df = df.rename(columns={'month': 'ds', 'resale_price':'y'})
    return df
