import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import pathlib
from propertywagontimeseries.ml_logic.data import query_data,clean_data
from propertywagontimeseries.ml_logic.model import initialize_model
from sklearn.model_selection import train_test_split
# loading library
import pickle
# create an iterator object with write permission - model.pkl


df = query_data()

df = clean_data(df)

model = initialize_model()

model.fit(df,freq='M')


    
future = model.make_future_dataframe(df, periods=60)
forecast = model.predict(future, decompose=False, raw=True)

print(forecast)

with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)