import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.parser import parse
import pickle
import warnings
warnings.filterwarnings('ignore')

from params import *
print("testing imports", GCP_PROJECT)
from ml_logic.data import get_data_with_cache, load_data_to_bq, clean_data, preprocess_data



def preprocess():

    # load data from local file
    df = get_data_with_cache()

    # clean and preprocess data
    df = clean_data(df)

    # for testing purposes
    # df = df.sample(n=10000)

    # define X, y
    target = 'resale_price'
    X = df.drop(columns=target)
    y = df[target]


    # train/test split and preprocess X_train, X_test
    X_train_processed, X_test_processed, y_train, y_test = preprocess_data(X, y)

    # Load data on Big Query a dataframe containing using data.load_data_to_bq()
    load_data_to_bq(pd.DataFrame(X_train_processed),
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'X_train_processed',
                    truncate=True)

    load_data_to_bq(pd.DataFrame(X_test_processed),
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'X_test_processed',
                    truncate=True)

    load_data_to_bq(pd.DataFrame(y_train),
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'y_train',
                    truncate=True)

    load_data_to_bq(pd.DataFrame(y_test),
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'y_test',
                    truncate=True)

    print("----------------Preprocessing done----------------")

def train_evaluate(n_estimators=200):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # Retrieve processed data
    X_train_processed = get_data_with_cache(processed_data = True,
                                        table_name = 'X_train_processed')
    X_test_processed = get_data_with_cache(processed_data = True,
                                        table_name = 'X_test_processed')
    y_train = get_data_with_cache(processed_data = True,
                                        table_name = 'y_train')
    y_test = get_data_with_cache(processed_data = True,
                                        table_name = 'y_test')

    # Build and train model
    print("----------------Training model----------------")
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train_processed, y_train)

    # Save model
    with open('model_pkl', 'wb') as files:
        pickle.dump(model, files)

    # Evaluate mse
    y_rf_pred = model.predict(X_test_processed)
    mse = np.sqrt(mean_squared_error(y_test, y_rf_pred))

    # Evaluate r2
    r2 = model.score(X_test_processed, y_test)

    print(f"✅ Training done. MSE: {mse}, r2: {r2}")

    return mse, r2


if __name__ == '__main__':
    preprocess()
    train_evaluate()
