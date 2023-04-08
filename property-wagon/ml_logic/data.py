import pandas as pd
from pathlib import Path
from google.cloud import bigquery

from interface.params import *


def get_data_with_cache(processed_data: bool = False,
                        table_name:str = '',
                        gcp_project:str = '',
                        query:str = '') -> pd.DataFrame:

    if processed_data:
        read_file = Path(LOCAL_DATA_PATH).joinpath(f'{table_name}.csv')
        if read_file.is_file():
            print("----------------Load processed data from CSV----------------")
            df = pd.read_csv(read_file)

        else:
            print("----------------Load processed data from Querying Big Query server----------------")

            client = bigquery.Client(project=gcp_project)
            query_job = client.query(query)
            result = query_job.result()
            df = result.to_dataframe()

            # Store as CSV if BQ query returned at least one valid line
            if df.shape[0] > 1:
                save_path = Path(LOCAL_DATA_PATH).joinpath(f'{table_name}.csv')
                df.to_csv(save_path, index=False)

    else:
        print("----------------Load raw data from CSV----------------")
        # Concat resale csv files vertically
        resale_csv_file = Path(LOCAL_DATA_PATH).joinpath('resale-flat-prices-from-2020-2023.csv')
        resale_df = pd.read_csv(resale_csv_file)
        resale_df['month'] = pd.to_datetime(resale_df['month']) # to datetime
        resale_df.sort_values(by='month', ascending=False, inplace=True, ignore_index=True)

        # Concat econ data csv files vertically
        econ_csv_file = Path(LOCAL_DATA_PATH).joinpath('econ_data.csv')
        econ_df = pd.read_csv(econ_csv_file)
        econ_df['month'] = pd.to_datetime(econ_df['month']) # to datetime
        econ_df.sort_values(by='month', ascending=False, inplace=True, ignore_index=True)

        # Concat resale and econ data into a single df
        df = resale_df.merge(econ_df, on='month', how='left')

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    drop_flattype = ['1 ROOM', 'MULTI-GENERATION']
    df = df[df['flat_type'].isin(drop_flattype) == False]

    drop_flatmodel = ['3Gen', '2-room', 'Premium Maisonette',
                  'Improved-Maisonette','Terrace',
                  'Premium Apartment Loft','Type S2',
                  'Type S1','Adjoined flat','Model A-Maisonette',
                  'Multi Generation', 'DBSS', ]
    df = df[df['flat_model'].isin(drop_flatmodel) == False]

    df.dropna(inplace=True)
    df.drop(columns=['month', 'block', 'street_name'], inplace=True)

    return df

def preprocess_data(X: pd.DataFrame, y: np.array) -> np.ndarray:
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.compose import make_column_transformer
    from sklearn.model_selection import train_test_split

    ohe = OneHotEncoder(sparse=False)  #town, flat_model
    oe = OrdinalEncoder() #flat_type, storey_range
    scaler = StandardScaler() #floor_area, lease_commence_date, cpi

    columntransformer = make_column_transformer(
    (ohe, ['town', 'flat_model']),
    (oe, ['flat_type', 'storey_range']),
    (scaler, ['floor_area_sqm', 'lease_commence_date', 'cpi'])
    )

    # train/test split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    # transform X
    print("----------------Preprocessing data----------------")
    X_train_processed = columntransformer.fit_transform(X_train)
    X_test_processed = columntransformer.transform(X_test)

    # save data in csv locally
    pd.DataFrame(X_train_processed).to_csv(Path(LOCAL_DATA_PATH).joinpath(
        'X_train_processed.csv'), index=False)
    print(f"✅ X_train_processed saved locally, with shape {X_train_processed.shape}")

    pd.DataFrame(X_test_processed).to_csv(Path(LOCAL_DATA_PATH).joinpath(
        'X_test_processed.csv'), index=False)
    print(f"✅ X_test_processed saved locally, with shape {X_test_processed.shape}")

    pd.DataFrame(y_train).to_csv(Path(LOCAL_DATA_PATH).joinpath(
        'y_train.csv'), index=False)
    print(f"✅ y_train saved locally, with shape {y_train.shape}")

    pd.DataFrame(y_test).to_csv(Path(LOCAL_DATA_PATH).joinpath(
        'y_test.csv'), index=False)
    print(f"✅ y_test saved locally, with shape {y_test.shape}")

    return X_train_processed, X_test_processed, y_train, y_test


def load_data_to_bq(data: pd.DataFrame,
              gcp_project:str,
              bq_dataset:str,
              table: str,
              truncate: bool) -> None:
    """
    - Save dataframe to bigquery
    - Empty the table beforehands if `truncate` is True, append otherwise.
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(f"----------------Save data to bigquery {full_table_name}----------------")

    # Load data to full_table_name
    # 🎯 Hint for "*** TypeError: expected bytes, int found":
    # BQ can only accept "str" columns starting with a letter or underscore column

    # TODO: simplify this solution if possible, but student may very well choose another way to do it.
    # We don't test directly against their own BQ table, but only the result of their query.
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_"
                                                        else str(column) for column in data.columns]

    client = bigquery.Client()

    # define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
