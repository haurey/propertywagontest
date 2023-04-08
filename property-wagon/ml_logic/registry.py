import time
import pickle

from interface.params import *

def save_model(model) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = f'../../data/model_{timestamp}'
    with open(model_path, 'wb') as files:
        pickle.dump(model, files)

    print("✅ Model saved locally")

    # Save to GCS
    from google.cloud import storage
    model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    print('model_filename: ', model_filename)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None
