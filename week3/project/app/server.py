import json
import math
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"
global_data = {}

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """

    global_data['file_log_handler'] = open(LOGS_OUTPUT_PATH, 'w+')
    global_data['cls'] = NewsCategoryClassifier(verbose=True)
    global_data['cls'].load(MODEL_PATH)

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """

    if global_data['file_log_handler'] is not None:
        global_data['file_log_handler'].close()
        global_data['file_log_handler'] = None
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """

    t0 = time.monotonic_ns()
    X = {
        'source': request.source,
        'url': request.url,
        'title': request.title,
        'description': request.description
    }
    probs = global_data['cls'].predict_proba(X)
    label = global_data['cls'].predict_label(X)
    response = PredictResponse(scores=probs, label=label)
    t1 = time.monotonic_ns()

    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'request': request.dict(),
        'latency(ms)': (t1 - t0) / 1000000
    }
    global_data['file_log_handler'].write(json.dumps(log) + '\n')
    global_data['file_log_handler'].flush()

    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
