from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time

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

app = FastAPI()

class timed(object):
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.t = (time.perf_counter() - self.t) * 1000

@app.on_event("startup")
def startup_event():
    """
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    logger.add(LOGS_OUTPUT_PATH)
    logger.info("Setup completed")
    global clf
    clf = NewsCategoryClassifier()
    clf.load(MODEL_PATH)


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    logger.info("Shutting down application")
    logger.remove()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
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
    with timed() as t:
        pred_label = clf.predict_label(request.dict())
        pred_probs = clf.predict_proba(request.dict())
    latency = t.t
    response = PredictResponse(scores=pred_probs, label=pred_label)
    date_time_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    logger.info({
        'timestamp': date_time_str,
        'request': request.dict(),
        'prediction': response.dict(),
        'latency (ms)': latency
    })
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
