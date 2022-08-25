# ---------------------------- IMPORTING DEPENDENCIES ---------------------------- #

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import Scripts.predictor as predictor


# ----------------------------- CONFIGURING FASTAPI ------------------------------- #


# initializing fastapi app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------- MAIN CODE ----------------------------------- #


# root endpoint for server status check
@app.get("/")
def read_root():
    return {"status": "Server is up and running"}



# Pydantic classes for processing incoming requests
class Filedata(BaseModel):
    filedata: str = Form(...)


# endpoint for audio upload and conversion to text using base64 data from frontend
@app.post("/predict-base64/")
def predict_base64_default(filedata: Filedata):
    return predictor.predict_base64(filedata.filedata)


# endpoint for audio upload and conversion to text using base64 data from frontend
@app.post("/predict-base64/{model_id}/")
def predict_base64(model_id: str, filedata: Filedata):
    return predictor.predict_base64(filedata.filedata, model_id)


# endpoint for audio upload and conversion to text using array data from frontend
@app.post("/predict-array/{model_id}/")
def predict_array(filedata: str = Form(...)):
    audio_array = eval(filedata)
    assert isinstance(audio_array, list)
    return {"status" : "please use base64 format to use this API"}


# endpoint for live transcription
@app.post("/live-transcribe/{model_id}/")
def predict_live(filedata: str = Form(...)):
    return {"status": "please use base64 format to use this API"}