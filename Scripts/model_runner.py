# ---------------------------- IMPORTiNG DEPENDENCIES ---------------------------- #

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import torch

from . import constants
from . import autocorrector

# ---------------------------------- MAIN CODE ----------------------------------- #


def get_model_path(model_id):
    return constants.MODEL_PATHS.get(model_id, False)


def check_audio_rate(audio_rate):
    return audio_rate == constants.MODEL_AUDIO_RATE


# ----------------------------- BINARY MODELS RUNNER ----------------------------- #


# Function to load bin models using transformers
def _load_bin_model(model_path):
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        if constants.DEBUGGING:
            print("Model Error : ", e)
        model = False

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    except Exception as e:
        if constants.DEBUGGING:
            print("Processor Error", e)
        processor = False

    return model, processor


def _predict_bin_model(model, processor, audio_array, audio_rate):
    # pad input values and return pt tensor
    input_values = processor(audio_array, sampling_rate=audio_rate, return_tensors="pt").input_values

    # INFERENCE
    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)

    # transcribe
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    # autocorrect prediction
    autocorrected_transcription = autocorrector.autocorrect(transcription)

    return {"transcribed_text": transcription, "suggestion": autocorrected_transcription, "error": None}




# --------------------------- QUANTIZED MODELS RUNNER --------------------------- #



def _predict_quant_model(model, processor, audio_array, audio_rate):
    # pad input values and return pt tensor
    input_values = processor(audio_array, sampling_rate=audio_rate, return_tensors="pt").input_values

    # getting transcription
    transcription = model(input_values)

    # autocorrect prediction
    autocorrected_transcription = autocorrector.autocorrect(transcription)

    return {"transcribed_text": transcription, "suggestion": autocorrected_transcription, "error": None}



# Fucntion to load quantized models using pytorch
def _load_quant_model(model_path):
    try:
        model = torch.jit.load(model_path)
    except Exception as e:
        if constants.DEBUGGING:
            print("Model Error : ", e)
        model = False

    try:
        processor_path = '/'.join(model_path.split('/')[:-1])
        processor = Wav2Vec2Processor.from_pretrained(processor_path)
    except Exception as e:
        if constants.DEBUGGING:
            print("Processor Error", e)
        processor = False

    return model, processor



# --------------------------- MAIN MODELS RUNNER --------------------------- #



def predict(audio_array, audio_rate, model_id):

    model_path = get_model_path(model_id)

    if not model_path:
        return {"transcribed_text": None, "Error : ": f"Invalid model id = {model_id}"}

    if not check_audio_rate(audio_rate):
        return {"transcribed_text": None, "Error : ": f"Invalid audio rate while loading the cache audio = {audio_rate}"}

    
    model_id_seperated = model_id.split('-')

    if model_id_seperated[0] == constants.QUANTIZED_MODEL_HEADER:
        try:
            model, processor = _load_quant_model(model_path)
            return _predict_quant_model(model, processor, audio_array, audio_rate)
        except Exception as e:
            print(e)
            return {"transcribed_text": None, "Error : ": f"Problem loading model or processor or both or problem running model inference (check print output of the server for more info)", "complete_error": e}

    
    elif model_id_seperated[0] == constants.BINARY_MODEL_HEADER:
        try:
            model, processor = _load_bin_model(model_path)
            return _predict_bin_model(model, processor, audio_array, audio_rate)
        except Exception as e:
            return {"transcribed_text": None, "Error : ": f"Problem loading model or processor or both or problem running model inference (check print output of the server for more info)", "complete_error": e}
