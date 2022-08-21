# ---------------------------- IMPORTiNG DEPENDENCIES ---------------------------- #

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

from . import constants
from . import autocorrector

# ---------------------------------- MAIN CODE ----------------------------------- #

def get_current_model(model_id):
    return constants.MODEL_PATHS.get(model_id, False)


# TODO : if audio rate doesn't match base model audio rate, fix the array
def check_audio_rate(audio_rate):
    return audio_rate == constants.MODEL_AUDIO_RATE


def load_model(model_path):

    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        print("Model Error : ", e)
        model = False

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    except Exception as e:
        print("Processor Error", e)
        processor = False

    return model, processor


def _predict(model, processor, audio_array, audio_rate):
    # pad input values and return pt tensor
    input_values = processor(audio_array, sampling_rate=audio_rate, return_tensors="pt").input_values

    # INFERENCE
    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)

    # transcribe
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    # autocorrect prediction
    transcription = autocorrector.autocorrect(transcription)

    return {"transcribed_text": transcription, "error": None}


def predict(audio_array, audio_rate, model_id):
    model_path = get_current_model(model_id)

    if not model_path:
        return {"transcribed_text": None, "Error : ": f"Invalid model id = {model_id}"}

    if not check_audio_rate(audio_rate):
        return {"transcribed_text": None, "Error : ": f"Invalid audio rate while loading the cache audio = {audio_rate}"}


    model, processor = load_model(model_path)

    if not model or not processor:
        return {"transcribed_text": None, "Error : ": f"Problem loading model or processor or both (check print output of the server for more info)"}

    return _predict(model, processor, audio_array, audio_rate)
