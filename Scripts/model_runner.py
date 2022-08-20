# ---------------------------- IMPORTiNG DEPENDENCIES ---------------------------- #

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from . import constants
import numpy as np

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

    return {"transcribed_text": transcription, "error": None}


def predict(audio_array, audio_rate, model_id):
    model_path = get_current_model(model_id)

    if not model_path:
        return {"transcribed_text": None, "error": f"Error: Invalid model id = {model_id}"}

    if not check_audio_rate(audio_rate):
        return {"transcribed_text": None, "error": f"Error: Invalid audio rate = {audio_rate}"}


    model, processor = load_model(model_path)

    if not model or not processor:
        return {"transcribed_text": None, "error": f"Error loading model or processor or both with code = M{str(model)[:15]} P{str(processor)[:15]}"}

    return _predict(model, processor, audio_array, audio_rate)
