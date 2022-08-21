# ---------------------------- IMPORTING DEPENDENCIES ---------------------------- #

import base64
import librosa

from . import model_runner
from . import constants

# ------------------------------- base64 functions ------------------------------- #


def get_base64_chunk(data):
    if data.find("data:audio") == -1:
        return False

    base_64_index = data.find("base64,")

    if base_64_index == -1:
        return False

    base_64_index += len("base64,")

    return data[base_64_index:]



def predict_base64(base64_audio_data, model_id=constants.DEFAULT_MODEL_ID):

    base64_audio_data = get_base64_chunk(base64_audio_data)

    if not base64_audio_data:
        return {"transcribed_text": None, "error": "Error: data is either not audio or not base64"}

    # you can write below bytes data directly to a file using 'wb' mode
    decoded_audio_bytes = base64.decodebytes(base64_audio_data.encode())

    # sending this audio bytes to cahe file to be used by librosa later
    open(constants.CACHE_AUDIO_FILE_PATH, "wb").write(decoded_audio_bytes)

    # decoding thee audio array and audio rate using librosa
    audio_array, audio_rate = librosa.load(constants.CACHE_AUDIO_FILE_PATH, sr=constants.MODEL_AUDIO_RATE)

    return model_runner.predict(audio_array, audio_rate, model_id)
