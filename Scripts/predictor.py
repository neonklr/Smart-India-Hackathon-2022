# ---------------------------- IMPORTING DEPENDENCIES ---------------------------- #

import base64
import io
import soundfile as sf

from . import model_runner

# ------------------------------- base64 functions ------------------------------- #


def get_base64_chunk(data):
    if data.find("data:audio") == -1:
        return False

    base_64_index = data.find("base64,")

    if base_64_index == -1:
        return False

    base_64_index += len("base64,")

    return data[base_64_index:]




def predict_base64(base64_audio_data, model_id):

    base64_audio_data = get_base64_chunk(base64_audio_data)

    if not base64_audio_data:
        return {"transcribed_text": None, "error": "Error: data is either not audio or not base64"}

    # you can write below bytes data directly to a file using 'wb' mode
    decoded_audio_bytes = base64.decodebytes(base64_audio_data.encode('utf-8'))

    # making bytes data file-like using io to be used by soundfile module
    audio_array, audio_rate = sf.read(io.BytesIO(decoded_audio_bytes))
    audio_array = audio_array.flatten()

    return model_runner.predict(audio_array, audio_rate, model_id)
