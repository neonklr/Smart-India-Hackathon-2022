# ---------------------------- IMPORTING DEPENDENCIES ---------------------------- #

import base64
import librosa
from ffmpy import FFmpeg
import os

from . import model_runner
from . import constants

# ------------------------------- base functions ------------------------------- #


def get_base64_chunk(data):
    if data.find("data:audio") == -1:
        return False

    base_64_index = data.find("base64,")

    if base_64_index == -1:
        return False

    base_64_index += len("base64,")

    return data[base_64_index:]


def remove_cache_file():
    os.remove(constants.CACHE_AUDIO_FILE_PATH + '.wav')
    os.remove(constants.CACHE_AUDIO_FILE_PATH + '.webm')


def get_audio_format(base64_audio_data):
    for format in constants.AUDIO_FORMATS:
        if base64_audio_data.find(format) != -1:
            return constants.AUDIO_FORMATS[format]


def _generate_wav_file(base64_data):
    # sending this audio bytes to cahe file to be used by librosa later
    with open(constants.CACHE_AUDIO_FILE_PATH + '.wav', "wb") as f:
        f.write(base64.decodebytes(base64_data.encode()))


def _generate_webm_file(base64_data):
    # sending this audio bytes to cahe file to be used by librosa later
    with open(constants.CACHE_AUDIO_FILE_PATH + ".webm", "wb") as f:
        f.write(base64.decodebytes(base64_data.encode()))

    # ff = FFmpeg(
    #     executable=r"C:\Users\gupta\Desktop\ffmpeg-2022-08-22-git-f23e3ce858-full_build\bin\ffmpeg.exe",
    #     inputs={constants.CACHE_AUDIO_FILE_PATH + '.webm': None},
    #     outputs={constants.CACHE_AUDIO_FILE_PATH: '-c:a pcm_f32le'},    # -c
    # )

    ff = FFmpeg(
        executable=constants.FFMPEG_PATH,
        inputs={constants.CACHE_AUDIO_FILE_PATH + '.webm': None},
        outputs={constants.CACHE_AUDIO_FILE_PATH +
                 '.wav': '-c:a pcm_f32le'},    # -c
    )

    ff.cmd
    ff.run()


def predict_base64(base64_audio_data, model_id=constants.DEFAULT_MODEL_ID):

    # get audio format
    audio_format = get_audio_format(base64_audio_data)

    if not audio_format:
        return {"transcribed_text": None, "error": "Unsupported audio format"}

    # get base64 data
    base64_audio_data = get_base64_chunk(base64_audio_data)

    if not base64_audio_data:
        return {"transcribed_text": None, "error": "data is either not audio or not base64"}

    if audio_format == 'webm':
        _generate_webm_file(base64_audio_data)

    elif audio_format == 'wav':
        _generate_wav_file(base64_audio_data)

    # decoding thee audio array and audio rate using librosa
    audio_array, audio_rate = librosa.load(
        constants.CACHE_AUDIO_FILE_PATH + '.wav', sr=constants.MODEL_AUDIO_RATE)

    # Removing cache files
    remove_cache_file()

    return model_runner.predict(audio_array, audio_rate, model_id)


if constants.DEBUGGING:
    from memory_profiler import profile
    predict_base64 = profile(predict_base64)
