from initializer import *
from transcriber import *

from nltk import word_tokenize, corpus

import secrets
import pyaudio
import wave
import json

import os

RATES = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORDING_TIME = 5
SPEECH_PATH = "temp"
CORPUS_LANG = "portuguese"
CONFIG_PATH = "config.json"

def capture_speech(recorder):
    record = recorder.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE, input=True, frames_per_buffer=RATES)

    print("Comece a falar!")

    speech = []
    for _ in range(0, int(SAMPLING_RATE / RATES * RECORDING_TIME)):
        speech.append(record.read(RATES))
    
    record.stop_stream()
    record.close()

    print("Fala capturada")

    return speech

def record_speech(speech):
    recorded, file = False, f"{SPEECH_PATH}/{secrets.token_hex(32).lower()}.wav"

    try:
        wav = wave.open(file, 'wb')
        wav.setframerate(SAMPLING_RATE)
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(recorder.get_sample_size(FORMAT))
        wav.writeframes(b''.join(speech))
        wav.close()
        
        recorded = True
    except Exception as e:
        print(f"Ocorreu um erro gravando o arquivo: {str(e)}")

    return recorded, file

def remove_stop_words(transcription, stop_words):
    command = []

    tokens = word_tokenize(transcription)
    for token in tokens:
        if token not in stop_words:
            command.append(token)

    return command

def validate_command(command, actions):
    valid, action, object = False, None, None

    if len(command) in [2,3]:
        action = command[0]
        if len(command) == 2:
            object = command[1]
        else:
            object = f"{command[1]} {command[2]}"
            

        for expected_action in actions:
            if action == expected_action["name"]:
                if object in expected_action["objects"]:
                    valid = True

                    break
    return valid, action, object

def start(device):
    recorder = pyaudio.PyAudio()

    started, processor, model, _ = start_model(MODELS[0], device)

    stop_words, actions = None, None

    if started:
        stop_words = corpus.stopwords.words(CORPUS_LANG)

        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
            actions = config["actions"]

    return started, processor, model, recorder, stop_words, actions

def start_cli():
    while True:
        speech = capture_speech(recorder)
        recorded, file = record_speech(speech)

        if recorded:
            transcription = transcribe(device, load_speech(file), model, processor)
            os.remove(file)

            print(f"Transcrição: {transcription}\n")

            command = remove_stop_words(transcription, stop_words)
            valid, action, object = validate_command(command, actions)

            if valid:
                execute_action(action, object)

def execute_action(action, object):
    if action == "iniciar" and object == "exploração":
        print("Exploração iniciada. Preparando sistemas de navegação e sensores.")
    elif action == "coletar" and object == "amostra":
        print("Coletando amostra. Ativando braços robóticos e compartimento de armazenamento.")
    elif action == "mapear" and object == "fundo oceânico":
        print("Mapeando fundo oceânico. Iniciando varredura com sonar e registro de dados.")
    elif action == "lançar" and object == "drone submarino":
        print("Lançando drone submarino. Estabelecendo conexão de controle remoto.")
    elif action == "retornar" and object == "superfície":
        print("Retornando à superfície. Despressurizando sistemas e iniciando subida.")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    started, processor, model, recorder, stop_words, actions = start(device)

    if started:
        start_cli()