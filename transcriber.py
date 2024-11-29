from initializer import *
import torchaudio
import torch

AUDIOS = ["audios/testando_fixed.wav", "audios/ola.wav"]

SAMPLING_RATE = 16000

def load_speech(location):
    audio, sample_rate = torchaudio.load(location)
    if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
    if sample_rate != SAMPLING_RATE:
            sampling_adapter = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
            audio = sampling_adapter(audio) 
        
    return audio.squeeze()

def transcribe(device, speech, model, processor):
    input_values = processor(speech, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_values.to(device)
    logits = model(input_values).logits

    prediction = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(prediction)[0]

    return transcription.lower()
        

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started, processor, model, errors = start_models(MODELS, device)

    if started:
        for audio in AUDIOS:
            speech = load_speech(audio)
            transcription = transcribe(device, speech, model, processor)
            print(f"Transcription: {transcription}")
    else:
        print("Failed to initialize models. Errors:", errors)