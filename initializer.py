from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODELS = [ "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2", "facebook/wav2vec2-base-960h", "Edresson/wav2vec2-large-xlsr-coraa-portuguese" ]

def start_model(model_name, device="cpu"):
    errors = []
    started = False
    processor = None
    model = None

    print(f"Starting model: {model_name}")

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        started = True
    except Exception as e:
        errors.append(e)
        print(f"Error while starting model {model_name}: {str(e)}")
        
    return started, processor, model, errors

def start_models(models=MODELS, device="cpu"):
    all_started = True
    errors = []

    for model_name in models:
        started, processor, model, model_errors = start_model(model_name, device)
        all_started = all_started and started
        errors.extend(model_errors)

    return all_started, processor, model, errors
