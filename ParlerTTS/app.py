import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
)


def generate_audio(prompt, description):

    input_ids = tokenizer(
        description,
        return_tensors="pt"
    ).input_ids.to(device)

    prompt_input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )

    audio_arr = generation.cpu().numpy().squeeze()

    return audio_arr, model.config.sampling_rate