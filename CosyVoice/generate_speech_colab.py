from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import os

# Google Drive paths
SENTENCE_FILE = "/content/drive/MyDrive/IASNLP Project/data/natural/sentences.txt"
OUTPUT_DIR = "/content/drive/MyDrive/IASNLP Project/data/CosyVoice/happy"

cosyvoice = AutoModel(
    model_dir="pretrained_models/CosyVoice-300M-Instruct"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SENTENCE_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} sentences")

for idx, sentence in enumerate(sentences, start=1):

    try:
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{idx:04d}.wav"
        )

        for result in cosyvoice.inference_instruct(
            sentence,
            "英文女",
            "Speak in a happy tone.<|endofprompt|>" #Change the prompt according to the desired emotion
        ):

            torchaudio.save(
                output_path,
                result["tts_speech"],
                cosyvoice.sample_rate
            )

        print(f"[{idx}/{len(sentences)}] Saved {output_path}")

    except Exception as e:
        print(f"Failed sentence {idx}")
        print(e)

print("Done!")