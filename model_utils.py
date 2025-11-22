import os
import torch
import numpy as np
from PIL import Image
import easyocr
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    BlipProcessor, BlipForConditionalGeneration
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ---- Load lightweight models once ----
print("🔹 Initializing models (this may take a moment)...")

reader = easyocr.Reader(['en'])
summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("✅ Models loaded successfully.")


# ---- OCR ----
def extract_text_from_image(image_path):
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)


# ---- Summarization ----
def generate_summaries(text):
    inputs = summ_tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    outputs = summ_model.generate(inputs['input_ids'], num_return_sequences=2, num_beams=4)
    return [summ_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


# ---- Caption generation ----
def generate_captions(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = caption_processor(raw_image, return_tensors="pt")
    outputs = caption_model.generate(**inputs, num_return_sequences=3, num_beams=5)
    return [caption_processor.decode(o, skip_special_tokens=True) for o in outputs]


# ---- Super-resolution with Real-ESRGAN ----
def enhance_resolution(input_path, output_path, scale):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("weights", exist_ok=True)

    # Define model architecture and expected weight file name
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)
    model_path = f"weights/RealESRGAN_x{scale}plus.pth"

    # Auto-download if weights are missing
    if not os.path.exists(model_path):
        print(f"⚙️  Downloading Real-ESRGAN x{scale} model weights...")
        import requests
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x{scale}plus.pth"
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Weights downloaded and saved at:", model_path)
        else:
            raise FileNotFoundError(f"Failed to download model from {url}")

    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    # Run enhancement
    image = np.array(Image.open(input_path).convert("RGB"))
    output, _ = upsampler.enhance(image, outscale=scale)
    Image.fromarray(output).save(output_path)
    print(f"✅ Super-resolution complete: {output_path}")
    return output_path
