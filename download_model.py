from basicsr.utils.download_util import load_file_from_url
import os

# Create a folder for model weights
os.makedirs('weights', exist_ok=True)

# Download the Real-ESRGAN model file (x4 upscaler)
url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth'
load_file_from_url(url, model_dir='weights')

print("✅ Model downloaded successfully into 'weights' folder!")
