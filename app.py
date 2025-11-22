import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from model_utils import (
    extract_text_from_image,
    generate_summaries,
    generate_captions,
    enhance_resolution
)

# Load config
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', 'outputs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    return jsonify({'filepath': filepath})


@app.route('/summarize', methods=['POST'])
def summarize():
    image_path = request.json['image']
    extracted_text = extract_text_from_image(image_path)
    summaries = generate_summaries(extracted_text)
    return jsonify({'summaries': summaries})


@app.route('/caption', methods=['POST'])
def caption():
    image_path = request.json['image']
    captions = generate_captions(image_path)
    return jsonify({'captions': captions})


@app.route('/super_res', methods=['POST'])
def super_res():
    image = request.files['image']
    scale = int(request.form['scale'])
    filename = secure_filename(image.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'sr_{scale}x_{filename}')
    image.save(input_path)
    enhanced = enhance_resolution(input_path, output_path, scale)
    return send_file(enhanced, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True,port=8000)
