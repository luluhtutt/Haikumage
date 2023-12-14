import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from haikumage import *

app = Flask(__name__)

kosmos_model, kosmos_processor, model, tokenizer = load_models()

# Configure upload folder and allowed extensions
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['DATA_FOLDER']):
    os.makedirs(app.config['DATA_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        file.save(filepath)

        # Call your notebook function to generate text from the image
        generated_text = process_image(filepath)

        return render_template('index.html', filename=filename, generated_text=generated_text)

    return redirect(request.url)

@app.route('/data/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

def process_image(filepath):
    # Replace this with your actual code to process the image and generate text
    # This is just a placeholder
    word = img2word(filepath, kosmos_model, kosmos_processor)
    return word2haiku(word, model, tokenizer)

if __name__ == '__main__':
    app.run(debug=True)