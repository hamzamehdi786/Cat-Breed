from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and class names
model = load_model('cat_breed_classifier.h5')
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize same as training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return "Cat Breed Classifier API is running! Use POST /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded file to temp folder
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(preds[0]))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    # Run Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)
