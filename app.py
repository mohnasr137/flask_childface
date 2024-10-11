import os
import cv2  # Using OpenCV to load images like in the DL model
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'my_model.h5')
model = None

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img_path):
    # Load image using OpenCV (BGR format)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    # Resize image to (224, 224) to match the model input
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    # Expand dimensions to match the input shape of the model (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    if model is None:
        return "Model not loaded"
    prediction = model.predict(img_array)
    return 0 if prediction[0][0] > 0.5 else 1

@app.route('/childFace', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img_file.save(img_path)

    try:
        img_array = preprocess_image(img_path)
        prediction = predict_image(img_array)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
