import os
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import requests.exceptions 
import traceback

# --- Import load_model ---
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not found. Please install it: pip install tensorflow")
    load_model = None

# --- Configuration ---
CONFIG = {
    "IMG_HEIGHT": 32, 
    "IMG_WIDTH": 32,
    "IMG_CHANNELS": 3,
    "CLASS_NAMES": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
}

# --- 1. LOAD THE MODEL (Global Scope) ---
model = None
if load_model:
    try:
        # Load the model file from the same directory
        model = load_model('cifar10_custom_acc8757.keras') 
        print("--- Model loaded successfully ---")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("load_model function not available. Model cannot be loaded.")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- [NEW] Main Page Route ---
@app.route("/")
def home():
    """Serves the main index.html page."""
    return render_template("index.html")

# --- Image Preprocessing Function ---
def preprocess_image(image_bytes):
    """
    Takes image bytes, opens with PIL, resizes, converts to RGB,
    normalizes, and expands dimensions for the model.
    """
    image = Image.open(BytesIO(image_bytes)) 
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((CONFIG["IMG_WIDTH"], CONFIG["IMG_HEIGHT"]))
    image_array = np.array(image)
    image_array = image_array / 255.0
    expanded_image = np.expand_dims(image_array, axis=0)
    return expanded_image.astype(np.float32)


# --- API Endpoint (File Upload) ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded or failed to load."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part in request"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)
        predictions_raw = model.predict(preprocessed_image)
        predictions = predictions_raw[0] 

        predicted_index = np.argmax(predictions)
        predicted_class = CONFIG["CLASS_NAMES"][predicted_index]
        confidence = float(predictions[predicted_index])

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": dict(zip(CONFIG["CLASS_NAMES"], [float(p) for p in predictions]))
        })

    except Exception as e:
        print("="*50)
        print(f"ERROR: An internal Flask error occurred:")
        traceback.print_exc()
        print("="*50)
        return jsonify({
            "error": "An internal server error occurred.",
            "detail": str(e)
        }), 500

# --- API Endpoint (URL from GET or POST) ---
@app.route("/predict_url", methods=["GET", "POST"])
def predict_url():
    if model is None:
        return jsonify({"error": "Model is not loaded or failed to load."}), 500
    
    image_url = None

    # --- THIS LOGIC IS NOW FIXED ---
    if request.method == "POST":
        # 1. Try to get data from a standard HTML form first.
        image_url = request.form.get('url')

        # 2. If no form data, *then* try to get data from a JSON payload.
        if not image_url:
            if request.is_json:
                try:
                    data = request.get_json()
                    image_url = data.get('url')
                except Exception:
                    pass # It's not JSON, which is fine.
            
    else: # request.method == "GET":
        # This part is for GET requests (e.g., .../predict_url?url=...)
        image_url = request.args.get('url')
    # --- END OF FIX ---

    if not image_url:
        return jsonify({"error": "No 'url' provided in form, JSON, or query parameter"}), 400

    try:
        # 1. Download the image
        download_response = requests.get(image_url, timeout=10)
        download_response.raise_for_status()
        image_bytes = download_response.content

        # 2. Preprocess the Image
        preprocessed_image = preprocess_image(image_bytes)

        # 3. *** RUN PREDICTION DIRECTLY ***
        predictions_raw = model.predict(preprocessed_image)
        predictions = predictions_raw[0] # Get the first (and only) prediction

        # 4. Post-process and Return
        predicted_index = np.argmax(predictions)
        predicted_class = CONFIG["CLASS_NAMES"][predicted_index]
        confidence = float(predictions[predicted_index])

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": dict(zip(CONFIG["CLASS_NAMES"], [float(p) for p in predictions]))
        })

    except requests.exceptions.RequestException as req_err:
        return jsonify({
            "error": "Failed to download image from URL.",
            "detail": str(req_err)
        }), 400

    except Exception as e:
        print("="*50)
        print(f"ERROR: An internal Flask error occurred:")
        traceback.print_exc()
        print("="*50)
        return jsonify({
            "error": "An internal server error occurred.",
            "detail": str(e)
        }), 500

# Gunicorn/Cloud Run AND for local testing.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

