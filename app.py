from flask import Flask, render_template, request, jsonify
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore
import numpy as np
import os

from werkzeug.utils import secure_filename  # ✅ Safer filename handling

app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/Solution_challenge_kaggle.keras')

# Predefined chatbot responses
agricultural_responses = {
    "aphids": "To control aphids organically: 1. Use neem oil spray 2. Release ladybugs 3. Spray strong water jets to dislodge them 4. Plant companion plants like marigolds.",
    "fungal": "For fungal issues in humid conditions: 1. Improve air circulation 2. Reduce watering frequency 3. Apply organic fungicides 4. Remove affected leaves.",
    "general": "I can help you with plant diseases, pest control, and organic farming practices. Please ask specific questions!",
    "late blight": "Late blight is caused by Phytophthora infestans. Treat with copper-based fungicides, remove infected leaves, and avoid wetting the foliage.",
    "early blight": "Early blight is caused by Alternaria. Use fungicides like chlorothalonil, rotate crops, and remove plant debris.",
    "pesticide": "Avoid chemical overuse! Use neem oil, insecticidal soaps, or introduce beneficial insects like ladybugs."
}

@app.route('/')
def home():
    return render_template('index.html')  # Main UI with form and chatbot

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    img = request.files['file']

    # Use secure_filename to avoid file path issues
    safe_filename = secure_filename(img.filename)
    img_path = os.path.join("static", safe_filename)
    img.save(img_path)

    # Preprocess image
    img_data = image.load_img(img_path, target_size=(400, 400))
    img_array = image.img_to_array(img_data)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Classes and precautions
    class_names = ["Healthy", "Early Blight", "Late Blight"]
    precautions = [
        "Your potato plant is healthy! Maintain good watering and fertilization practices.",
        "Early Blight detected! Remove affected leaves, apply fungicide, and avoid overhead watering.",
        "Late Blight detected! Isolate plant, remove affected parts, and apply copper-based fungicide."
    ]

    if confidence < 0.3:
        return render_template('result.html',
                               predicted_class="Uncertain Prediction",
                               confidence=f"{confidence*100:.1f}",
                               advice="Please try uploading a clearer image.")

    predicted_label = class_names[predicted_class]
    advice = precautions[predicted_class]

    return render_template('result.html',
                           predicted_class=predicted_label,
                           confidence=f"{confidence*100:.1f}",
                           advice=advice)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').lower()

    # Match keywords to predefined responses
    response = None
    for keyword, reply in agricultural_responses.items():
        if keyword in user_msg:
            response = reply
            break

    # Fallback response
    if not response:
        response = "🌾 I'm not sure about that. Try asking about pests, diseases, or organic treatments!"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
