from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import EMNISTapp 
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origins='*')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction = EMNISTapp.make_predictions(filepath)

    return jsonify({"prediction": prediction})

    

if __name__ == "__main__":
    app.run(debug=True)