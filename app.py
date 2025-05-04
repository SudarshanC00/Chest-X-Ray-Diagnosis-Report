# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from inference import infer_image

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    results = infer_image(img)
    return jsonify(results)

if __name__ == "__main__":
    # in production, use gunicorn or similar
    app.run(host="0.0.0.0", port=5000, debug=True)