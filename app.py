from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure static folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run detection
            results = model(filepath)

            # Save result image
            results[0].save(filename=filepath)

            return render_template("index.html", image=filename)

    return render_template("index.html", image=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)