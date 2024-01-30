from flask import Flask, render_template, request
import torch
from models.common import DetectMultiBackend
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model
model = DetectMultiBackend('weights/best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Process file
        f = request.files['file']
        # Save the file to ./static/uploads
        filepath = './static/uploads/' + f.filename
        f.save(filepath)

        # Read the image with OpenCV
        image = cv2.imread(filepath)

        # Ensure image is read correctly
        if image is None:
            return "Error: File is not a valid image."

        # Optional: Resize or preprocess the image as needed for YOLOv5
        # ...

        # Run YOLOv5 model
        results = model(image)

        # Process results
        # (Add code to process and display results)

        return 'Object detection completed'
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
