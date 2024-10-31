from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

app = Flask(__name__)

# Function to load and classify an image
def classify_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    
    # Preprocess the image for the model
    input_image = preprocess_image(image)
    
    # Make a prediction
    size, color = model.predict(input_image)
    
    # Create folders if they don't exist
    create_folders(['Large', 'Medium', 'Small'])
    
    # Move the image to the appropriate folder
    if size == 'Large':
        folder = 'Large'
    elif size == 'Medium':
        folder = 'Medium'
    else:
        folder = 'Small'
    
    # Save the classified image to the folder
    save_image(image_path, folder, image)

    return size, color

# Preprocess the image for the model
def preprocess_image(image):
    # Dummy preprocessing for illustration
    # Resize, normalize, etc.
    resized_image = cv2.resize(image, (224, 224))  # Example size
    input_image = np.expand_dims(resized_image, axis=0)
    return input_image

# Create folders if they don't exist
def create_folders(folder_names):
    for folder in folder_names:
        if not os.path.exists(folder):
            os.makedirs(folder)

# Save the image to the appropriate folder
def save_image(image_path, folder, image):
    filename = os.path.basename(image_path)
    save_path = os.path.join(folder, filename)
    cv2.imwrite(save_path, image)

# Dummy model for illustration
class DummyModel:
    def predict(self, image):
        # Return dummy size and color
        return 'Medium', 'Brown'

@app.route('/')
def upload_image():
    return render_template('upload.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        model = DummyModel()
        size, color = classify_image(file_path, model)
        
        return render_template('result.html', size=size, color=color, image_path=file.filename)

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
