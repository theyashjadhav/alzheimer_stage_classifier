from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import os
import numpy as np
from model import createModel
import matplotlib.pyplot as plt


def predict_single_image(image_path, SIZE):
    categories = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]
    

    model = createModel()
    
    data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_data = cv2.resize(data, (SIZE, SIZE))
    new_data = new_data / 255.0
    image = np.array(new_data).reshape(-1, SIZE, SIZE, 1)
    

    prediction = model.predict(image)
    

    plt.imshow(new_data, cmap="gray")
    ptitle = "Prediction: {0}".format(categories[np.argmax(prediction)])
    plt.title(ptitle)
    plt.show()
    print(ptitle)
    return ptitle

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected.")

    uploaded_file = request.files['file']
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    
    result = predict_single_image(image_path, 120)
    image_path =  os.path.join("C:\\Users\\Shubhangi Jadhav\\ML-Mini\\alzheimer-stage-classifier-master\\",image_path)
    image_path = "C:\\Users\\Shubhangi Jadhav\\ML-Mini\\alzheimer-stage-classifier-master\\uploads\\26 (71).jpg"
    # image_path = ".\\uploads\\26 (71).jpg"
    print(image_path)
    return render_template('index.html', result=result, image_path= image_path)

if __name__ == '__main__':
    app.run(debug=True)
