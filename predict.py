import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from model import createModel


def predict(SIZE):
    categories = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]

    path = "./test/"
    images = []
    for img in os.listdir(path):
        data = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_data = cv2.resize(data, (SIZE, SIZE))
        new_data = new_data / 255.0
        images.append(new_data)
    model = createModel()
    title = os.listdir('./test')
    x = 0
    for img ,indx,value in images,enumerate(title):
        image = np.array(img).reshape(-1, SIZE, SIZE, 1)
        prediction = model.predict(image)
        plt.imshow(img, cmap="gray")
        ptitle = "Prediction: {0}".format(categories[np.argmax(prediction)])

        plt.figtext(0, 0, title[indx])
        plt.title(ptitle)
        plt.show()
        print(prediction)


    print(len(images), len(title))


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
    print(prediction)
    
predict_single_image()
predict(120)
