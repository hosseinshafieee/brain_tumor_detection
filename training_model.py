import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = "./training_dataset"

CATEGORIES = ["cancer", "healthy"]

IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num  = CATEGORIES.index(category) 
        for img in os.listdir(path):
            try: 
                image_array  = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

x = []
y = []
for features, labels in training_data: 
    x.append(features)
    y.append(labels)

X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


x = X / 255.0

model = Sequential() 

model.add(Conv2D(64, (3, 3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))


model.compile(optimizer=tf.keras.optimizers.Adam() ,loss=tf.keras.losses.BinaryCrossentropy(),  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()])
model.fit(x, y, epochs=10, validation_split=0.1)

model.save('cancer_detector_model')

