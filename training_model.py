import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Conv1D

DATADIR = "./training_dataset"

CATEGORIES = ["cancer", "healthy"]

IMG_SIZE = 128
training_data = []

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def flip_image(image, type):
    result = cv2.flip(image, type)
    return result

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num  = CATEGORIES.index(category) 
        for img in os.listdir(path):
            try: 
                image_array  = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                angle = 90
                for i in range(1, 5): 
                    rotated_image = rotate_image(new_array, angle * i)
                    training_data.append([rotated_image, class_num])
                    for j in range(-1, 2): 
                        print(j)
                        training_data.append([flip_image(rotated_image, j), class_num])
                    
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

# random.shuffle(training_data)

x = []
y = []
for features, labels in training_data: 
    x.append(features)
    y.append(labels)

X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


x = X / 255.0

model = Sequential() 
print(x.shape[1:])
model.add(Conv2D(32, (3, 3),  input_shape = x.shape[1:]))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam() ,loss=tf.keras.losses.BinaryCrossentropy(),  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()])
history = model.fit(x, y, epochs=5, validation_split=0.25)
from ann_visualizer.visualize import ann_viz

ann_viz(model, view=True, filename="sdfg", title="sdfgs")

def create_testing_data():
    DATADIR = "./../../datasets/archive/Training"
    testing_data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num  = CATEGORIES.index(category) 
        for img in os.listdir(path):
            try: 
                image_array  = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
                # plt.imshow(new_array)
                # plt.show()
            except Exception as e:
                pass
    x = []
    y = []
    for features, labels in testing_data: 
        x.append(features)
        y.append(labels)

    X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    x = X / 255.0
    return [x, y]
    

result = create_testing_data()

tests = model.evaluate(result[0],  result[1], verbose=2)

print('Loss= ', tests[0], 'Acc= ', tests[1])

plt.plot(history.history['binary_accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_binary_accuracy'], label = 'Validation Accuracy')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1.5])
plt.legend(loc='lower right')

plt.show()

model.save('cancer_detector_model')

