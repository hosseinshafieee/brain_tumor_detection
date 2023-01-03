import cv2
import tensorflow as tf

CATEGORIES = ["cancer", "healthy"]
 
def prepare(filepaht):
    IMG_SIZE = 100
    image_array = cv2.imread(filepaht, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

model = tf.keras.models.load_model('cancer_detector_model')

test_image_path = [
    './random_test_images/cancer1.jpeg',
    './random_test_images/cancer2.webp',
    './random_test_images/healthy1.jpeg',
    './random_test_images/healthy2.jpeg',
    './random_test_images/healthy3.webp',
]
prediction= model.predict([prepare(test_image_path[2])])

print(CATEGORIES[int(prediction[0][0])])