import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.utils import plot_model


img_height = 270
img_width = 480
batch_size = 3

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    label_mode = 'categorical',
    class_names = ['left', 'right', 'straight'],
    color_mode = 'grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    subset = 'training',
    verbose=True,
    seed = 3
)


ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    label_mode = 'categorical',
    class_names = ['left', 'right', 'straight'],
    color_mode = 'grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    subset = 'validation',
    verbose=True,
    seed=3
)


model = keras.Sequential ([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(12, activation='relu'),
    
    #layers.Dense(13 , activation='relu'),
    layers.Dense(3, activation='softmax')
])

print(model.summary())

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

batch_size = 10
epochs = 4

model.fit(ds_train, epochs=epochs, batch_size=batch_size, verbose=2)

model.evaluate(ds_validation, batch_size=batch_size)

if (input('do you want to save the model? (y/n):\n') == 'y'):
    model_name = input("What do you want to name it?\n")
    model.save(f"{model_name}.keras")


'''
class_names = {0: 'left', 1: 'right', 2: 'straight'}
for image_batch, label_batch in ds_validation:
    for i in range(image_batch.shape[0]):
        image = image_batch[i].numpy()
        label = label_batch[i]
        plt.imshow(image, cmap='gray')
        plt.show()

        image = cv2.resize(image, (480, 270))

        frame_array = np.array(image)

        frame_array = frame_array[np.newaxis, :, :, np.newaxis]

        frame_tensor = tf.convert_to_tensor(frame_array)

        prediction = (model.predict(frame_tensor))

        prediction = class_names[np.argmax(prediction)]
        
        print(f"prediction: {prediction}\nlabel: {label}")
        image = ''
'''


# Assuming you have a CNN model named 'model'
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
