import tensorflow as tf
import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.utils import plot_model 


img_height = 270
img_width = 480
batch_size = 1



ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './data/',
    label_mode = 'categorical',
    class_names = ['Stop', 'Go'],
    color_mode = 'rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    subset = 'training',
    verbose=True,
    seed = 10
)


ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    './data/',
    label_mode = 'categorical',
    class_names = ['Stop', 'Go'],
    color_mode = 'rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    subset = 'validation',
    verbose=True,
    seed=10
)


normalization_layer = tf.keras.layers.Rescaling(1./255)
ds_validation = ds_validation.map(lambda x, y: (normalization_layer(x), y))
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))


model = keras.Sequential ([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

print(model.summary())


class TestCallback(callbacks.Callback):
    '''
    Outputs the metrics of the test sample at the end of each epoch.
    '''
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs | {'verbose': 0} # Verbose is disable
    
    def on_epoch_end(self, epoch, logs={}):
        loss, accuracy = model.evaluate(*self.args, **self.kwargs)
        print(f'Test accuracy: {accuracy} test loss: {loss}')
        
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

epochs = 3
model.fit(ds_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[TestCallback(ds_validation)])
print('\n<---------------------------- Evaluation ---------------------------->')
model.evaluate(ds_validation, batch_size=batch_size)


model_name = ''
if (input('\n\ndo you want to save the model? (y/n):\n') == 'y'):
    model_name = input("What do you want to name it?\n")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(f'{model_name}Q.tflite', 'wb') as f:     
        f.write(tflite_quant_model)
    model.save(f"{model_name}.keras")
    
class_names = {0: 'Stop', 1: 'Go'}


for image_batch, label_batch in ds_validation:
    for i in range(image_batch.shape[0]):
        image = image_batch[i].numpy()
        label = label_batch[i]
        #plt.imshow(image, cmap='gray')
        #plt.show()

        image = cv2.resize(image, (480, 270))

        frame_array = np.array(image)

        frame_array = frame_array[np.newaxis, :, :, :]

        frame_tensor = tf.convert_to_tensor(frame_array)

        prediction = (model.predict(frame_tensor))

        prediction = class_names[np.argmax(prediction)]
        
        label = class_names[np.argmax(label)]
        
        print(f"prediction: {prediction}\nlabel: {label}")
        image = ''
 
