import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# Load the TFLite model
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    label_mode = 'categorical',
    class_names = ['left', 'right', 'straight', 'turn'],
    color_mode = 'grayscale',
    batch_size=1,
    image_size=(270, 480),
    shuffle=True, 
    validation_split=0.1,
    subset = 'validation',
    verbose=True,
    seed=10
)
#normalization_layer = tf.keras.layers.Rescaling(1./255)
#ds_validation = ds_validation.map(lambda x, y: (normalization_layer(x), y))

correct_predictions = 0
wrong_predictions = 0

interpreter = tf.lite.Interpreter(model_path="100PercentLabelerQ.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#model = tf.keras.models.load_model('test9.keras')

class_names = {0: 'left', 1: 'right', 2:'straight'}
start_time = time.time()
for image_batch, label_batch in ds_validation:
    for i in range(image_batch.shape[0]):
        image = image_batch[i].numpy()
        label = label_batch[i]

        image = cv2.resize(image, (480, 270))
        
        frame_array = np.array(image)
        
        frame_array = (frame_array / 255)
        
        frame_array = frame_array[np.newaxis, :, :, np.newaxis]
        
        frame_tensor = tf.convert_to_tensor(frame_array)
        
        frame_tensor = tf.cast(frame_tensor, tf.float32)
        
        print(frame_tensor.shape)
            
        interpreter.set_tensor(input_details[0]['index'], frame_tensor)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        #prediction = model.predict(frame_array)
        
        prediction_a = class_names[np.argmax(prediction)]
        label_a = class_names[np.argmax(label)]
        print(prediction)
        print(prediction_a, label_a)
        if label_a == prediction_a:
            correct_predictions += 1
        elif label_a != prediction_a:
            wrong_predictions += 1
            
        #print(f'raw label: {label}\nraw_prediction: {prediction}\nprediction: {prediction_a}\nlabel: {label_a}\n\n')
        
        image = ''
end_time = time.time()
print(correct_predictions, wrong_predictions)

accuracy = (correct_predictions / (correct_predictions + wrong_predictions)) * 100

print(accuracy)
average_latency = (end_time - start_time) / 23
print(average_latency)