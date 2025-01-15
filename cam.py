import picamera2
import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpiozero import AngularServo, DistanceSensor
import os
from adafruit_servokit import ServoKit


interpreter = tf.lite.Interpreter(model_path="Big3Q.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Traffic_interpreter = tf.lite.Interpreter(model_path="99PercentTrafficBNQ.tflite")
#Traffic_interpreter.allocate_tensors()
#Traffic_input_details = Traffic_interpreter.get_input_details()
#Traffic_output_details = Traffic_interpreter.get_output_details()

camera = picamera2.Picamera2()
camera.start()

left_level = [45, 30, 10]
right_level = [100, 140, 160]

kit = ServoKit(channels=8)
Distance = DistanceSensor(echo=23, trigger=24)
def left():
    kit.servo[0].angle = right_level[0]
    kit.servo[1].angle = left_level[1]

def right():
    kit.servo[0].angle = right_level[1]
    kit.servo[1].angle = left_level[0]

def straight():
    kit.servo[0].angle = right_level[2]
    kit.servo[1].angle = left_level[1]

def right_turn():
    kit.servo[0].angle = right_level[2]
    kit.servo[1].angle = left_level[0]

def left_turn():
    kit.servo[0].angle = right_level[0]
    kit.servo[1].angle = left_level[2]


function_mapping = {'left': left, 'right': right, 'straight': straight, 'right-turn': right_turn, 'left-turn': left_turn}
class_names = {0: 'left', 1:'right', 2:'straight', 3:'right-turn', 4: 'left-turn'}
Traffic_class_names = {0:'Stop', 1:'Go'}
number = 817
photo_count = 0
try:
    while True:

        raw_image = camera.capture_array()

        raw_image_colored = cv2.cvtColor(raw_image, cv2.COLOR_RGBA2RGB)

        raw_image_colored = cv2.cvtColor(raw_image_colored, cv2.COLOR_BGR2RGB)

        raw_image_resized = cv2.resize(raw_image_colored, (384, 216))

        RGB_frame_arr = np.array(raw_image_resized)

        RGB_frame_array = (RGB_frame_arr / 255.0)

        #cv2.imwrite(f'/home/hemo2995/Proggraming/evading/Traffic_data/a.jpg', RGB_frame_array)

        RGB_frame_array = RGB_frame_array[np.newaxis, :, :, :]

        RGB_frame_tensor = tf.convert_to_tensor(RGB_frame_array)

        RGB_frame_tensor = tf.cast(RGB_frame_tensor, tf.float32)
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2GRAY)

        resized_frame = cv2.resize(image, (480, 270))

        frame_array = np.array(resized_frame)

        frame_array = frame_array[np.newaxis, :, :, np.newaxis]

        frame_array = frame_array / 255.0
    
        frame_tensor = tf.convert_to_tensor(frame_array)
        
        frame_tensor = tf.cast(frame_tensor, tf.float32)

        interpreter.set_tensor(input_details[0]['index'], frame_tensor)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        prediction = class_names[np.argmax(predictions)]
        #Traffic_interpreter.set_tensor(Traffic_input_details[0]['index'], RGB_frame_tensor)
        #Traffic_interpreter.invoke()
        #Traffic_predictions = Traffic_interpreter.get_tensor(Traffic_output_details[0]['index'])
        print(prediction)
        
        #Traffic_prediction = Traffic_class_names[np.argmax(Traffic_predictions)]
        #print(Traffic_prediction)
        #distance = Distance.distance * 100
        
        #Threshold = distance
        function_mapping[prediction]()
        #elif distance < Threshold and Traffic_prediction == 'Go':
        #    right()
        #    print('left')
        #    time.sleep(2.5)
        #    left()
        #    print('right')
        #    time.sleep(3)
            #left()
            #time.sleep(1.2)
        #elif Traffic_prediction == 'Stop':
        #    kit.servo[0].angle = 75
        #    kit.servo[1].angle = 20

        if photo_count >= 1:# and Traffic_prediction == 'Go':
            #cv2.imwrite(os.path.join(os.getcwd(), 'data', prediction, f'{number}.jpg'), raw_image)
            print(f'image saved to path: {prediction} with number: {number}')
            photo_count = 0
            number += 1
        else:
            photo_count += 1
        #cv2.imwrite(f'/home/hemo2995/Proggraming/evading/data/a.jpg', )
        #cv2.imwrite(f'/home/hemo2995/Proggraming/evading/analyze/{number}.jpg', raw_image)
        #print('Distance: ', Distance.distance * 100)

        time.sleep(0.04)    
except KeyboardInterrupt:
    kit.servo[0].angle = 60
    kit.servo[1].angle = 60
    time.sleep(1)
    

camera.stop()
