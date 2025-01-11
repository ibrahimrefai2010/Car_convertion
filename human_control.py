import picamera2
import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpiozero import AngularServo, DistanceSensor
import os
import keyboard
import tty
import termios
import sys

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

#model = tf.keras.models.load_model("98percentSSL.keras")

camera = picamera2.Picamera2()
camera.start()

right_servo = AngularServo(27, min_pulse_width=0.10/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=180)
left_servo = AngularServo(17, min_pulse_width=0.10/100, max_pulse_width=0.20/100, frame_width=0.020, min_angle=0, max_angle=90)

def right():
    right_servo.angle = 175
    left_servo.angle = 10


def left():
    right_servo.angle = 0
    left_servo.angle = 90

def straight():
    right_servo.angle = 180
    left_servo.angle = 70

class_names = {0: 'left', 1:'right', 2:'straight'}
number = 546
try:
    while True:
        print('first')
        raw_image = camera.capture_array()
        print('second')
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        resized_frame = cv2.resize(image, (480, 270))
    
        frame_arr = np.array(resized_frame)
     
        frame_array = frame_arr[np.newaxis, :, :, np.newaxis]
    
        frame_tensor = tf.convert_to_tensor(frame_array)

        prediction = ''
        if getch() == 'w':
            prediction = 'straight'
        elif getch() == 'a':
            prediction = 'left'
        elif getch() == 'd':
            prediction = 'right'

        #predictions = model.predict(frame_tensor)
        path = ''
        if prediction == 'right':
            number += 1
            path = './psuedo-labeled/right'
        elif prediction == 'left':
            number += 1
            path = './psuedo-labeled/left/'
        elif prediction == 'straight':
            number += 1
            path = './psuedo-labeled/straight/'
        
        #cv2.imwrite(os.path.join(path , f'{number}.jpg'), raw_image)
        #print(f'image saved to path: {path} with number: {number}')
        if prediction == 'left':
            left()
        elif prediction == 'right':
            right()
        elif prediction == 'straight':
            straight()   
        else: 
            right_servo.angle = 60
            left_servo.angle = 10

        print(prediction)

except KeyboardInterrupt:
    right_servo.angle = 60
    left_servo.angle = 10
    time.sleep(1)
    

camera.stop()
