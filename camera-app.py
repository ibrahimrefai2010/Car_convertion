from time import sleep
import picamera2
import cv2
import os
from gpiozero import AngularServo, DistanceSensor
from adafruit_servokit import ServoKit
camera = picamera2.Picamera2()
camera.start()

kit = ServoKit(channels=8)
number = 1111

raw_image = camera.capture_array()
cv2.imwrite(os.path.join(os.getcwd(), 'dataset', 'right', '80.jpg'), raw_image)
try:
    while True:
        raw_image = camera.capture_array()
        kit.servo[0].angle = 90
        kit.servo[1].angle = 50
        print(f'image saved with the title {number}.jpg in ./turn')
        cv2.imwrite(os.path.join(os.getcwd(), 'data', 'left-turn', f'{number}.jpg'), raw_image)
        sleep(0.5)
        number += 1
except KeyboardInterrupt:
    kit.servo[0].angle = 60
    kit.servo[1].angle = 110