from time import sleep
import picamera2
import cv2
import os
from gpiozero import AngularServo, DistanceSensor
from adafruit_servokit import ServoKit
camera = picamera2.Picamera2()
camera.start()

kit = ServoKit(channels=8)
number = 496
left_servo = AngularServo(17, min_pulse_width=0.1/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=90)
right_servo = AngularServo(27, min_pulse_width=0.1/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=90)

raw_image = camera.capture_array()
cv2.imwrite(f'/home/hemo2995/Proggraming/evading/A.jpg', raw_image)
'''while True:
    raw_image = camera.capture_array()
    kit.servo[0].angle = 75
    kit.servo[1].angle = 60
    print(f'image saved with the title {number}.jpg in ./turn')
    cv2.imwrite(f'/home/hemo2995/Proggraming/evading/data/turn/{number}.jpg', raw_image)
    sleep(0.1)
    number += 1
'''