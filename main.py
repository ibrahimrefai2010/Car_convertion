from gpiozero import AngularServo, DistanceSensor
from time import sleep
import cv2
import time
from adafruit_servokit import ServoKit

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

#left_servo.angle = 0
#right_servo.angle = 0
#sleep(2)


right()
sleep(2)
#straight()
#sleep(2)

kit.servo[0].angle = 60
kit.servo[1].angle = 45

'''while True:
    print(Distance.distance)
    sleep(0.5)
'''

#while cap.isOpened():
#    ret, frame = cap.read()
    
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #resized_frame = cv2.resize(gray_frame, (480, 270))
    
    #frame_array = np.array(resized_frame)
    
    #frame_array = frame_array[np.newaxis, :, :, np.newaxis]
    
    #predictions = model.predict(frame_array)
    
    #if (predictions[0][0] > predictions[0][1]):
    #    print('left')
    #elif(predictions[0][1] > predictions[0][0]):
    #    print('right')
    
    #print(predictions)
#    cv2.imshow("Frame", frame)

#    if measure_distance( ) < 10:
#        gas_off()
#        brake_on()
#    else:
#        brake_off()
#        gas_on()
#    sleep(0.2)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
