from gpiozero import AngularServo, DistanceSensor
from time import sleep
import cv2
import time

left_servo = AngularServo(17, min_pulse_width=0.09/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=90)
right_servo = AngularServo(27, min_pulse_width=0.09/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=90)
Distance = DistanceSensor(echo=23, trigger=24)
#def left():
#    left_servo.angle = 0
#    right_servo.angle = 45


#def right():
#    left_servo.angle = 45
#    right_servo.angle = 0

#def straight():
#    left_servo.angle = 45
#    right_servo.angle = 45

#def turn():
#    left_servo.angle = 0
#    right_servo.angle = 90

#def manuver():
#    left_servo.angle = 0
#    right_servo.angle = 50
#    time.sleep(2)
#    left_servo.angle = 50
#    right_servo.angle = 0
#    sleep(4)

#left_servo.angle = 0
#right_servo.angle = 0
#sleep(2)


left_servo.angle = 80
right_servo.angle = 80
sleep(1)

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
