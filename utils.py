import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=8)

kit.servo[0].angle = 100 #idle
#kit.servo[0].angle = 140 #level 1
#kit.servo[0].angle = 160 #level 2


kit.servo[1].angle = 45 #idle
#kit.servo[1].angle = 30 #level 1
#kit.servo[1].angle = 10 #level 2


def get_postitions(direction, level):
    left_level = [45, 30, 10]
    right_level = [100, 140, 160]
    if direction == 'right':
        return right_level[level]
    elif direction == 'left':
        return left_level[level]
    else:
        raise 'Direction unknown'


#1 = left
#0 = right