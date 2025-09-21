

import cv2
import numpy as np
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()

from constants import SHOT_LINE_START_POINT, SHOT_LINE_END_POINT, SHOT_Y_AXIS, FIRST_HOLE, SECOND_HOLE

carom_holes = [FIRST_HOLE, SECOND_HOLE] 
cue_length = 100 

def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera")
        return None
    cap.release()
    return frame

def detect_green_coin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > 10:
            cv2.circle(image, center, int(radius), (0, 255, 0), 2)
            
            return center
    
    return None

def calculate_and_display_shot(image, coin_center, hole):
    global line_start
    
    hole_vector = np.array(hole) - np.array(coin_center)
    hole_vector_length = np.linalg.norm(hole_vector)
    
    if hole_vector_length == 0:
        return None
    
    hole_unit_vector = hole_vector / hole_vector_length
    
    cue_end = (int(coin_center[0] + cue_length * hole_unit_vector[0]), int(coin_center[1] + cue_length * hole_unit_vector[1]))
    
    line_start = (int(coin_center[0] - cue_length * hole_unit_vector[0]), int(coin_center[1] - cue_length * hole_unit_vector[1]))
    
    line_start = (max(SHOT_LINE_START_POINT, min(SHOT_LINE_END_POINT, line_start[0])), SHOT_Y_AXIS)
    
    stepper_motor_length = round((SHOT_LINE_END_POINT - line_start[0]) / 10) + 20
    print('Stepper Motor Length (cm) = ', stepper_motor_length)

    cv2.line(image, line_start, cue_end, (255, 0, 0), 2)
    
    cv2.arrowedLine(image, coin_center, hole, (0, 255, 255), 2)
    
    servo_angle = -(calculate_angle(line_start, coin_center))
    print('Servo Angle =', f'{servo_angle:.0f}')
    
    return line_start, stepper_motor_length, servo_angle

def calculate_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point1[0] - point2[0]
    angle = round(np.arctan2(delta_y, delta_x) * 180 / np.pi)
    return angle

def display_shot(image, coin_center, hole, line_start):
    cv2.line(image, line_start, coin_center, (0, 255, 255), 2)
    cv2.arrowedLine(image, coin_center, hole, (0, 255, 255), 2)
    
    angle_start_to_coin = calculate_angle(line_start, coin_center)
    angle_coin_to_hole = calculate_angle(coin_center, hole)
    
    cv2.putText(image, f'Start to Coin: {angle_start_to_coin:.2f}', (line_start[0], line_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f'Coin to Hole: {angle_coin_to_hole:.2f}', (coin_center[0], coin_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Shot Angles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_best_hole(coin_center, holes):
    best_hole = None
    min_distance = float('inf')
    
    for hole in holes:
        distance = np.linalg.norm(np.array(hole) - np.array(coin_center))
        if distance < min_distance:
            min_distance = distance
            best_hole = hole
            
    return best_hole

def wait_for_arduino_response(ser, timeout=5):
    start_time = time.time()
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(f"Received from Arduino: {line}")
            if "Movement completed." in line:
                return True
        if time.time() - start_time > timeout:
            print("Timeout waiting for Arduino response.")
            return False
        time.sleep(0.1)  # Small delay to avoid busy-waiting

def main():
    while True:
        image = capture_image()
        if image is None:
            print("Failed to capture image")
            continue
        
        coin_center = detect_green_coin(image)
        if coin_center is None:
            print("No green coin detected")
            continue
        
        best_hole = find_best_hole(coin_center, carom_holes)
        
        if best_hole is not None:
            line_start, stepper_motor_length, servo_angle = calculate_and_display_shot(image, coin_center, best_hole)
            if line_start is not None:
                if 0 <= servo_angle <= 180:
                    print(f"Sending Stepper Motor Length: {stepper_motor_length} cm")
                    ser.write(f"{stepper_motor_length}\n".encode('utf-8'))
                    print(f"Sending Servo Angle: {servo_angle}")
                    ser.write(f"{servo_angle}\n".encode('utf-8'))
                    
                    display_shot(image.copy(), coin_center, best_hole, line_start)
                    
                    if wait_for_arduino_response(ser):
                        print("Arduino action completed.")
                        break
                    else:
                        print("Failed to get confirmation from Arduino.")
                else:
                    print("Angle is not within the valid range (0-180 degrees). Please try again.")
            
if _name_ == "_main_":
    main()