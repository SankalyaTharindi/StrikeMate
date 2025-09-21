
import cv2
import numpy as np
import serial
import time
import json

from constants import SHOT_LINE_START_POINT, SHOT_LINE_END_POINT, SHOT_Y_AXIS, FIRST_HOLE, SECOND_HOLE

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()

carom_holes = [FIRST_HOLE, SECOND_HOLE]
cue_length = 100

coin_count_file = 'coin_counts.json'

def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera")
        return None
    cap.release()
    return frame

def detect_coins(image, lower_color, upper_color, color_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_count = 0

    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            coin_count += 1
            center = (int(x), int(y))
            cv2.circle(image, center, int(radius), (0, 255, 0), 2)
            cv2.putText(image, color_name, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return coin_count

def count_coins(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Initial range 1 for red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # Initial range 2 for red
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    
    # Combine both red masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2)
    
    red_count = detect_coins(image, lower_red1, upper_red1, "Red")
    green_count = detect_coins(image, lower_green, upper_green, "Green")
    blue_count = detect_coins(image, lower_blue, upper_blue, "Blue")

    return red_count, green_count, blue_count

def save_coin_counts(red_count, green_count, blue_count):
    coin_counts = {
        "red_count": red_count,
        "green_count": green_count,
        "blue_count": blue_count
    }
    with open(coin_count_file, 'w') as file:
        json.dump(coin_counts, file)

def load_previous_counts():
    try:
        with open(coin_count_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"red_count": 0, "green_count": 0, "blue_count": 0}

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

def find_best_hole(coin_center, holes):
    best_hole = None
    min_distance = float('inf')
    
    for hole in holes:
        distance = np.linalg.norm(np.array(hole) - np.array(coin_center))
        if distance < min_distance:
            min_distance = distance
            best_hole = hole
            
    return best_hole

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
        time.sleep(0.1)

def check_win_condition(red_count, green_count, blue_count):
    if blue_count == 0:
        print("Blue wins!")
        ser.write("0".encode('utf-8'))
        return True
    elif green_count == 0:
        print("Green wins!")
        ser.write("1".encode('utf-8'))
        return True
    return False

def main():
    previous_counts = load_previous_counts()
    print(f"Previous counts: Red={previous_counts['red_count']}, Green={previous_counts['green_count']}, Blue={previous_counts['blue_count']}")

    while True:
        image = capture_image()
        if image is None:
            print("Failed to capture image")
            continue

        red_count, green_count, blue_count = count_coins(image)
        print(f"Current counts: Red={red_count}, Green={green_count}, Blue={blue_count}")

        red_diff = previous_counts["red_count"] - red_count
        green_diff = previous_counts["green_count"] - green_count
        blue_diff = previous_counts["blue_count"] - blue_count

        print(f"Coins removed: Red={red_diff}, Green={green_diff}, Blue={blue_diff}")

        save_coin_counts(red_count, green_count, blue_count)

        if check_win_condition(red_count, green_count, blue_count):
            break

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
            else:
                print("Unable to calculate a valid shot line start point.")

if _name_ == "_main_":
    main()