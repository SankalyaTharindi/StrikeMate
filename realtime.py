
import cv2
import numpy as np

# Initialize camera capture
cap = cv2.VideoCapture(0)

def calibrate_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sample regions for red, white, and black coins
    red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))

    # Find contours to get average HSV values
    def get_average_hsv(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hsv_values = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            roi = hsv[y:y+h, x:x+w]
            hsv_values.extend(roi.reshape(-1, 3))
        return np.mean(hsv_values, axis=0)


    red_avg = get_average_hsv(red_mask)
    white_avg = get_average_hsv(white_mask)
    black_avg = get_average_hsv(black_mask)

    # Define ranges using mean values (you can adjust the tolerance)
    def define_range(avg):
        return (np.clip(avg - 10, 0, 255), np.clip(avg + 10, 0, 255))

    return {
        'red_max': define_range(red_avg)[1],
        'red_min': define_range(red_avg)[0],
        'white_max': define_range(white_avg)[1],
        'white_min': define_range(white_avg)[0],
        'black_max': define_range(black_avg)[1],
        'black_min': define_range(black_avg)[0],
    }

def detect_coins(frame, calib_data):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Hough Circle detection
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=15, minRadius=7, maxRadius=10)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Create masks for detected circles
        mask = np.zeros_like(gray)
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], 255, thickness=-1)

        # Apply masks to original frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

        white_count = 0
        black_count = 0
        red_count = 0

        for i in circles[0, :]:
            hsv_val = hsv[i[1], i[0]]

            if calib_data.get('red_min') is not None and calib_data.get('red_max') is not None:
                if calib_data['red_min'][0] <= hsv_val[0] <= calib_data['red_max'][0] and calib_data['red_min'][1] <= hsv_val[1] <= calib_data['red_max'][1]:
                    red_count += 1
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 3)

            if calib_data.get('white_min') is not None and calib_data.get('white_max') is not None:
                if calib_data['white_min'][0] <= hsv_val[0] <= calib_data['white_max'][0] and calib_data['white_min'][1] <= hsv_val[1] <= calib_data['white_max'][1]:
                    white_count += 1
                    cv2.circle(frame, (i[0], i[1]), i[2], (255, 255, 255), 3)

            if calib_data.get('black_min') is not None and calib_data.get('black_max') is not None:
                if calib_data['black_min'][0] <= hsv_val[0] <= calib_data['black_max'][0] and calib_data['black_min'][1] <= hsv_val[1] <= calib_data['black_max'][1]:
                    black_count += 1
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 0), 3)

        # Display counts
        cv2.putText(frame, f'Red Coins: {red_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'White Coins: {white_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Black Coins: {black_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame

# Calibrate colors using an initial frame
ret, initial_frame = cap.read()
if not ret:
    print("Failed to capture initial frame for calibration.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

calib_data = calibrate_colors(initial_frame)

# Define the desired output frame size (width, height)
output_width = 640
output_height = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to desired output size
    frame = cv2.resize(frame, (output_width, output_height))

    # Detect and classify coins
    output_frame = detect_coins(frame, calib_data)

    # Display the frame
    cv2.imshow("Coin Detection", output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()