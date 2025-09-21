

import RPi.GPIO as GPIO
import subprocess
import time
import os
import signal

BUTTON_PIN = 2 

current_process = None

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def run_script():
    global current_process

    script_path = '/home/pi/Desktop/WinorLose.py'

    if current_process is not None:
        current_process.terminate()
        current_process.wait()  # Wait for the process to terminate

    current_process = subprocess.Popen(['python3', script_path])

try:
    while True:
        GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
        run_script()
        time.sleep(0.2)  # Debounce time

except KeyboardInterrupt:
    GPIO.cleanup()

finally:
    GPIO.cleanup()