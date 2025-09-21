#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Define pins for the stepper motor driver
int stepPin = 2;
int directionPin = 3;

// Conversion ratios based on provided measurements
const float stepsPerCm = 5000.0 / 5.1; // Steps per centimeter

// Declare steps19cm globally
long steps19cm;

// Create servo objects
Servo baseArmServo;     // MG995 for the base arm
Servo secondArmServo;   // MG995 for the second arm
Servo thirdArmServo;    // MG90S for the third arm
Servo myServo;          // Additional servo for solenoid rotation

// Define the pin numbers
const int baseArmPin = 9;
const int secondArmPin = 10;
const int thirdArmPin = 5;
const int servoPin = 8;
const int relayPin = 7; // Pin connected to relay module

// Current angle of the additional servo
int currentAngle = 90;  // Initial angle (starting position)

// Set the LCD address to 0x3F for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Variables to store received data
float distance;
int targetAngle;

// Pins for the Ultrasonic Sensor
const int trigPin = A2;
const int echoPin = A3;

// Define the range tolerance
const float tolerance = 3.0; // in cm

// Distance from the thread bar end to the arm's starting point
const float fixedDistance = 9.8; // in cm

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(directionPin, OUTPUT);
  pinMode(relayPin, OUTPUT);

  // Attach the servos to their respective pins
  baseArmServo.attach(baseArmPin);
  secondArmServo.attach(secondArmPin);
  thirdArmServo.attach(thirdArmPin);
  myServo.attach(servoPin);

  // Set initial positions for the servos
  baseArmServo.write(90);      // Base arm starting position
  secondArmServo.write(52);    // Second arm starting position
  thirdArmServo.write(5);      // Third arm starting position
  myServo.write(90);           // Solenoid rotation servo starting position

  // Initialize the LCD with the number of columns and rows
  lcd.init();
  lcd.backlight();

  Serial.begin(9600); // Initialize serial communication
  Serial.println("Starting...");

  String phrase1 = "WELCOME TO STRIKEMATE !";
  displayCentered(phrase1);
  delay(2000);
  String phrase2 = "Let's Start ...";
  displayCentered(phrase2);
  delay(2000);
  String phrase3 = "Your coin colour is blue";
  displayCentered(phrase3);
  delay(2000);
  String phrase4 = "GAME ON";
  displayCentered(phrase4);
  delay(2000);

  // Calculate steps for 19 cm
  steps19cm = distanceToSteps(19);

  // Initialize the pins for the Ultrasonic Sensor
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  // Check if there's any serial input
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    if (message == "Blue wins!" || message == "Green wins!") {
      displayCentered(message);
      delay(2000);
      return;
    } else {
      distance = Serial.parseFloat();
      targetAngle = Serial.parseInt();

      Serial.print("Distance received: ");
      Serial.println(distance);
      Serial.print("Target angle received: ");
      Serial.println(targetAngle);

      // Measure distance using the Ultrasonic Sensor
      float measuredDistance = measureDistance(trigPin, echoPin);

      // Check if the measured distance is within the tolerance range
      bool withinRange = isWithinRange(measuredDistance, distance, tolerance);

      // Decide which distance to use for arm movement
      float finalDistance = withinRange ? distance : measuredDistance;
      long stepsInput = distanceToSteps(finalDistance);

      if (stepsInput >= 0) {
        // Move the input distance in the HIGH direction
        digitalWrite(directionPin, HIGH);
        moveStepper(stepsInput);

        // Claw movements
        thirdArmServo.write(3);

        // Move the base arm continuously
        for (int pos = 90; pos >= 30; pos -= 1) {
          baseArmServo.write(pos);
          delay(50);
        }
        secondArmServo.write(52);
        delay(1000);

        // Move the third arm servo
        delay(3000);
        thirdArmServo.write(60);
        delay(1000);
        thirdArmServo.write(3);
        delay(1000);

        // Move the base arm back to its original position
        for (int pos = 30; pos <= 90; pos += 1) {
          baseArmServo.write(pos);
          delay(50);
        }
        delay(2000);

        // Move 6 cm in the LOW direction
        long steps6cm = distanceToSteps(6);
        digitalWrite(directionPin, LOW);
        moveStepper(steps6cm);

        // Handle solenoid rotation
        handleSolenoidRotation(targetAngle);

        // Move back to the starting position
        long stepsBack = stepsInput - steps6cm;
        moveStepper(stepsBack);

        Serial.println("Movement completed.");
        String phrase = "Your Turn ...";
        displayCentered(phrase);
        delay(2000);

      } else {
        Serial.println("Invalid distance. Please enter a valid cm value:");
      }

      // Prevent further execution of the loop
      while (1) {
        // Do nothing
      }
    }
  }
}

// Function to convert distance to steps
long distanceToSteps(float distance) {
  if (distance > 0) {
    return distance * stepsPerCm;
  } else {
    return -1; // Invalid distance
  }
}

// Function to move the stepper motor
void moveStepper(long steps) {
  for (long stepCount = 0; stepCount < steps; stepCount++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(700);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(700);
  }
}

// Function to rotate the servo gradually
void rotateServoGradually(int targetAngle) {
  Serial.print("Current angle: ");
  Serial.println(currentAngle);
  Serial.print("Target angle: ");
  Serial.println(targetAngle);

  if (targetAngle > currentAngle) {
    for (int angle = currentAngle; angle <= targetAngle; angle++) {
      myServo.write(angle);
      delay(20);
    }
  } else {
    for (int angle = currentAngle; angle >= targetAngle; angle--) {
      myServo.write(angle);
      delay(20);
    }
  }
  currentAngle = targetAngle;
}

// Function to display a centered message on the LCD
void displayCentered(String message) {
  lcd.clear();

  int messageLength = message.length();
  if (messageLength <= 16) {
    int startCol = (16 - messageLength) / 2;
    lcd.setCursor(startCol, 0);
    lcd.print(message);
  } else {
    String line1 = message.substring(0, 16);
    String line2 = message.substring(16);
    int startCol1 = (16 - line1.length()) / 2;
    int startCol2 = (16 - line2.length()) / 2;

    lcd.setCursor(startCol1, 0);
    lcd.print(line1);

    lcd.setCursor(startCol2, 1);
    lcd.print(line2);
  }
}

// Function to handle solenoid rotation
void handleSolenoidRotation(int targetAngle) {
  if (targetAngle >= 0 && targetAngle <= 180) {
    rotateServoGradually(targetAngle);
    Serial.print("Servo rotated to ");
    Serial.print(targetAngle);
    Serial.println(" degrees.");
    delay(2000);

    // Solenoid activation
    digitalWrite(relayPin, HIGH);
    delay(1000);
    digitalWrite(relayPin, LOW);

    rotateServoGradually(90);
    Serial.println("Servo returned to 90 degrees.");
  } else {
    Serial.println("Invalid input. Enter an angle between 0 and 180:");
  }
}

// Function to measure distance using the Ultrasonic Sensor
float measureDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  float distance = (duration / 2.0) * 0.0343;
  return distance;
}

// Function to check if the measured distance is within the tolerance range
bool isWithinRange(float measuredDistance, float targetDistance, float tolerance) {
  return (measuredDistance >= (targetDistance - tolerance)) && (measuredDistance <= (targetDistance + tolerance));
}
