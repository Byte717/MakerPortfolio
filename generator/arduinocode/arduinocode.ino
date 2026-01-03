#include <Servo.h>

Servo esc;

const int escPin    = 10;
const int buttonPin = 43;
const int relayPin  = 53;

const unsigned long escArmTime  = 5000; 
const unsigned long spinTime    = 2000; 

int  lastButtonState = HIGH;   // INPUT_PULLUP: HIGH = not pressed

void setup() {
  Serial.begin(9600);

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(relayPin, OUTPUT);

  digitalWrite(relayPin, LOW);
  esc.attach(escPin, 1000, 2000);

}

void loop() {
  int buttonState = digitalRead(buttonPin);

  if (lastButtonState == HIGH && buttonState == LOW) {
    Serial.println("Button pressed: starting sequence");

    digitalWrite(relayPin, HIGH); 
    Serial.println("Relay ON");

    delay(200);
    Serial.println("Arming ESC");
    esc.writeMicroseconds(1000);
    delay(escArmTime);
    Serial.println("ESC armed");

    Serial.println("Spinning motor");
    esc.writeMicroseconds(1800);
    delay(spinTime);

    digitalWrite(relayPin, LOW);
    Serial.println("Relay OFF");
  }

  lastButtonState = buttonState;
  delay(20); 
}
