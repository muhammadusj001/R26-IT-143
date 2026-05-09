#include <OneWire.h>
#include <DallasTemperature.h>

// ===== TEMPERATURE SENSOR (DS18B20) =====
#define ONE_WIRE_BUS 2
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ===== TURBIDITY SENSOR =====
const int turbidityPin = A0;
const int turbiditySamples = 10;

// ===== PH SENSOR =====
const int phPin = A1;
const int phSamples = 20; // Increased for better stability

/* CALIBRATION: 
  1. Put probe in clean water. 
  2. Look at 'pH Voltage' in Serial Monitor. 
  3. If pH is not 7.0, adjust 'voltageOffset' below.
  Current Voltage (4.16) - Neutral Target (2.5) = ~1.66
*/
float voltageOffset = 1.66; 

void setup() {
  Serial.begin(9600);
  sensors.begin();
  
  pinMode(turbidityPin, INPUT);
  pinMode(phPin, INPUT);
  
  Serial.println("==================================");
  Serial.println("Water Quality Monitoring System");
  Serial.println("==================================");
}

void loop() {
  // 1. TEMPERATURE SENSOR
  sensors.requestTemperatures();
  float temperatureC = sensors.getTempCByIndex(0);

  // 2. TURBIDITY SENSOR (Averaging)
  long turbSum = 0;
  for (int i = 0; i < turbiditySamples; i++) {
    turbSum += analogRead(turbidityPin);
    delay(10);
  }
  float turbAverage = turbSum / turbiditySamples;
  float turbVoltage = turbAverage * (5.0 / 1023.0);

  // 3. PH SENSOR (Averaging + Calibration)
  long phSum = 0;
  for (int i = 0; i < phSamples; i++) {
    phSum += analogRead(phPin);
    delay(10);
  }
  float phAverage = (float)phSum / phSamples;
  float phVoltage = phAverage * (5.0 / 1023.0);
  
  // Adjusted PH Formula
  // We subtract the offset to bring your 4.16V closer to the 2.5V neutral point
  float correctedVoltage = phVoltage - voltageOffset;
  float phValue = 3.5 * correctedVoltage; 
  
  // Constrain pH to realistic limits (0-14)
  if (phValue < 0) phValue = 0;
  if (phValue > 14) phValue = 14;

  // ===== SERIAL OUTPUT =====
  Serial.println("--- NEW READING ---");
  
  Serial.print("Temp:      ");
  Serial.print(temperatureC);
  Serial.println(" °C");

  Serial.print("Turbidity: ");
  Serial.print(turbAverage, 0); 
  Serial.print(" ("); Serial.print(turbVoltage); Serial.println("V)");

  Serial.print("pH Level:  ");
  Serial.print(phValue);
  Serial.print(" (Raw: "); Serial.print(phVoltage); Serial.println("V)");
  
  Serial.println("-------------------");

  delay(3000); // 3-second interval for readability
}