// =====================================================
// WATER QUALITY MONITOR - pH + TURBIDITY + TEMP + TDS
// Arduino Uno
//
// WIRING:
//   pH module:        V+ -> 3.3V,  G -> GND,  Po   -> A1
//   Turbidity module: VCC -> 5V,   GND -> GND, AOUT -> A0
//   TDS module:       VCC -> 5V,   GND -> GND, AOUT -> A2
//   DS18B20 temp:     VCC -> 5V,   GND -> GND, DATA -> D2
//                     + 4.7k resistor between DATA and 5V
//
// LIBRARIES NEEDED:
//   - OneWire
//   - DallasTemperature
//
// pH COMMANDS:   CAL4 / CAL7 / CAL9
// TURB COMMANDS: CALCLEAR / CALDIRTY
// GENERAL:       SHOW / RESET
// =====================================================

#include <EEPROM.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------------- PIN CONFIG ----------------
#define PH_PIN        A1
#define TURBIDITY_PIN A0
#define TDS_PIN       A2
#define TEMP_PIN      2

// ---------------- TEMP SETUP ----------------
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// ---------------- ADC / VOLTAGE ----------------
const float PH_VREF   = 3.3;
const float TURB_VREF = 5.0;
const float TDS_VREF  = 5.0;
const float ADC_MAX_VALUE = 1023.0;

// ---------------- pH BUFFER VALUES (30C) ----------------
const float PH_LOW  = 4.01;
const float PH_MID  = 6.85;
const float PH_HIGH = 9.14;

// ---------------- TURBIDITY NTU MAP ----------------
const float CLEAR_NTU = 0.0;
const float DIRTY_NTU = 1000.0;

// ---------------- TDS ----------------
// Default temperature used if temp sensor fails
const float TDS_DEFAULT_TEMP = 25.0;
// Scaling factor (0.5 typical for Gravity TDS). Increase if
// readings look low vs a known reference, decrease if high.
const float TDS_FACTOR = 0.5;

// ---------------- SAMPLING ----------------
const int SAMPLE_COUNT = 100;
const int SAMPLE_DELAY_MS = 10;
const int TRIM_COUNT = 15;
const float PH_MAX_NOISE_MV   = 50.0;
const float TURB_MAX_NOISE_MV = 150.0;

// ---------------- EEPROM ADDRESSES ----------------
const int PH_EEPROM_ADDR   = 0;
const int TURB_EEPROM_ADDR = 32;

const uint32_t PH_MAGIC   = 0x50484333;
const uint32_t TURB_MAGIC = 0x54555242;

// ---------------- STRUCTS ----------------
struct PHCalibration {
  uint32_t magic;
  float adcAtLow;
  float adcAtMid;
  float adcAtHigh;
};

struct TurbCalibration {
  uint32_t magic;
  float voltageClear;
  float voltageDirty;
};

struct AnalysisResult {
  float filteredADC;
  float voltage;
  float noiseMV;
};

PHCalibration phCal;
TurbCalibration turbCal;
int samples[SAMPLE_COUNT];

// last good temperature (used for TDS correction)
float lastTempC = TDS_DEFAULT_TEMP;

// =====================================================
// EEPROM LOAD / SAVE
// =====================================================

void savePH()   { phCal.magic = PH_MAGIC;   EEPROM.put(PH_EEPROM_ADDR, phCal); }
void saveTurb() { turbCal.magic = TURB_MAGIC; EEPROM.put(TURB_EEPROM_ADDR, turbCal); }

void resetPH() {
  phCal.magic = PH_MAGIC;
  phCal.adcAtLow = -1.0;
  phCal.adcAtMid = -1.0;
  phCal.adcAtHigh = -1.0;
  savePH();
}

void resetTurb() {
  turbCal.magic = TURB_MAGIC;
  turbCal.voltageClear = -1.0;
  turbCal.voltageDirty = -1.0;
  saveTurb();
}

void loadCalibration() {
  EEPROM.get(PH_EEPROM_ADDR, phCal);
  if (phCal.magic != PH_MAGIC) resetPH();

  EEPROM.get(TURB_EEPROM_ADDR, turbCal);
  if (turbCal.magic != TURB_MAGIC) resetTurb();
}

// =====================================================
// CALIBRATION CHECKS
// =====================================================

bool pointValid(float adc) { return adc > 0 && adc < 1023; }

bool phCalibrated() {
  return pointValid(phCal.adcAtLow) &&
         pointValid(phCal.adcAtHigh) &&
         abs(phCal.adcAtLow - phCal.adcAtHigh) > 5;
}

bool turbCalibrated() {
  if (turbCal.voltageClear <= 0 || turbCal.voltageDirty <= 0) return false;
  float diff = turbCal.voltageClear - turbCal.voltageDirty;
  if (diff < 0) diff = -diff;
  return diff > 0.02;
}

// =====================================================
// SHARED ANALOG ANALYSIS
// =====================================================

void sortSamples() {
  for (int i = 1; i < SAMPLE_COUNT; i++) {
    int current = samples[i];
    int j = i - 1;
    while (j >= 0 && samples[j] > current) {
      samples[j + 1] = samples[j];
      j--;
    }
    samples[j + 1] = current;
  }
}

AnalysisResult analysePin(int pin, float vref) {
  analogRead(pin);
  delayMicroseconds(300);

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    samples[i] = analogRead(pin);
    delay(SAMPLE_DELAY_MS);
  }

  sortSamples();

  long sum = 0;
  for (int i = TRIM_COUNT; i < SAMPLE_COUNT - TRIM_COUNT; i++) {
    sum += samples[i];
  }
  int usedSamples = SAMPLE_COUNT - (TRIM_COUNT * 2);
  float filteredADC = sum / (float)usedSamples;

  int lowerADC = samples[TRIM_COUNT];
  int upperADC = samples[SAMPLE_COUNT - TRIM_COUNT - 1];
  float noiseMV = (upperADC - lowerADC) * (vref / ADC_MAX_VALUE) * 1000.0;

  AnalysisResult r;
  r.filteredADC = filteredADC;
  r.voltage = filteredADC * vref / ADC_MAX_VALUE;
  r.noiseMV = noiseMV;
  return r;
}

// =====================================================
// TEMPERATURE READING
// =====================================================

float readTemperature() {
  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  return tempC;   // -127 if not found
}

// =====================================================
// pH CALCULATION
// =====================================================

float calculatePH(float adc) {
  bool haveMid = pointValid(phCal.adcAtMid);

  if (haveMid) {
    float adcMid = phCal.adcAtMid;
    float adcLow = phCal.adcAtLow;
    float adcHigh = phCal.adcAtHigh;

    bool acidicSide;
    if (adcLow > adcHigh) {
      acidicSide = (adc >= adcMid);
    } else {
      acidicSide = (adc <= adcMid);
    }

    if (acidicSide) {
      return PH_LOW + (adc - adcLow) * (PH_MID - PH_LOW) / (adcMid - adcLow);
    } else {
      return PH_MID + (adc - adcMid) * (PH_HIGH - PH_MID) / (adcHigh - adcMid);
    }
  }

  return PH_LOW + (adc - phCal.adcAtLow) *
         (PH_HIGH - PH_LOW) / (phCal.adcAtHigh - phCal.adcAtLow);
}

// =====================================================
// TURBIDITY CALCULATION
// =====================================================

float calculateNTU(float voltage) {
  float denom = turbCal.voltageClear - turbCal.voltageDirty;
  if (denom == 0) denom = 0.0001;

  float ntu = CLEAR_NTU +
              (turbCal.voltageClear - voltage) *
              (DIRTY_NTU - CLEAR_NTU) / denom;

  if (ntu < 0) ntu = 0;
  if (ntu > DIRTY_NTU) ntu = DIRTY_NTU;
  return ntu;
}

// =====================================================
// TDS CALCULATION (temperature compensated)
// Gravity TDS factory formula
// =====================================================

float calculateTDS(float voltage, float tempC) {
  // temperature compensation to 25C
  float compCoeff = 1.0 + 0.02 * (tempC - 25.0);
  float compVoltage = voltage / compCoeff;

  // factory polynomial: converts voltage -> TDS (ppm)
  float tds = (133.42 * compVoltage * compVoltage * compVoltage
             - 255.86 * compVoltage * compVoltage
             + 857.39 * compVoltage) * TDS_FACTOR;

  if (tds < 0) tds = 0;
  return tds;
}

// =====================================================
// CALIBRATION COMMANDS
// =====================================================

void calibratePHpoint(const char* label, float* target) {
  Serial.print("Analysing ");
  Serial.print(label);
  Serial.println(" buffer...");

  AnalysisResult r = analysePin(PH_PIN, PH_VREF);

  if (r.noiseMV > PH_MAX_NOISE_MV) {
    Serial.print("FAILED: Unstable. Noise = ");
    Serial.print(r.noiseMV, 1);
    Serial.println(" mV");
    return;
  }

  *target = r.filteredADC;
  savePH();
  Serial.print(label);
  Serial.print(" saved. ADC = ");
  Serial.println(r.filteredADC, 1);
}

void calibrateTurb(const char* label, float* target) {
  Serial.print("Analysing ");
  Serial.print(label);
  Serial.println(" water...");

  AnalysisResult r = analysePin(TURBIDITY_PIN, TURB_VREF);

  if (r.noiseMV > TURB_MAX_NOISE_MV) {
    Serial.print("WARNING: Noisy (");
    Serial.print(r.noiseMV, 1);
    Serial.println(" mV) but saving anyway.");
  }

  *target = r.voltage;
  saveTurb();
  Serial.print(label);
  Serial.print(" saved. Voltage = ");
  Serial.print(r.voltage, 4);
  Serial.println(" V");
}

void handleCommand() {
  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();

  if (command == "CAL4") {
    calibratePHpoint("pH 4.01", &phCal.adcAtLow);
  }
  else if (command == "CAL7") {
    calibratePHpoint("pH 6.85", &phCal.adcAtMid);
  }
  else if (command == "CAL9") {
    calibratePHpoint("pH 9.14", &phCal.adcAtHigh);
  }
  else if (command == "CALCLEAR") {
    calibrateTurb("CLEAR", &turbCal.voltageClear);
  }
  else if (command == "CALDIRTY") {
    calibrateTurb("DIRTY", &turbCal.voltageDirty);
  }
  else if (command == "SHOW") {
    Serial.println("--- pH calibration ---");
    Serial.print("ADC 4.01 = "); Serial.println(phCal.adcAtLow, 1);
    Serial.print("ADC 6.85 = "); Serial.println(phCal.adcAtMid, 1);
    Serial.print("ADC 9.14 = "); Serial.println(phCal.adcAtHigh, 1);
    Serial.print("pH calibrated? "); Serial.println(phCalibrated() ? "YES" : "NO");
    Serial.println("--- Turbidity calibration ---");
    Serial.print("Clear V = "); Serial.println(turbCal.voltageClear, 4);
    Serial.print("Dirty V = "); Serial.println(turbCal.voltageDirty, 4);
    Serial.print("Turb calibrated? "); Serial.println(turbCalibrated() ? "YES" : "NO");
  }
  else if (command == "RESET") {
    resetPH();
    resetTurb();
    Serial.println("ALL calibration deleted.");
  }
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(200);

  pinMode(PH_PIN, INPUT);
  pinMode(TURBIDITY_PIN, INPUT);
  pinMode(TDS_PIN, INPUT);

  tempSensor.begin();
  loadCalibration();

  Serial.println("=================================");
  Serial.println("WATER QUALITY MONITOR READY");
  Serial.println("pH->A1 Turb->A0 TDS->A2 Temp->D2");
  Serial.println("=================================");
  Serial.println("pH:   CAL4 / CAL7 / CAL9");
  Serial.println("Turb: CALCLEAR / CALDIRTY");
  Serial.println("Gen:  SHOW / RESET");
  Serial.println("=================================");

  if (!phCalibrated())
    Serial.println("[!] pH not calibrated");
  if (!turbCalibrated())
    Serial.println("[!] Turbidity not calibrated");
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {
  handleCommand();

  Serial.println("---------------------------------");

  // ---------- TEMPERATURE ----------
  float tempC = readTemperature();
  bool tempOK = (tempC > -100.0);

  if (tempOK) {
    lastTempC = tempC;   // save for TDS correction
    Serial.print("Temp: ");
    Serial.print(tempC, 1);
    Serial.println(" C");
  } else {
    Serial.println("Temp: ERROR (check wiring & 4.7k resistor)");
  }

  // ---------- pH ----------
  if (phCalibrated()) {
    AnalysisResult phR = analysePin(PH_PIN, PH_VREF);

    if (phR.noiseMV > PH_MAX_NOISE_MV) {
      Serial.println("pH: ERROR (unstable signal)");
    } else {
      float ph = calculatePH(phR.filteredADC);
      if (ph < 0.0 || ph > 14.0) {
        Serial.println("pH: ERROR (invalid calibration)");
      } else {
        Serial.print("pH: ");
        Serial.println(ph, 2);
      }
    }
  } else {
    Serial.println("pH: not calibrated");
  }

  // ---------- TURBIDITY ----------
  if (turbCalibrated()) {
    AnalysisResult tR = analysePin(TURBIDITY_PIN, TURB_VREF);
    float ntu = calculateNTU(tR.voltage);

    Serial.print("Turbidity: ");
    Serial.print(ntu, 1);
    Serial.print(" NTU  ->  ");

    if (ntu <= 5.0) {
      Serial.println("GOOD (safe/clear)");
    } else if (ntu <= 50.0) {
      Serial.println("AVERAGE (slightly cloudy)");
    } else {
      Serial.println("BAD (very cloudy/dirty)");
    }
  } else {
    Serial.println("Turbidity: not calibrated");
  }

  // ---------- TDS ----------
  AnalysisResult tdsR = analysePin(TDS_PIN, TDS_VREF);
  float tds = calculateTDS(tdsR.voltage, lastTempC);

  Serial.print("TDS: ");
  Serial.print(tds, 0);
  Serial.print(" ppm  ->  ");

  // Drinking water quality based on TDS (ppm)
  if (tds <= 300.0) {
    Serial.println("GOOD (excellent/good)");
  } else if (tds <= 600.0) {
    Serial.println("AVERAGE (fair)");
  } else if (tds <= 900.0) {
    Serial.println("POOR");
  } else {
    Serial.println("BAD (unacceptable)");
  }

  delay(1500);
}
