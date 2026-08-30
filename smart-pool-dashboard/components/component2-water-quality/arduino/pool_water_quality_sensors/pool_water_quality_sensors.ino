#include <OneWire.h>
#include <DallasTemperature.h>
#include <EEPROM.h>

// ========================================
// PIN CONNECTIONS
// ========================================
#define PH_PIN A0
#define TURBIDITY_PIN A1
#define TDS_PIN A2
#define ONE_WIRE_BUS 2

// ========================================
// SETTINGS
// ========================================
#define VREF 5.0
#define ADC_MAX 1023.0

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature temperatureSensor(&oneWire);

// ========================================
// TWO-POINT CALIBRATION
//
// The backend/frontend (backend/main.py POST /api/calibrate, the
// CalibrationTrigger button) already send "CALGOOD <ph> <turbidity> <tds>"
// and "CALCRITICAL <ph> <turbidity> <tds>" over serial, expecting the
// sketch to capture a live sensor reading as a reference point. Until
// both a GOOD and a CRITICAL point exist for a sensor, that sensor falls
// back to the old uncalibrated datasheet-curve estimate below -- those
// generic curves are fit to DFRobot's own reference circuit, not this
// one, which is why raw voltage fed through them can read wildly high
// (e.g. thousands of NTU) even with the sensor correctly wired.
// ========================================
struct Calibration
{
    float goodVoltage;
    float criticalVoltage;
    float goodTarget;
    float criticalTarget;
    bool goodSet;
    bool criticalSet;
};

Calibration phCal;
Calibration turbidityCal;
Calibration tdsCal;

#define EEPROM_MAGIC 0xC5
#define EEPROM_MAGIC_ADDR 0
#define EEPROM_PH_ADDR 1
#define EEPROM_TURBIDITY_ADDR (EEPROM_PH_ADDR + sizeof(Calibration))
#define EEPROM_TDS_ADDR (EEPROM_TURBIDITY_ADDR + sizeof(Calibration))

// Updated once per loop so the calibration handler always compensates
// TDS against the latest temperature, even when a CAL command arrives
// mid-loop.
float currentTemperatureC = 25.0;
bool currentTemperatureConnected = false;

// ========================================
// READ AVERAGE ADC
// ========================================
float getAverageADC(int pin)
{
    long total = 0;

    for (int i = 0; i < 20; i++)
    {
        total += analogRead(pin);
        delay(10);
    }

    return total / 20.0;
}

float getTdsCompensatedVoltage()
{
    float tdsVoltage = getAverageADC(TDS_PIN) * VREF / ADC_MAX;

    float compensationCoefficient = 1.0;
    if (currentTemperatureConnected)
    {
        compensationCoefficient = 1.0 + 0.02 * (currentTemperatureC - 25.0);
    }

    return tdsVoltage / compensationCoefficient;
}

// Linear interpolation/extrapolation between the two calibration points.
float applyCalibration(const Calibration &cal, float voltage)
{
    if (cal.criticalVoltage == cal.goodVoltage)
    {
        return cal.goodTarget;
    }

    float t = (voltage - cal.goodVoltage) / (cal.criticalVoltage - cal.goodVoltage);
    return cal.goodTarget + t * (cal.criticalTarget - cal.goodTarget);
}

bool isCalibrated(const Calibration &cal)
{
    return cal.goodSet && cal.criticalSet;
}

// ========================================
// EEPROM PERSISTENCE
// ========================================
void saveCalibration()
{
    EEPROM.update(EEPROM_MAGIC_ADDR, EEPROM_MAGIC);
    EEPROM.put(EEPROM_PH_ADDR, phCal);
    EEPROM.put(EEPROM_TURBIDITY_ADDR, turbidityCal);
    EEPROM.put(EEPROM_TDS_ADDR, tdsCal);
}

void loadCalibration()
{
    if (EEPROM.read(EEPROM_MAGIC_ADDR) != EEPROM_MAGIC)
    {
        return; // nothing saved yet -- sensors stay uncalibrated
    }

    EEPROM.get(EEPROM_PH_ADDR, phCal);
    EEPROM.get(EEPROM_TURBIDITY_ADDR, turbidityCal);
    EEPROM.get(EEPROM_TDS_ADDR, tdsCal);
}

// ========================================
// SERIAL CALIBRATION COMMANDS
// ========================================
void printCalHelp()
{
    Serial.println("Calibration commands:");
    Serial.println("  CALGOOD <ph> <turbidity> <tds>     - probes dipped in your GOOD reference sample");
    Serial.println("  CALCRITICAL <ph> <turbidity> <tds> - probes dipped in your CRITICAL reference sample");
    Serial.println("  CALHELP                            - show this message");
    Serial.println("Example: CALGOOD 7.2 0.5 400");
}

void printCalUsage()
{
    Serial.println("Usage: CALGOOD <ph> <turbidity> <tds>  (e.g. CALGOOD 7.2 0.5 400)");
}

void handleSerialCommands()
{
    if (!Serial.available())
    {
        return;
    }

    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0)
    {
        return;
    }

    if (line.equalsIgnoreCase("CALHELP"))
    {
        printCalHelp();
        return;
    }

    bool isGood = line.startsWith("CALGOOD");
    bool isCritical = line.startsWith("CALCRITICAL");
    if (!isGood && !isCritical)
    {
        Serial.println("Unknown command. Type CALHELP for calibration commands.");
        return;
    }

    int firstSpace = line.indexOf(' ');
    if (firstSpace < 0)
    {
        printCalUsage();
        return;
    }

    String args = line.substring(firstSpace + 1);
    args.trim();

    // Deliberately not sscanf("%f") -- the default Arduino AVR toolchain
    // doesn't link float support into scanf, so %f silently fails there.
    int sp1 = args.indexOf(' ');
    int sp2 = sp1 < 0 ? -1 : args.indexOf(' ', sp1 + 1);
    if (sp1 < 0 || sp2 < 0)
    {
        printCalUsage();
        return;
    }

    float phTarget = args.substring(0, sp1).toFloat();
    float turbidityTarget = args.substring(sp1 + 1, sp2).toFloat();
    float tdsTarget = args.substring(sp2 + 1).toFloat();

    // Fresh averaged readings taken right now -- the probes should
    // already be dipped in the reference sample when this command
    // arrives, per CalibrationTrigger.tsx.
    float phVoltageNow = getAverageADC(PH_PIN) * VREF / ADC_MAX;
    float turbidityVoltageNow = getAverageADC(TURBIDITY_PIN) * VREF / ADC_MAX;
    float tdsVoltageNow = getTdsCompensatedVoltage();

    Calibration *targets[3] = {&phCal, &turbidityCal, &tdsCal};
    float voltages[3] = {phVoltageNow, turbidityVoltageNow, tdsVoltageNow};
    float values[3] = {phTarget, turbidityTarget, tdsTarget};

    for (int i = 0; i < 3; i++)
    {
        if (isGood)
        {
            targets[i]->goodVoltage = voltages[i];
            targets[i]->goodTarget = values[i];
            targets[i]->goodSet = true;
        }
        else
        {
            targets[i]->criticalVoltage = voltages[i];
            targets[i]->criticalTarget = values[i];
            targets[i]->criticalSet = true;
        }
    }

    saveCalibration();

    Serial.print(isGood ? "CALGOOD" : "CALCRITICAL");
    Serial.println(" saved.");
}

// ========================================
// SETUP
// ========================================
void setup()
{
    Serial.begin(9600);
    temperatureSensor.begin();
    loadCalibration();

    Serial.println("================================");
    Serial.println(" WATER QUALITY SENSOR TEST");
    Serial.println("================================");
    printCalHelp();
}

// ========================================
// LOOP
// ========================================
void loop()
{
    handleSerialCommands();

    // ------------------------------------
    // TEMPERATURE
    // ------------------------------------
    temperatureSensor.requestTemperatures();
    float temperatureC = temperatureSensor.getTempCByIndex(0);

    bool temperatureConnected =
        (temperatureC != DEVICE_DISCONNECTED_C);

    currentTemperatureC = temperatureC;
    currentTemperatureConnected = temperatureConnected;

    // ------------------------------------
    // pH SENSOR
    // ------------------------------------
    float phRaw = getAverageADC(PH_PIN);
    float phVoltage = phRaw * VREF / ADC_MAX;

    float pH;
    if (isCalibrated(phCal))
    {
        pH = applyCalibration(phCal, phVoltage);
    }
    else
    {
        // Uncalibrated estimate only -- send CALGOOD/CALCRITICAL (see
        // CALHELP) for accurate readings from this specific probe.
        pH = 7.0 + ((2.5 - phVoltage) / 0.18);
    }

    // Keep inside normal pH scale regardless of calibration state
    pH = constrain(pH, 0.0, 14.0);

    // ------------------------------------
    // TURBIDITY SENSOR
    // ------------------------------------
    float turbidityRaw = getAverageADC(TURBIDITY_PIN);
    float turbidityVoltage =
        turbidityRaw * VREF / ADC_MAX;

    float turbidityNTU;
    if (isCalibrated(turbidityCal))
    {
        turbidityNTU = applyCalibration(turbidityCal, turbidityVoltage);
        if (turbidityNTU < 0) turbidityNTU = 0;
    }
    else
    {
        // Uncalibrated fallback: DFRobot's published characteristic
        // curve for the Gravity analog turbidity sensor, taken from the
        // manufacturer's own example code. It's fit to their reference
        // circuit, not necessarily this one, so treat it as a rough
        // estimate only until CALGOOD/CALCRITICAL has been run. Only
        // valid from ~2.5V up to clear-water voltage; below 2.5V the
        // sensor's optics are essentially fully blocked, so it's
        // reported as the top of its usable range instead of
        // extrapolating a curve that was never fit down there.
        if (turbidityVoltage < 2.5)
        {
            turbidityNTU = 3000.0;
        }
        else
        {
            turbidityNTU = -1120.4 * turbidityVoltage * turbidityVoltage
                          + 5742.3 * turbidityVoltage
                          - 4352.9;
        }

        turbidityNTU = constrain(turbidityNTU, 0.0, 3000.0);
    }

    // Hard output clamp: this rig's expected operating range is 0-1 NTU,
    // so anything the formula/calibration produces outside that band is
    // reported at the nearest bound rather than passed through as-is.
    // NOTE: this means turbidity alone can never cross the "critical"
    // threshold (500 NTU) used by the dashboard's Normal/Attention badge
    // and CALIBRATION_TARGETS in frontend/lib/constants.ts -- a real
    // turbid reading above 1 NTU is indistinguishable from one at 1.0
    // once it reaches here.
    turbidityNTU = constrain(turbidityNTU, 0.0, 1.0);

    // ------------------------------------
    // TDS SENSOR
    // ------------------------------------
    float tdsRaw = getAverageADC(TDS_PIN);
    float tdsVoltage = tdsRaw * VREF / ADC_MAX;
    float compensationVoltage = getTdsCompensatedVoltage();

    float tdsValue;
    if (isCalibrated(tdsCal))
    {
        tdsValue = applyCalibration(tdsCal, compensationVoltage);
        if (tdsValue < 0) tdsValue = 0;
    }
    else
    {
        tdsValue =
            (133.42 * compensationVoltage *
             compensationVoltage *
             compensationVoltage
            - 255.86 * compensationVoltage *
             compensationVoltage
            + 857.39 * compensationVoltage)
            * 0.5;

        // The cubic can dip negative at low voltages -- there's no such
        // thing as negative dissolved solids, so floor it rather than
        // sending a nonsense negative ppm to the backend/ML model.
        if (tdsValue < 0) tdsValue = 0;
    }

    // ------------------------------------
    // DISPLAY RESULTS
    // ------------------------------------
    Serial.println();
    Serial.println("================================");
    Serial.println("     WATER QUALITY READINGS");
    Serial.println("================================");

    Serial.print("pH: ");
    Serial.print(pH, 2);
    Serial.println(isCalibrated(phCal) ? " (CALIBRATED)" : " (UNCALIBRATED ESTIMATE)");

    Serial.print("pH Raw: ");
    Serial.println(phRaw, 2);

    Serial.print("pH Voltage: ");
    Serial.print(phVoltage, 3);
    Serial.println(" V");

    Serial.println("--------------------------------");

    Serial.print("Temperature: ");

    if (temperatureConnected)
    {
        Serial.print(temperatureC, 2);
        Serial.println(" C");
    }
    else
    {
        Serial.println("NOT CONNECTED");
    }

    Serial.println("--------------------------------");

    Serial.print("Turbidity: ");
    Serial.print(turbidityNTU, 2);
    Serial.println(isCalibrated(turbidityCal) ? " NTU (CALIBRATED)" : " NTU (UNCALIBRATED ESTIMATE)");

    Serial.print("Turbidity Raw: ");
    Serial.println(turbidityRaw, 2);

    Serial.print("Turbidity Voltage: ");
    Serial.print(turbidityVoltage, 3);
    Serial.println(" V");

    Serial.println("--------------------------------");

    Serial.print("TDS: ");
    Serial.print(tdsValue, 2);
    Serial.println(isCalibrated(tdsCal) ? " ppm (CALIBRATED)" : " ppm (UNCALIBRATED ESTIMATE)");

    Serial.print("TDS Raw: ");
    Serial.println(tdsRaw, 2);

    Serial.print("TDS Voltage: ");
    Serial.print(tdsVoltage, 3);
    Serial.println(" V");

    Serial.println("--------------------------------");

    // DATA FORMAT FOR PYTHON / ML MODEL
    Serial.print("DATA:");

    Serial.print(pH, 2);
    Serial.print(",");

    if (temperatureConnected)
        Serial.print(temperatureC, 2);
    else
        Serial.print("0");

    Serial.print(",");
    Serial.print(turbidityNTU, 2);
    Serial.print(",");
    Serial.println(tdsValue, 2);

    // RAW ADC DIAGNOSTIC FOR THE DASHBOARD (backend/serial_reader.py)
    Serial.print("RAW:");
    Serial.print((int)round(phRaw));
    Serial.print(",");
    Serial.print((int)round(turbidityRaw));
    Serial.print(",");
    Serial.println((int)round(tdsRaw));

    Serial.println("================================");

    // Poll for CAL commands during the wait instead of blocking for a
    // flat 5s, so a calibration click from the dashboard isn't delayed.
    unsigned long delayStart = millis();
    while (millis() - delayStart < 5000)
    {
        handleSerialCommands();
        delay(50);
    }
}
