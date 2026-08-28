#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_BUS 2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup()
{
    Serial.begin(9600);

    sensors.begin();

    Serial.println("System Ready");
}

void loop()
{
    sensors.requestTemperatures();

    float temperature = sensors.getTempCByIndex(0);

    // Example values
    float ph = 7.4;
    float chlorine = 2.1;
    float turbidity = 0.6;
    float tds = 350;

    Serial.print(ph);
    Serial.print(",");

    Serial.print(temperature);
    Serial.print(",");

    Serial.print(chlorine);
    Serial.print(",");

    Serial.print(turbidity);
    Serial.print(",");

    Serial.println(tds);

    delay(2000);
}