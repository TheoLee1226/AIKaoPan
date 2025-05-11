#include <Wire.h>
#include <Adafruit_MCP9600.h>

const int PIN_MOS1_GATE = 9;
const int PIN_MOS2_GATE = 10;
const int PIN_MOS3_GATE = 11;

const int PIN_Voltage = A0;
const int PIN_Current = A1;

Adafruit_MCP9600 mcp_1;
Adafruit_MCP9600 mcp_2;
Adafruit_MCP9600 mcp_3;
Adafruit_MCP9600 mcp_4;
Adafruit_MCP9600 mcp_5;

unsigned long lastTempRead = 0;         // 上次讀取溫度的時間
const unsigned long TEMP_INTERVAL = 10; // 每 100ms 讀取一次溫度

void setup()
{

  pinMode(PIN_MOS1_GATE, OUTPUT);
  pinMode(PIN_MOS2_GATE, OUTPUT);
  pinMode(PIN_MOS3_GATE, OUTPUT);

  pinMode(PIN_Voltage, INPUT);
  pinMode(PIN_Current, INPUT);

  Serial.begin(9600);

  if (!mcp_1.begin(0x60))
  {
    Serial.print("Error_0x60");
    while (1)
      ;
  }

  if (!mcp_2.begin(0x61))
  {
    Serial.print("Error_0x61");
    while (1)
      ;
  }

  if (!mcp_3.begin(0x62))
  {
    Serial.print("Error_0x62");
    while (1)
      ;
  }

  if (!mcp_4.begin(0x63))
  {
    Serial.print("Error_0x63");
    while (1)
      ;
  }

  if (!mcp_5.begin(0x66))
  {
    Serial.print("Error_0x66");
    while (1)
      ;
  }

  mcp_1.setADCresolution(MCP9600_ADCRESOLUTION_18);
  mcp_1.setThermocoupleType(MCP9600_TYPE_K);
  mcp_1.setFilterCoefficient(3);
  mcp_1.enable(true);

  mcp_2.setADCresolution(MCP9600_ADCRESOLUTION_18);
  mcp_2.setThermocoupleType(MCP9600_TYPE_K);
  mcp_2.setFilterCoefficient(3);
  mcp_2.enable(true);

  mcp_3.setADCresolution(MCP9600_ADCRESOLUTION_18);
  mcp_3.setThermocoupleType(MCP9600_TYPE_K);
  mcp_3.setFilterCoefficient(3);
  mcp_3.enable(true);

  mcp_4.setADCresolution(MCP9600_ADCRESOLUTION_18);
  mcp_4.setThermocoupleType(MCP9600_TYPE_K);
  mcp_4.setFilterCoefficient(3);
  mcp_4.enable(true);

  mcp_5.setADCresolution(MCP9600_ADCRESOLUTION_18);
  mcp_5.setThermocoupleType(MCP9600_TYPE_K);
  mcp_5.setFilterCoefficient(3);
  mcp_5.enable(true);
}

void loop()
{
  while (Serial.available() > 0)
  {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.startsWith("PWM:"))
    {
      int pwmValue = input.substring(4).toInt();
      pwmValue = constrain(pwmValue, 0, 255);
      analogWrite(PIN_MOS1_GATE, pwmValue);
      analogWrite(PIN_MOS2_GATE, pwmValue);
      analogWrite(PIN_MOS3_GATE, pwmValue);
    }

    while (Serial.available() > 0)
    {
      Serial.read();
    }
  }

  unsigned long currentMillis = millis();
  // if (currentMillis - lastTempRead >= TEMP_INTERVAL)
  while (1)
  {
    lastTempRead = currentMillis;

    float temperature_1 = mcp_1.readThermocouple();
    float temperature_2 = mcp_2.readThermocouple();
    float temperature_3 = mcp_3.readThermocouple();
    float temperature_4 = mcp_4.readThermocouple();
    float temperature_5 = mcp_5.readThermocouple();

    double read_voltage = analogRead(PIN_Voltage);
    double read_current = analogRead(PIN_Current);

    double voltage = (read_voltage * 5.0) / 1023.0 * 5;
    double current = (read_current * 5.0) / 1023.0;

    voltage = round(voltage * 1000) / 1000;
    current = round(current * 1000) / 1000;

    if (!isnan(temperature_1) && !isnan(temperature_2) && !isnan(temperature_3) && !isnan(temperature_4) && !isnan(temperature_5))
    { // 確保 MCP9600 讀取成功

      Serial.print("DATA:");
      Serial.print(temperature_1);
      Serial.print(",");
      Serial.print(temperature_2);
      Serial.print(",");
      Serial.print(temperature_3);
      Serial.print(",");
      Serial.print(temperature_4);
      Serial.print(",");
      Serial.print(temperature_5);
      Serial.print(",");
      Serial.print(voltage);
      Serial.print(",");
      Serial.print(current);
      Serial.println();
    }
    else
    {
      Serial.println("TEMP:ERROR"); // 讀取失敗時回應，確保 Python 不會掛掉
    }
  }
}
