#include <Wire.h>
#include <Adafruit_MCP9600.h>

const int PIN_MOS1_GATE = 9;
const int PIN_MOS2_GATE = 10;
const int PIN_MOS3_GATE = 11;

Adafruit_MCP9600 mcp_1;
Adafruit_MCP9600 mcp_2;

unsigned long lastTempRead = 0;         // 上次讀取溫度的時間
const unsigned long TEMP_INTERVAL = 10; // 每 100ms 讀取一次溫度

void setup()
{

  pinMode(PIN_MOS1_GATE, OUTPUT);
  pinMode(PIN_MOS2_GATE, OUTPUT);
  pinMode(PIN_MOS3_GATE, OUTPUT);

  Serial.begin(9600);

  if (!mcp_1.begin(0x66))
  {
    Serial.print("Error_0x66");
    while (1)
      ;
  }

  if (!mcp_2.begin(0x64))
  {
    Serial.print("Error_0x63");
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
}

void loop()
{
  // **檢查 Serial 指令**
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

    // **確保 Serial 緩衝區不會堆積**
    while (Serial.available() > 0)
    {
      Serial.read();
    }
  }

  // **定時讀取溫度**
  unsigned long currentMillis = millis();
  if (currentMillis - lastTempRead >= TEMP_INTERVAL)
  {
    lastTempRead = currentMillis;

    float temperature_1 = mcp_1.readThermocouple();
    float temperature_2 = mcp_2.readThermocouple();
    if (!isnan(temperature_1) && !isnan(temperature_2))
    { // 確保 MCP9600 讀取成功
      Serial.print("TEMP:");
      Serial.print(temperature_1);
      Serial.print(",");
      Serial.println(temperature_2);
    }
    else
    {
      Serial.println("TEMP:ERROR"); // 讀取失敗時回應，確保 Python 不會掛掉
    }
  }
}
