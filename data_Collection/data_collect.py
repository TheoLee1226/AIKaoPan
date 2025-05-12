import control_meter as cm
import control_arduino as ca
import time
import numpy as np
import matplotlib.pyplot as plt
import os

voltage_meter = cm.control_meter()
voltage_meter.connect("ASRL11::INSTR")
voltage_meter.set_voltage_mode()

time.sleep(1)

current_meter = cm.control_meter()
current_meter.connect("ASRL4::INSTR")
current_meter.set_current_mode()

time.sleep(1)

arduino = ca.control_arduino("COM3", 9600)

time.sleep(5)

data = []

start_time = time.time()
collection_time = 10

PWM_power = 254

print("Starting data collection...")
while(time.time() - start_time < collection_time):
    current_time = time.time()
    arduino.control_arduino(PWM_power)
    voltage = voltage_meter.read_voltage()
    current = current_meter.read_current()
    temperature = arduino.return_temperature()
    data.append([current_time - start_time,voltage[0], current[0], temperature[0], temperature[1], temperature[2], temperature[3], temperature[4]])
    print(f"Time: {current_time - start_time:.2f} s, Voltage: [{voltage[0]:.3f}], Current: [{current[0]:.3f}], Temperature: {temperature}")
    time.sleep(0.01)
print("Data collection finished.")

print("Closing connections...")
voltage_meter.close()
current_meter.close()
arduino.close()

data_np = np.array(data)
steak = "ribeye"
current = data_np[0,2]
voltage = data_np[0,1]
power = current * voltage
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"data_collection\data\{steak}_{collection_time}S_{power:.2f}W_{timestamp}.csv"
directory = os.path.dirname(filename)
if not os.path.exists(directory):
    os.makedirs(directory)
np.savetxt(filename, data, delimiter=",",header="Time,voltage,current,TH, Ttest, TM_1, TM_2, TM_3", comments="", fmt="%.5f")
print(f"Data saved to {filename}")

data_np = np.array(data)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data_np[:, 0], data_np[:, 1], label="Voltage (V)")
plt.plot(data_np[:, 0], data_np[:, 2] * 10, label="Current (A/10)")
plt.plot(data_np[:, 0], np.multiply(data_np[:, 1], data_np[:, 2]), label="Power (W)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(data_np[:, 0], data_np[:, 3], label="TH (°C)")
plt.plot(data_np[:, 0], data_np[:, 4], label="Ttest (°C)")
plt.plot(data_np[:, 0], data_np[:, 5], label="TM_1 (°C)")
plt.plot(data_np[:, 0], data_np[:, 6], label="TM_2 (°C)")
plt.plot(data_np[:, 0], data_np[:, 7], label="TM_3 (°C)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid()

plt.show()





