import control_meter as cm
import control_arduino as ca
import time
import numpy as np
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

print("Starting data collection...")
while(time.time() - start_time < collection_time):
    current_time = time.time()
    arduino.control_arduino(200)
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


steak = "ribeye"
power = voltage[0] * current[0]
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"data_collection\data\{steak}_{collection_time}S_{power:.2f}W_{timestamp}.csv"
directory = os.path.dirname(filename)
if not os.path.exists(directory):
    os.makedirs(directory)
np.savetxt(filename, data, delimiter=",",header="Time,voltage,current,TH, Ttest, TM_1, TM_2, TM_3", comments="", fmt="%.5f")
print(f"Data saved to {filename}")




