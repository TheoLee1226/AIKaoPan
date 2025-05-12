import control_meter as cm
import control_arduino as ca
import time
import numpy as np
import os

voltage_meter = cm.control_meter()
voltage_meter.connect("ASRL11::INSTR")
voltage_meter.set_voltage_mode()

time.sleep(1)

#current_meter = cm.control_meter()
#current_meter.connect("ASRL12::INSTR")

time.sleep(1)

arduino = ca.control_arduino("COM3", 9600)

time.sleep(5)

data = []

start_time = time.time()

for i in range(10):
    current_time = time.time()
    arduino.control_arduino(155)
    voltage = voltage_meter.read_voltage()
    #current = current_meter.read_current()
    temperature = arduino.return_temperature()
    data.append([current_time - start_time ,voltage[0], temperature[0], temperature[1], temperature[2], temperature[3], temperature[4]])
    #data.append([current_time - start_time,voltage[0], voltage[1], current[0], current[1], temperature[0], temperature[1], temperature[2], temperature[3], temperature[4]])
    print(f"Time: {current_time - start_time:.2f} s, Voltage: {voltage}, Temperature: {temperature}")
    time.sleep(0.01)

voltage_meter.close()
#current_meter.close()
arduino.close()

timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"data_Collection\data\output_{timestamp}.csv"
directory = os.path.dirname(filename)
if not os.path.exists(directory):
    os.makedirs(directory)
np.savetxt(filename, data, delimiter=",",header="Time,voltage,current,TH, Ttest, TM_1, TM_2, TM_3", comments="", fmt="%.5f")
print(f"Data saved to {filename}")




