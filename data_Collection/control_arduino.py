import serial
import time 
import numpy as np
import threading
import matplotlib.pyplot as plt
"""
PC =======[PWM:{pwm value}]=======> Arduino
PC <=[TEMP:{tempature of sensor}]== Arduino
"""

class control_arduino:
    def __init__(self, com_port, baudrate):
        self.latest_temp_1 = -1.0 
        self.latest_temp_2 = -1.0
        self.latest_temp_3 = -1.0
        self.latest_temp_4 = -1.0
        self.latest_temp_5 = -1.0
        self.latest_voltage = -1.0
        self.latest_current = -1.0

        self.lock = threading.Lock() 
        self.running = False # 

        self.logging_time_start = time.time()

        self.max_data_history_length = 100000 # 記錄的長度
        self.data_history = []

        try:
            print("Trying to connect arduino")
            self.arduino = serial.Serial(com_port, baudrate, timeout=1) # 縮短讀取超時
            self.arduino.reset_input_buffer()
            print("Connected arduino")
            self.running = True
            self.read_thread = threading.Thread(target=self._read_serial, daemon=True)
            self.read_thread.start()
            print("Serial reading thread started.")
        except serial.SerialException as e:
            print(f"Serial exception: {e}")
            self.arduino = None
            self.running = False
    
    def _read_serial(self):
        while self.running and self.arduino:
            try:
                raw_data = self.arduino.readline()
                if raw_data:
                    try:
                        line = raw_data.decode('utf-8').strip()
                        if line.startswith("DATA:"):
                            parts = line.split(':')
                            if len(parts) == 2 and "," in parts[1]:
                                data_parts = parts[1].split(',')
                                if len(data_parts) == 7:
                                    try:
                                        temp1 = float(data_parts[0].strip())
                                        temp2 = float(data_parts[1].strip())
                                        temp3 = float(data_parts[2].strip())
                                        temp4 = float(data_parts[3].strip())
                                        temp5 = float(data_parts[4].strip())
                                        voltage = float(data_parts[5].strip())
                                        current = float(data_parts[6].strip())

                                        with self.lock:
                                            self.latest_temp_1 = temp1
                                            self.latest_temp_2 = temp2
                                            self.latest_temp_3 = temp3
                                            self.latest_temp_4 = temp4
                                            self.latest_temp_5 = temp5
                                            self.latest_voltage = voltage
                                            self.latest_current = current
                                            self._append_data_history(np.array([temp1, temp2, temp3, temp4, temp5, voltage, current]))
                                            
                                    except ValueError:
                                        print(f"ValueError parsing temps: {parts[1]}") 
                                        pass 

                                elif "ERROR" in parts[1]:
                                    with self.lock: 
                                        self.latest_temp_1 = -1
                                        self.latest_temp_2 = -1
                                        self.latest_temp_3 = -1
                                        self.latest_temp_4 = -1
                                        self.latest_temp_5 = -1
                                        self.latest_voltage = -1
                                        self.latest_current = -1
                                        print("Received TEMP:ERROR") 
                    except UnicodeDecodeError:
                        print("UnicodeDecodeError, skipping line.")
                        pass 
            except serial.SerialException as e:
                print(f"Serial error in read thread: {e}")
                self.running = False
                break
            except Exception as e:
                print(f"Unexpected error in read thread: {e}")
                time.sleep(0.1) 

        print("Serial reading thread finished.")
    
    def _append_data_history(self, data):
        if len(self.data_history) == 0:
            self.logging_time_start = time.time()
        if len(self.data_history) >= self.max_data_history_length:
            self.data_history.pop(0)
        logging_time = time.time() - self.logging_time_start
        logging_data =  np.array([logging_time, data[0], data[1], data[2], data[3], data[4], data[5], data[6]])    
        self.data_history.append(logging_data)
        
    def return_data_history(self):
        if len(self.data_history) == 0:
            print("No data history")
            return np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        else:
            return np.array(self.data_history)
    
    def control_arduino_and_return_data(self, control_pwm_value):
        """
        control_pwm_value: 0~255
        will send to arduino to modify the pwm output
        and it will return the tempature of the sensor
        return type: np.array([temp_1_return, temp_2_return])
        """
        if not self.arduino or not self.running:
            print("Arduino not connected or running.")
            return np.array([-1, -1, -1, -1, -1, -1, -1])

        try:
            pwm_value = int(control_pwm_value)
            pwm_value = max(0, min(pwm_value, 255)) 
            command = f"PWM:{pwm_value}\n"
            self.arduino.write(command.encode("utf-8"))

            with self.lock:
                temp_1_return = self.latest_temp_1
                temp_2_return = self.latest_temp_2
                temp_3_return = self.latest_temp_3
                temp_4_return = self.latest_temp_4
                temp_5_return = self.latest_temp_5
                voltage_return = self.latest_voltage
                current_return = self.latest_current
            return np.array([temp_1_return, temp_2_return, temp_3_return, temp_4_return, temp_5_return, voltage_return, current_return])
        
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            self.running = False
            return np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        except Exception as e:
            print(f"Error sending PWM: {e}")
            return np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    def return_temperature(self):
        '''
        only return the tempature of the sensor
        return type: np.array([current_temp_1, current_temp_2, current_temp_3, current_temp_4, current_temp_5])
        '''
        if not self.running:
             print("Arduino not connected or thread stopped.")
             return np.array([-1.0, -1.0, -1.0, -1.0, -1.0])

        with self.lock:
            current_temp_1 = self.latest_temp_1
            current_temp_2 = self.latest_temp_2
            current_temp_3 = self.latest_temp_3
            current_temp_4 = self.latest_temp_4
            current_temp_5 = self.latest_temp_5
        return np.array([current_temp_1, current_temp_2, current_temp_3, current_temp_4, current_temp_5])
    
    def return_voltage_and_current(self):
        '''
        only return the voltage of the sensor
        return type: np.array([temp_value_1, temp_value_2])
        '''
        if not self.running:
             print("Arduino not connected or thread stopped.")
             return np.array([-1.0, -1.0])
        with self.lock:
            current_voltage = self.latest_voltage
            current_current = self.latest_current
        return np.array([current_voltage, current_current])
    
           
    def control_arduino(self, control_pwm_value):
        '''
        control the arduino by setting the pwm value
        '''
        if not self.arduino or not self.running:
            print("Arduino not connected or not running. Cannot send PWM command.")
            return
        try:
            pwm_value = int(control_pwm_value)
            pwm_value = max(0, min(pwm_value, 255))
            command = f"PWM:{pwm_value}\n"
            self.arduino.write(command.encode("utf-8"))
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while sending PWM: {e}")
        
    def close(self):
        self.control_arduino(0)
        self.running = False 
        if hasattr(self, 'read_thread') and self.read_thread.is_alive():
             self.read_thread.join(timeout=1.0) 
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                print("Disconnected arduino")
            except Exception as e:
                print(f"Error closing serial port: {e}")
        self.arduino = None

    def reset(self):
        if self.arduino and self.arduino.is_open:
             try:
                 self.arduino.reset_input_buffer()
                 print("Input buffer reset")
             except Exception as e:
                 print(f"Error resetting buffer: {e}")
        else:
             print("Cannot reset, Arduino not connected.")
            
if __name__ == "__main__":
    '''
    only for test the control_arduino class
    '''
    arduino = control_arduino("COM3", 9600)

    time.sleep(1)

    for i in range(100):
        #arduino.control_arduino_and_return_data(155)
        arduino.control_arduino(155)
        print(arduino.return_temperature())
        time.sleep(0.1)
