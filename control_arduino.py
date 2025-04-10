import serial
import time 
import numpy as np
import threading

"""
PC =======[PWM:{pwm value}]=======> Arduino
PC <=[TEMP:{tempature of sensor}]== Arduino
"""

class control_arduino:
    def __init__(self, com_port, baudrate):
        self.latest_temp_1 = -1.0 
        self.latest_temp_2 = -1.0
        self.lock = threading.Lock() 
        self.running = False # 

        self.logging_time_start = time.time()

        self.max_tempature_history_length = 1000 # 記錄的長度
        self.teaprature_history = []

        try:
            self.arduino = serial.Serial(com_port, baudrate, timeout=100)
            self.arduino.reset_input_buffer()
            print("Connected")
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
                        if line.startswith("TEMP:"):
                            parts = line.split(':')
                            if len(parts) == 2 and "," in parts[1]:
                                temp_parts = parts[1].split(',')
                                if len(temp_parts) == 2:
                                    try:
                                        temp1 = float(temp_parts[0].strip())
                                        temp2 = float(temp_parts[1].strip())
                                        with self.lock:
                                            self.latest_temp_1 = temp1
                                            self.latest_temp_2 = temp2
                                            self._append_tempature_history(np.array([temp1, temp2]))
                                            
                                    except ValueError:
                                        print(f"ValueError parsing temps: {parts[1]}") 
                                        pass 
                                elif "ERROR" in parts[1]:
                                     with self.lock: #
                                         self.latest_temp_1 = -1
                                         self.latest_temp_2 = -1
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
    
    def _append_tempature_history(self, tempature):
        if len(self.teaprature_history) == 0:
            self.logging_time_start = time.time()
            return
        if len(self.teaprature_history) >= self.max_tempature_history_length:
            self.teaprature_history.pop(0)
        logging_time = time.time() - self.logging_time_start
        logging_data =  np.array([logging_time, tempature[0], tempature[1]])    
        self.teaprature_history.append(logging_data)
        
    def return_tempature_history(self):
        if len(self.teaprature_history) == 0:

            print("No tempature history")
            return np.array([-1.0, -1.0])
        else:
            return np.array(self.teaprature_history)
    
    def control_arduino_and_return_temp(self, control_pwm_value):
        """
        control_pwm_value: 0~255
        will send to arduino to modify the pwm output
        and it will return the tempature of the sensor
        return type: np.array([temp_1_return, temp_2_return])
        """
        if not self.arduino or not self.running:
            print("Arduino not connected or running.")
            return np.array([-1, -1])

        try:
            pwm_value = int(control_pwm_value)
            pwm_value = max(0, min(pwm_value, 255)) 
            command = f"PWM:{pwm_value}\n"
            self.arduino.write(command.encode("utf-8"))

            with self.lock:
                temp_1_return = self.latest_temp_1
                temp_2_return = self.latest_temp_2
            return np.array([temp_1_return, temp_2_return])
        
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            self.running = False
            return np.array([-1.0, -1.0])
        except Exception as e:
            print(f"Error sending PWM: {e}")
            return np.array([-1.0, -1.0])

    def return_tempure(self):
        '''
        only return the tempature of the sensor
        return type: np.array([temp_value_1, temp_value_2])
        '''
        if not self.running:
             print("Arduino not connected or thread stopped.")
             return np.array([-1.0, -1.0])

        with self.lock:
            current_temp_1 = self.latest_temp_1
            current_temp_2 = self.latest_temp_2
        return np.array([current_temp_1, current_temp_2])
           
    def control_arduino(self,  control_pwm_value):
        '''
        control the arduino by setting the pwm value
        '''
        if not self.arduino or not self.running:

            try:
                pwm_value = int(control_pwm_value)
                pwm_value = max(0, min(pwm_value, 255)) 
                command = f"PWM:{pwm_value}\n"
                self.arduino.write(command.encode("utf-8"))
            except serial.SerialException as e:
                print(f"Serial write error: {e}")
        
    def close(self):
        self.running = False 
        self.arduino.control_arduino(0)
        if hasattr(self, 'read_thread') and self.read_thread.is_alive():
             self.read_thread.join(timeout=1.0) 
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                print("Disconnected")
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
    arduino = control_arduino("COM7", 9600)
    for i in range(100):
        print(time.time())
        print(arduino.control_arduino_and_return_temp(100))
        time.sleep(0.1)

    temp = arduino.return_tempature_history()
    print(temp)

    arduino.close()
    print("End of program")
    print("Arduino closed")

    
    



