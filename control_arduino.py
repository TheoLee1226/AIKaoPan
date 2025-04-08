import serial
import time 

"""
PC =======[PWM:{pwm value}]=======> Arduino
PC <=[TEMP:{tempature of sensor}]== Arduino
"""

class control_arduino:
    def __init__(self, com_port, baudrate):
        try:
            self.arduino = serial.Serial(com_port, baudrate)
            self.arduino.reset_input_buffer()
            print("Connected")
        except serial.SerialException as e:
            print("Serial exception:", str(e))
    
    def control_arduino_and_return_temp(self, control_pwm_value):
        """
        control_pwm_value: 0~255
        will send to arduino to modify the pwm output
        and it will return the tempature of the sensor
        """
        temp_value = -1
        try:
            # self.arduino.write("TEMP\n".encode())
            if self.arduino.in_waiting >= 0:
                data = self.arduino.readline().decode("utf-8").strip()
                print("data:" + str(data))
                if "TEMP:" in data:
                    try:
                        temp_value = float(data.split(":")[1].strip())
                    except ValueError as e:
                        print("value error:" + e)
                    pwm_value = control_pwm_value
                    self.arduino.write(f"PWM:{pwm_value}\n".encode("utf-8"))
                    current_time = time.time()
            return temp_value

        except Exception as e:
            print("Serial error:", e)

    def return_tempure(self):
        '''
        only return the tempature of the sensor
        '''
        temp_value = -1
        self.arduino.reset_input_buffer()
        try:
            # self.arduino.write("TEMP\n".encode())
            if self.arduino.in_waiting >= 0:
                data = self.arduino.readline().decode("utf-8").strip()
                print("data:" + str(data))
                if "TEMP:" in data:
                    try:
                        temp_value = float(data.split(":")[1].strip())
                    except ValueError as e:
                        print("value error:" + e)
            return temp_value
        except Exception as e:
            print("Serial error:", e)
    
    def close(self):
        self.arduino.close()
        print("Disconnected")

    def reset(self):
        self.arduino.reset_input_buffer()
        print("Reset")
            


if __name__ == "__main__":
    '''
    only for test the control_arduino class
    '''
    arduino = control_arduino("COM7", 9600)
    for i in range(100):
        print(arduino.control_arduino_and_return_temp(100))
        time.sleep(0.5)
    



