import pyvisa

class control_meter:
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        print("Available VISA resources:")
        print(self.rm.list_resources())

    def connect(self, resource_string):
        print(f"Attempting to connect to: {resource_string}")
        self.instrument = self.rm.open_resource(resource_string)
        self.instrument.timeout = 5000
        print(f"Connected to: {self.instrument.query('*IDN?')}")
        self.instrument.clear()
        

    def close(self):
        try:
            if hasattr(self, 'instrument') and self.instrument:
                self.instrument.close()
        except Exception as e:
            print(f"Error closing VISA instrument: {e}")
        finally:
            self.instrument = None # Ensure it's cleared

        try:
            if hasattr(self, 'rm') and self.rm:
                self.rm.close()
        except Exception as e:
            print(f"Error closing VISA resource manager: {e}")
        finally:
            self.rm = None # Ensure it's cleared
        print("Connection closed (control_meter).")


    def set_voltage_mode(self):
        self.instrument.write("CONFigure:VOLTage:DC 50") 

    def set_current_mode(self):
        self.instrument.write("CONFigure:CURRent:DC 5")

    def read_voltage(self):
        measured_value_str = self.instrument.query("MEASure:VOLTage:DC? 50")
        measured_value_str = measured_value_str.strip()
        measured_value_str = measured_value_str.split(",")
        measured_value = []
        for value_str in measured_value_str:
            try:
                measured_value.append(float(value_str))
            except ValueError as e:
                print(f"無法轉換 '{value_str}': {e}") 
        return measured_value
    
    def read_current(self):
        measured_value_str = self.instrument.query("MEASure:CURRent:DC? 5")
        measured_value_str = measured_value_str.strip()
        measured_value_str = measured_value_str.split(",")
        measured_value = []
        for value_str in measured_value_str:
            try:
                measured_value.append(float(value_str))
            except ValueError as e:
                print(f"無法轉換 '{value_str}': {e}") 
        return measured_value
    
if __name__ == "__main__":
    meter = control_meter()
    meter.connect("ASRL4::INSTR")
    meter.set_current_mode()
    for i in range(10):
        current = meter.read_current()
        print(f"Current: {current}")
    meter.close()

    