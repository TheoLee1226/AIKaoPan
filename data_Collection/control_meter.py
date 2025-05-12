import pyvisa

class control_meter:
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        print("Available VISA resources:")
        print(self.rm.list_resources())

    def connect(self, resource_string):
        print(f"Attempting to connect to: {resource_string}")
        self.instrument = self.rm.open_resource(resource_string)
        self.instrument.timeout = 30000
        print(f"Connected to: {self.instrument.query('*IDN?')}")
        self.instrument.clear()
        

    def close(self):
        self.instrument.close()
        self.rm.close()

    def set_voltage_mode(self):
        self.instrument.write("CONFigure:VOLTage:DC 50") 

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
        self.instrument.write("FUNCtion 'CURRent:DC'")
        measured_value_str = self.instrument.query("READ?") 
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
    meter.connect("ASRL11::INSTR")
    meter.set_voltage_mode()
    print(meter.read_voltage())
    print(meter.read_voltage())
    print(meter.read_voltage())
    meter.close()

    