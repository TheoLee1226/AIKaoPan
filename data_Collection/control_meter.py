import pyvisa

class control_meter:
    def __init__(self, resource_string):
        self.rm = pyvisa.ResourceManager()
        print("Available VISA resources:")
        print(self.rm.list_resources())
        print(f"Attempting to connect to: {resource_string}")
        self.instrument = self.rm.open_resource(resource_string)
        self.instrument.timeout = 30000 

    def close(self):
        self.instrument.close()
        self.rm.close()

    def read_voltage(self):
        self.instrument.write("FUNCtion 'VOLT:DC'")
        measured_value_str = self.instrument.query("READ?") 
        return float(measured_value_str) 
    
    def read_current(self):
        self.instrument.write("FUNCtion 'CURRent:DC'")
        measured_value_str = self.instrument.query("READ?")
        return float(measured_value_str)
    