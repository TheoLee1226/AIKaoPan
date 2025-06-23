import tkinter as tk
from tkinter import messagebox, ttk # Added ttk for better Entry validation potentially
import time
import threading
import os
import numpy as np # Make sure numpy is imported
import matplotlib.pyplot as plt

import pyvisa # For specific VISA error handling
import control_meter as cm
import control_arduino as ca
import serial.tools.list_ports # For listing COM ports

# --- Mock/Debug Classes ---
class MockControlArduino:
    """A mock class for control_arduino for UI testing without hardware."""
    def __init__(self, ui_instance):
        self.ui = ui_instance # Reference to the UI to get inversion state
        self.running = True
        self.raw_pwm_sent = 0
        self.temps = np.array([25.0, 24.8, 25.2, 25.1, 24.9])
        self.last_temp_update_time = time.time()
        print("[MockArduino] Initialized.")

    def get_logical_pwm(self):
        if self.ui.invert_pwm_var.get():
            return 255 - self.raw_pwm_sent
        return self.raw_pwm_sent

    def control_arduino(self, pwm_value):
        self.raw_pwm_sent = int(pwm_value)

    def return_temperature(self):
        now = time.time()
        time_delta = now - self.last_temp_update_time
        
        logical_pwm = self.get_logical_pwm()
        heating_rate = (logical_pwm / 255.0) * 0.5 
        cooling_rate = 0.1
        
        temp_change = (heating_rate - cooling_rate) * time_delta
        
        self.temps += temp_change
        self.temps = np.clip(self.temps, 20, 300)
        self.last_temp_update_time = now
        
        return self.temps + np.random.normal(0, 0.05, 5)

    def close(self):
        self.running = False
        print("[MockArduino] Closed.")

class MockControlMeter:
    """A mock class for control_meter for UI testing without hardware."""
    def __init__(self, mock_arduino_instance):
        self.mock_arduino = mock_arduino_instance
        self.simulated_resistance = 20.0 # ohms
        print("[MockMeter] Initialized.")

    def connect(self, resource_string): print(f"[MockMeter] 'Connected' to {resource_string}")
    def close(self): print("[MockMeter] 'Closed'")
    def set_voltage_mode(self): pass
    def set_current_mode(self): pass

    def read_voltage(self):
        logical_pwm = self.mock_arduino.get_logical_pwm()
        v_supply = 30.0 - (logical_pwm / 255.0) * 1.5
        voltage = v_supply * (logical_pwm / 255.0)
        return [voltage + np.random.normal(0, 0.02)]

    def read_current(self):
        voltage = self.read_voltage()[0]
        current = voltage / self.simulated_resistance
        return [current + np.random.normal(0, 0.01)]

# --- End Mock/Debug Classes ---

# --- PID Controller Class ---
class SimplePID:
    def __init__(self, Kp, Ki, Kd, setpoint, sample_time, output_limits=(0, 255), integral_limits=(-250, 250)):
        print(f"[PID_INIT] Kp={Kp}, Ki={Ki}, Kd={Kd}, Setpoint={setpoint}, SampleTime={sample_time}")
        print(f"[PID_INIT] OutputLimits={output_limits}, IntegralLimits={integral_limits}")
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.sample_time = sample_time  # Control loop interval in seconds

        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits  # Anti-windup for integral

        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0
        self._last_input = 0.0 # For derivative on measurement

        self.output = 0.0 # Store last output
        self._last_output = 0.0 # Store last output for logging if needed
    def update(self, current_value):
        now = time.time()
        time_change = now - self._last_time

        # print(f"[PID_UPDATE] Called. current_value={current_value}, time_change={time_change:.4f}s, sample_time={self.sample_time}")

        if current_value is None: # Cannot compute if current_value is invalid
            print("[PID_UPDATE] current_value is None, returning last output.") # Added print
            return self.output # Return last known good output or 0

        if time_change >= self.sample_time:
            # print(f"[PID_UPDATE] Time to update. Last update was {time_change:.4f}s ago.")
            error = float(self.setpoint) - float(current_value)
            print(f"[PID_UPDATE] Setpoint={self.setpoint}, CurrentValue={current_value}, Error={error:.4f}")
            
            # Store values before calculation for logging
            integral_before_clamp = self._integral

            # Proportional term
            p_term = self.Kp * error
            print(f"[PID_UPDATE] P_Term (Kp={self.Kp} * Error={error:.4f}) = {p_term:.4f}") # Added print

            # Integral term (with anti-windup)
            integral_change = self.Ki * error * time_change # Use actual time_change
            self._integral += self.Ki * error * time_change # Use actual time_change
            print(f"[PID_UPDATE] Integral_Change (Ki={self.Ki} * Error={error:.4f} * TC={time_change:.4f}) = {integral_change:.4f}") # Added print
            print(f"[PID_UPDATE] Integral before clamp: {self._integral:.4f}") # Added print
            self._integral = max(self.integral_min, min(self._integral, self.integral_max))
            print(f"[PID_UPDATE] Integral after clamp ({self.integral_min}, {self.integral_max}): {self._integral:.4f}") # Added print
            i_term = self._integral
            print(f"[PID_UPDATE] I_Term = {i_term:.4f}") # Added print

            # Derivative term (on measurement to reduce derivative kick)
            input_change = float(current_value) - self._last_input
            print(f"[PID_UPDATE] InputChange (Current={current_value} - LastInput={self._last_input}) = {input_change:.4f}") # Added print
            d_term = 0.0
            if time_change > 0:
                d_term = -self.Kd * (input_change / time_change)
                print(f"[PID_UPDATE] D_Term (-Kd={-self.Kd} * InChg={input_change:.4f} / TC={time_change:.4f}) = {d_term:.4f}") # Added print
            else:
                print(f"[PID_UPDATE] D_Term = 0 (time_change <= 0)") # Added print
            self.output = p_term + i_term + d_term
            # print(f"[PID_UPDATE] Output before clamp (P={p_term:.4f} + I={i_term:.4f} + D={d_term:.4f}) = {self.output:.4f}")
            self.output = max(self.output_min, min(self.output, self.output_max))
            # print(f"[PID_UPDATE] Output after clamp ({self.output_min}, {self.output_max}): {self.output:.4f}")

            self._last_error = error
            self._last_time = now
            self._last_input = float(current_value)
            self._last_output = self.output # Store the output after clamping
            return self.output
        else: # Added else block for clarity
             print(f"[PID_UPDATE] Not time to update yet (time_change={time_change:.4f}s < sample_time={self.sample_time}). Returning last output: {self.output}") # Added print
        return self.output # Not time to update yet, return last output

    def set_tunings(self, Kp, Ki, Kd):
        print(f"[PID_SET_TUNINGS] Old: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
        if Kp is not None: self.Kp = float(Kp)
        if Ki is not None: self.Ki = float(Ki)
        if Kd is not None: self.Kd = float(Kd)
        print(f"[PID_SET_TUNINGS] New: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")

    def set_setpoint(self, setpoint):
        new_setpoint_val = float(setpoint)
        # Removed logic that reset integral when setpoint was lowered.
        self.setpoint = new_setpoint_val
        print(f"[PID_SET_SETPOINT] Setpoint set to: {self.setpoint}")

    def reset(self):
        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0
        self._last_input = 0.0
        self.output = 0.0
        print(f"[PID_RESET] PID has been reset. Setpoint: {self.setpoint}, Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}")

# --- End PID Controller Class ---
class DataCollectionUI:
    # --- Constants for Device Configuration ---
    DEFAULT_VOLTAGE_METER_ADDRESS = "ASRL11::INSTR"
    DEFAULT_CURRENT_METER_ADDRESS = "ASRL4::INSTR"
    DEFAULT_ARDUINO_COM_PORT = "COM3"
    ARDUINO_BAUDRATE = 9600

    # --- Power Verification Constants ---
    MAX_ZEROING_ATTEMPTS = 3
    POWER_ZERO_THRESHOLD = 0.5  # Watts

    # --- Default UI Values ---
    DEFAULT_STEAK_TYPE = "test_steak"
    DEFAULT_PWM_PERCENTAGE = 0
    DEFAULT_PID_SETPOINT = 60.0 # Target temperature for preheat (Temperature PID - Unchanged)
    DEFAULT_PID_KP = 10.0
    DEFAULT_PID_KI = 0.1
    DEFAULT_PID_KD = 0.5
    DEFAULT_POWER_PID_SETPOINT = 20.0 # Target Watts (Power PID)
    DEFAULT_POWER_PID_KP = 3.0      # Slightly reduced Kp
    DEFAULT_POWER_PID_KI = 0.3     # Slightly reduced Ki
    DEFAULT_POWER_PID_KD = 0.1     # Significantly increased Kd to counteract undershoot
    DEFAULT_R_REF = 1.0 # Default reference resistance if file not loaded   
    DEFAULT_EQ_PARAM_A = 14.5341 # Example: R0 for R(T) = A * exp(B*T)
    DEFAULT_EQ_PARAM_B = 0.001999 # Example: temperature coefficient for copper/nichrome like
    DEFAULT_EQ_PARAM_C = 0.0 # Example: Constant offset for R(T) = A * exp(B*T) + C
    DEFAULT_RDS_ON = 0 # Default MOSFET Rds(on) in Ohms
    DEFAULT_NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM = 30 # Assumed V_out at PWM=255. Adjust as needed.
    DEFAULT_CORRECTED_POWER_TARGET = 20.0 # Default target power for Corrected Power mode
    def __init__(self, master):
        self.master = master
        master.title("Data Collection UI")
        master.geometry("1600x800") # Corrected geometry string
        master.resizable(False, False) # Make the window not resizable

        # --- Instance Variables ---
        # Device handles
        self.voltage_meter = None
        self.current_meter = None
        self.arduino = None

        # Data collection state
        self.data = [] # Stores collected [time, V, I, T1, T2, T3, T4, T5]
        self.recording = False # True if data recording is active
        self.flip_event_times = [] # Stores timestamps of flip events
        self.elapsed_time = 0 # Elapsed time during recording

        # Threading related
        self.timer_thread = None # Thread for UI timer during recording
        self.data_collection_thread = None # Thread for main data collection loop
        self.pre_recording_display_thread = None # Thread for live data display when not recording
        self.stop_pre_recording_display_event = threading.Event() # Event to stop pre-recording display

        # PID Preheat related
        self.preheat_pid_controller = None # PID controller instance for preheating
        self.preheating_active = False # True if PID preheating is active
        self.preheat_thread = None # Thread for PID preheat loop
        self.stop_preheat_event = threading.Event() # Event to stop preheat loop
        self.power_pid_controller = None # PID controller instance for power
        self.power_pid_active = False # True if PID power control is active
        self.power_pid_thread = None # Thread for PID power control loop
        self.stop_power_pid_event = threading.Event() # Event to stop power PID loop
        
        # Corrected Direct Power related (No PID controller needed for this mode)
        self.corrected_direct_power_active = False
        self.corrected_direct_power_thread = None
        self.stop_corrected_direct_power_event = threading.Event()
        # Removed T-R file specific variables
        
        self.equation_param_A_var = tk.DoubleVar(value=self.DEFAULT_EQ_PARAM_A)
        self.equation_param_B_var = tk.DoubleVar(value=self.DEFAULT_EQ_PARAM_B)
        self.equation_param_C_var = tk.DoubleVar(value=self.DEFAULT_EQ_PARAM_C)
        self.rds_on_var = tk.DoubleVar(value=self.DEFAULT_RDS_ON) # New: MOSFET Rds(on)
        self.nominal_supply_voltage_var = tk.DoubleVar(value=self.DEFAULT_NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM)
        self.corrected_power_target_var = tk.DoubleVar(value=self.DEFAULT_CORRECTED_POWER_TARGET)
        # UI Variables (Tkinter variables) for PID and other settings
        self.pid_sensor_var = tk.StringVar(value="TM_1") # Default PID sensor, options: "TM_1", "TH"
        self.pid_setpoint_var = tk.DoubleVar(value=self.DEFAULT_PID_SETPOINT)
        self.pid_kp_var = tk.DoubleVar(value=self.DEFAULT_PID_KP)
        self.pid_ki_var = tk.DoubleVar(value=self.DEFAULT_PID_KI)
        self.pid_kd_var = tk.DoubleVar(value=self.DEFAULT_PID_KD)
        self.power_pid_setpoint_var = tk.DoubleVar(value=self.DEFAULT_POWER_PID_SETPOINT)
        self.power_pid_kp_var = tk.DoubleVar(value=self.DEFAULT_POWER_PID_KP) 
        self.power_pid_ki_var = tk.DoubleVar(value=self.DEFAULT_POWER_PID_KI) 
        self.power_pid_kd_var = tk.DoubleVar(value=self.DEFAULT_POWER_PID_KD) 
        self.control_mode_var = tk.StringVar(value="MANUAL_PWM") # Modes: MANUAL_PWM, PID_PREHEAT, PID_POWER, CORRECTED_DIRECT_POWER 
        # Removed self.temp_resist_filename_var
        self.pwm_percentage_var = tk.IntVar(value=self.DEFAULT_PWM_PERCENTAGE)
        # Port/Address StringVars
        self.arduino_port_var = tk.StringVar(value=self.DEFAULT_ARDUINO_COM_PORT)
        self.voltage_meter_address_var = tk.StringVar(value=self.DEFAULT_VOLTAGE_METER_ADDRESS)
        self.current_meter_address_var = tk.StringVar(value=self.DEFAULT_CURRENT_METER_ADDRESS)

        self.steak_type_var = tk.StringVar(value=self.DEFAULT_STEAK_TYPE) # For PWM inversion
        self.invert_pwm_var = tk.BooleanVar(value=True) # For PWM inversion
        self.debug_mode_var = tk.BooleanVar(value=False) # For debug mode

        # Internal state for PWM
        self.current_pwm_setting_0_255 = self._calculate_pwm_actual(self.pwm_percentage_var.get()) # Actual PWM value (0-255) for Arduino
        self.last_active_pwm_0_255 = 0 # Stores the last PWM value (0-255) sent by any controller

        # Configuration
        self.pre_recording_display_interval = 1.0 # seconds for display update when not recording or preheating
        self.pid_control_interval = 0.2 # seconds for PID control loop
        self.sample_interval = 0.01 # seconds

        # --- Main Layout Frames ---
        # Status bar should be packed to master first if we want it at the very bottom,
        # and other content frames packed above it.
        # OR, pack content frames first, then status bar last to master.
        # Let's try packing status bar LAST to master.
        
        # Calculate fixed width for columns
        # Window width 800. Paddings: 10 (outer_left) + 10 (col1_col2) + 10 (col2_col3) + 10 (outer_right) = 40
        # Usable width for columns = 800 - 40 = 760
        # Width per column = 760 / 3 = 253
        column_width = (800 - 10 - 10 - 10 - 10) // 3 

        self.left_column_frame = ttk.Frame(master, width=column_width)
        self.left_column_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        self.left_column_frame.pack_propagate(False) # Prevent children from resizing this frame

        self.middle_column_frame = ttk.Frame(master, width=column_width)
        self.middle_column_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5), pady=10)
        self.middle_column_frame.pack_propagate(False)

        self.right_column_frame = ttk.Frame(master, width=column_width)
        self.right_column_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        self.right_column_frame.pack_propagate(False) # Prevent children from resizing this frame

        # --- Setup UI ---
        # Left Column
        self._setup_connection_controls()
        self._setup_realtime_display()
        # Middle Column
        self._setup_control_mode_selection() 
        self._setup_pwm_controls() 
        self._setup_pid_controls() 
        self._setup_resistance_equation_controls() 
        self._setup_power_pid_controls() 

        # Right Column
        self._setup_recording_controls()
        self._setup_steak_type_input()
        self._setup_action_controls()
        self._setup_timer_display()
        self._setup_emergency_stop_controls() # Moved to be part of the right column, after timer
        self._setup_status_bar() # Moved here, to be below emergency stop in the right column

        # Bottom (Spanning)
        # self._setup_status_bar() # Removed from here

        self.reset_realtime_display() # Initialize display fields
        master.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close event
        self._handle_control_mode_change() # Set initial UI state based on default mode
        self.update_status("Disconnected. Press 'Connect Devices'.")

    # --- UI Setup Helper Methods ---
    def _setup_connection_controls(self):
        """Sets up the device connection button."""
        # Connection Frame
        self.connection_frame = ttk.LabelFrame(self.left_column_frame, text="Device Connection & Ports")
        self.connection_frame.pack(pady=(0, 5), padx=5, fill="x")

        # --- Arduino Port Selection ---
        ttk.Label(self.connection_frame, text="Arduino Port:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        available_com_ports = [port.device for port in serial.tools.list_ports.comports()]
        if not available_com_ports:
            available_com_ports = [self.arduino_port_var.get()] # Show default if none found
        elif self.arduino_port_var.get() not in available_com_ports:
             # If default is not in list, add it to the list and make it the first item
             # Or, simply let it be; Combobox will show first available if default isn't there.
             # For now, let's ensure the default is an option if it was set.
             pass # The StringVar holds the default, Combobox will try to match.

        self.arduino_port_combobox = ttk.Combobox(self.connection_frame, textvariable=self.arduino_port_var,
                                                 values=available_com_ports, width=18, state='readonly')
        if self.arduino_port_var.get() not in available_com_ports and available_com_ports:
            # If the default port is not in the list of available_com_ports,
            # set the var to the first available port to avoid blank Combobox.
            # However, the StringVars are already initialized with defaults.
            # If the default isn't available, the combobox will show the first item from 'values'.
            # The user's default value in the StringVar will persist until they pick something.
            pass
        elif not available_com_ports: # No ports found at all
            self.arduino_port_combobox.set("No COM ports") # Placeholder
            self.arduino_port_combobox.config(values=["No COM ports"])

        self.arduino_port_combobox.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

        # --- VISA Resource (Meter) Selection ---
        available_visa_resources = []
        try:
            rm = pyvisa.ResourceManager()
            available_visa_resources = list(rm.list_resources())
            rm.close() # Close resource manager after listing
        except Exception as e:
            print(f"Error listing VISA resources: {e}")
            # Add default values to the list if VISA listing fails, so they appear in dropdown
            if self.DEFAULT_VOLTAGE_METER_ADDRESS not in available_visa_resources:
                available_visa_resources.append(self.DEFAULT_VOLTAGE_METER_ADDRESS)
            if self.DEFAULT_CURRENT_METER_ADDRESS not in available_visa_resources:
                available_visa_resources.append(self.DEFAULT_CURRENT_METER_ADDRESS)
            available_visa_resources = sorted(list(set(available_visa_resources))) # Ensure unique and sorted

        if not available_visa_resources: # If list is still empty
            available_visa_resources = ["No VISA resources"]

        # Arduino Port
        # Voltage Meter Address
        ttk.Label(self.connection_frame, text="Voltage Meter:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.voltage_meter_address_combobox = ttk.Combobox(self.connection_frame, textvariable=self.voltage_meter_address_var,
                                                          values=available_visa_resources, width=18, state='readonly')
        self.voltage_meter_address_combobox.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        # Current Meter Address
        ttk.Label(self.connection_frame, text="Current Meter:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.current_meter_address_combobox = ttk.Combobox(self.connection_frame, textvariable=self.current_meter_address_var,
                                                          values=available_visa_resources, width=18, state='readonly')
        self.current_meter_address_combobox.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        # Connect Button
        self.connect_button = tk.Button(self.connection_frame, text="Connect Devices", command=self.connect_devices)
        self.connect_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # Debug Mode Checkbox
        self.debug_mode_checkbox = ttk.Checkbutton(self.connection_frame, text="Debug Mode (No Hardware)", variable=self.debug_mode_var)
        self.debug_mode_checkbox.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        self.connection_frame.grid_columnconfigure(1, weight=1) # Allow entry fields to expand


    def _setup_realtime_display(self):
        """Sets up the labels for displaying real-time data."""
        # Real-time data display (Moved Here)
        self.realtime_data_frame = ttk.LabelFrame(self.left_column_frame, text="Real-time Data")
        self.realtime_data_frame.pack(pady=5, padx=5, fill="x")

        self.voltage_display_label = tk.Label(self.realtime_data_frame, text="Voltage: --- V")
        self.voltage_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.current_display_label = tk.Label(self.realtime_data_frame, text="Current: --- A")
        self.current_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.power_display_label = tk.Label(self.realtime_data_frame, text="Power: --- W") # Power display
        self.power_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.th_display_label = tk.Label(self.realtime_data_frame, text="TH: --- °C")
        self.th_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.ttest_display_label = tk.Label(self.realtime_data_frame, text="Ttest: --- °C")
        self.ttest_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.tm1_display_label = tk.Label(self.realtime_data_frame, text="TM_1: --- °C")
        self.tm1_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.tm2_display_label = tk.Label(self.realtime_data_frame, text="TM_2: --- °C")
        self.tm2_display_label.pack(anchor=tk.W, padx=5, pady=2)
        self.tm3_display_label = tk.Label(self.realtime_data_frame, text="TM_3: --- °C")
        self.tm3_display_label.pack(anchor=tk.W, padx=5, pady=2)
    
    def _setup_control_mode_selection(self):
        """Sets up the radio buttons for selecting the control mode."""
        self.control_mode_frame = ttk.LabelFrame(self.middle_column_frame, text="Live Control Mode")
        self.control_mode_frame.pack(pady=5, padx=5, fill="x")

        # Radio buttons will now be packed directly into control_mode_frame, each on a new line.

        self.manual_pwm_radio = ttk.Radiobutton(self.control_mode_frame, text="Manual PWM", variable=self.control_mode_var,
                                                 value="MANUAL_PWM", command=self._handle_control_mode_change, state=tk.DISABLED)
        self.manual_pwm_radio.pack(anchor=tk.W, padx=5, pady=2)

        self.pid_preheat_radio = ttk.Radiobutton(self.control_mode_frame, text="PID Preheat", variable=self.control_mode_var,
                                                  value="PID_PREHEAT", command=self._handle_control_mode_change, state=tk.DISABLED)
        self.pid_preheat_radio.pack(anchor=tk.W, padx=5, pady=2)

        self.corrected_direct_power_radio = ttk.Radiobutton(self.control_mode_frame, text="Corrected Power (Eqn)", variable=self.control_mode_var,
                                                        value="CORRECTED_DIRECT_POWER", command=self._handle_control_mode_change, state=tk.DISABLED)
        self.corrected_direct_power_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        self.pid_power_radio = ttk.Radiobutton(self.control_mode_frame, text="PID Power", variable=self.control_mode_var,
                                                value="PID_POWER", command=self._handle_control_mode_change, state=tk.DISABLED)
        self.pid_power_radio.pack(anchor=tk.W, padx=5, pady=2)
        
    def _setup_pwm_controls(self):
        """Sets up the manual PWM control elements."""
        self.pwm_frame = ttk.LabelFrame(self.middle_column_frame, text="Manual PWM Control")
        self.pwm_frame.pack(pady=5, padx=5, fill="x")

        # Row 0
        self.pwm_label_display = tk.Label(self.pwm_frame, text="PWM (%):")
        self.pwm_label_display.grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)

        self.pwm_entry = tk.Entry(self.pwm_frame, textvariable=self.pwm_percentage_var, width=5)
        self.pwm_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)
        
        self.pwm_range_label = tk.Label(self.pwm_frame, text="(0-100)")
        self.pwm_range_label.grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)

        self.set_pwm_button = ttk.Button(self.pwm_frame, text="Set PWM", command=self.set_pwm_from_entry, state=tk.DISABLED)
        self.set_pwm_button.grid(row=0, column=3, padx=5, pady=3, sticky="ew")

        self.set_pwm_zero_button = ttk.Button(self.pwm_frame, text="Set to 0%", command=self.set_pwm_to_zero, state=tk.DISABLED)
        self.set_pwm_zero_button.grid(row=0, column=4, padx=5, pady=3, sticky="ew") # Changed column to 4

        self.set_pwm_hundred_button = ttk.Button(self.pwm_frame, text="Set to 100%", command=self.set_pwm_to_hundred, state=tk.DISABLED)
        self.set_pwm_hundred_button.grid(row=0, column=5, padx=5, pady=3, sticky="ew") # New button in column 5

        # Row 1: Invert PWM Checkbox
        self.pwm_invert_checkbox = ttk.Checkbutton(self.pwm_frame, text="Invert PWM Output", variable=self.invert_pwm_var, state=tk.DISABLED)
        self.pwm_invert_checkbox.grid(row=1, column=0, columnspan=3, padx=5, pady=3, sticky=tk.W)


        # Configure column weights for pwm_frame grid to allow expansion if needed
        self.pwm_frame.grid_columnconfigure(0, weight=0) # Label column
        self.pwm_frame.grid_columnconfigure(1, weight=0) # Entry/Button column
        self.pwm_frame.grid_columnconfigure(2, weight=0) # Range Label/Button column
        self.pwm_frame.grid_columnconfigure(3, weight=1) # Button column
        self.pwm_frame.grid_columnconfigure(4, weight=1) # Button column (for Set to 0%)
        self.pwm_frame.grid_columnconfigure(4, weight=1) # Button column


    def _setup_pid_controls(self):
        """Sets up the PID preheat control elements."""
        # PID Preheat Control Frame
        self.preheat_frame = ttk.LabelFrame(self.middle_column_frame, text="PID Preheat Control (Temp)")
        self.preheat_frame.pack(pady=5, padx=5, fill="x")

        # Row 0: Sensor Selection
        ttk.Label(self.preheat_frame, text="PID Sensor:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        sensor_radio_container = ttk.Frame(self.preheat_frame) # Use a sub-frame for radio buttons
        sensor_radio_container.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)
        self.pid_sensor_tm1_radio = ttk.Radiobutton(sensor_radio_container, text="TM_1", variable=self.pid_sensor_var, value="TM_1", state=tk.DISABLED)
        self.pid_sensor_tm1_radio.pack(anchor=tk.W, pady=(0,2)) # Anchor West, add a bit of padding below
        self.pid_sensor_th_radio = ttk.Radiobutton(sensor_radio_container, text="TH", variable=self.pid_sensor_var, value="TH", state=tk.DISABLED)
        self.pid_sensor_th_radio.pack(anchor=tk.W) # Anchor West

        # Row 1: Target Temp
        ttk.Label(self.preheat_frame, text="Target Temp (°C):").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.pid_setpoint_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_setpoint_var, width=7)
        self.pid_setpoint_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 2: Kp
        ttk.Label(self.preheat_frame, text="Kp:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.pid_kp_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_kp_var, width=7)
        self.pid_kp_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 3: Ki
        ttk.Label(self.preheat_frame, text="Ki:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.pid_ki_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_ki_var, width=7)
        self.pid_ki_entry.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 4: Kd
        ttk.Label(self.preheat_frame, text="Kd:").grid(row=4, column=0, padx=5, pady=3, sticky=tk.W)
        self.pid_kd_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_kd_var, width=7)
        self.pid_kd_entry.grid(row=4, column=1, padx=5, pady=3, sticky=tk.W)

        # Column 2: Button and Status (spanning multiple rows on the left)
        self.preheat_button = ttk.Button(self.preheat_frame, text="Start Preheat", command=self.toggle_preheat, state=tk.DISABLED)
        self.preheat_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky="ew") 
        
        self.preheat_status_label = tk.Label(self.preheat_frame, text="Preheat Status: Off", wraplength=200, justify=tk.LEFT)
        self.preheat_status_label.grid(row=1, column=2, rowspan=4, padx=10, pady=2, sticky="nsew") # Starts at row 1, spans 4 rows

        # Configure column weights for preheat_frame grid
        self.preheat_frame.grid_columnconfigure(0, weight=0)
        self.preheat_frame.grid_columnconfigure(1, weight=0)
        self.preheat_frame.grid_columnconfigure(2, weight=1)
    
    def _setup_resistance_equation_controls(self):
        """Sets up controls for the resistance equation R(T) = A * exp(B*T)."""
        self.res_eq_frame = ttk.LabelFrame(self.middle_column_frame, text="Corrected Power (Eqn: R = A*exp(B*T)+C)") # Updated Label
        self.res_eq_frame.pack(pady=5, padx=5, fill="x")

        ttk.Label(self.res_eq_frame, text="Target Power (W):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.corrected_power_target_entry = ttk.Entry(self.res_eq_frame, textvariable=self.corrected_power_target_var, width=10, state=tk.DISABLED)
        self.corrected_power_target_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

        # Parameters A, B, C on subsequent rows
        ttk.Label(self.res_eq_frame, text="Param A:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.eq_A_entry = ttk.Entry(self.res_eq_frame, textvariable=self.equation_param_A_var, width=10, state=tk.DISABLED)
        self.eq_A_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)

        ttk.Label(self.res_eq_frame, text="Param B:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.eq_B_entry = ttk.Entry(self.res_eq_frame, textvariable=self.equation_param_B_var, width=10, state=tk.DISABLED)
        self.eq_B_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)

        ttk.Label(self.res_eq_frame, text="Param C:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.eq_C_entry = ttk.Entry(self.res_eq_frame, textvariable=self.equation_param_C_var, width=10, state=tk.DISABLED)
        self.eq_C_entry.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)
        
        # Row 4: MOSFET Rds(on)
        ttk.Label(self.res_eq_frame, text="MOSFET Rds(on) (Ω):").grid(row=4, column=0, padx=5, pady=3, sticky=tk.W)
        self.rds_on_entry = ttk.Entry(self.res_eq_frame, textvariable=self.rds_on_var, width=10, state=tk.DISABLED)
        self.rds_on_entry.grid(row=4, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 5: Nominal Supply Voltage at Max PWM
        ttk.Label(self.res_eq_frame, text="V_supply_max_pwm (V):").grid(row=5, column=0, padx=5, pady=3, sticky=tk.W)
        self.nominal_supply_voltage_entry = ttk.Entry(self.res_eq_frame, textvariable=self.nominal_supply_voltage_var, width=10, state=tk.DISABLED)
        self.nominal_supply_voltage_entry.grid(row=5, column=1, padx=5, pady=3, sticky=tk.W)

        # Button and Status Label to the right, similar to PID control blocks
        self.corrected_power_button = ttk.Button(self.res_eq_frame, text="Start Corr.Pwr", command=self.toggle_corrected_direct_power, state=tk.DISABLED)
        self.corrected_power_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky="ewns") # Span 1 row

        self.corrected_power_status_label = tk.Label(self.res_eq_frame, text="Corrected Pwr: Off", wraplength=220, justify=tk.LEFT) # Increased wraplength
        self.corrected_power_status_label.grid(row=1, column=2, rowspan=5, padx=10, pady=2, sticky="nsew") # Starts at row 1, spans 5 rows
        self.res_eq_frame.grid_columnconfigure(2, weight=1)
        

    def _setup_power_pid_controls(self):
        """Sets up the PID power control elements."""
        self.power_pid_frame = ttk.LabelFrame(self.middle_column_frame, text="PID Power Control (Watts)")
        self.power_pid_frame.pack(pady=(5,0), padx=5, fill="x") # Reduced bottom padding

        # Row 0: Target Power
        ttk.Label(self.power_pid_frame, text="Target Power (W):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_setpoint_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_setpoint_var, width=7)
        self.power_pid_setpoint_entry.grid(row=0, column=1, padx=5, pady=(3,0), sticky=tk.W) # Reduced bottom padding

        # Row 1: Kp
        ttk.Label(self.power_pid_frame, text="Kp:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_kp_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_kp_var, width=7)
        self.power_pid_kp_entry.grid(row=1, column=1, padx=5, pady=(3,0), sticky=tk.W) # Reduced bottom padding

        # Row 2: Ki
        ttk.Label(self.power_pid_frame, text="Ki:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_ki_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_ki_var, width=7)
        self.power_pid_ki_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 3: Kd
        ttk.Label(self.power_pid_frame, text="Kd:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_kd_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_kd_var, width=7)
        self.power_pid_kd_entry.grid(row=3, column=1, padx=5, pady=(3,0), sticky=tk.W) # Reduced bottom padding

        # Column 2: Button and Status
        self.power_pid_button = ttk.Button(self.power_pid_frame, text="Start Power Ctrl", command=self.toggle_power_pid, state=tk.DISABLED)
        self.power_pid_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky="ewns") 
        
        self.power_pid_status_label = tk.Label(self.power_pid_frame, text="Power PID Status: Off", wraplength=200, justify=tk.LEFT)
        self.power_pid_status_label.grid(row=1, column=2, rowspan=3, padx=10, pady=(2,0), sticky="nsew") # Starts at row 1, spans 3 rows, reduced bottom padding

        self.power_pid_frame.grid_columnconfigure(0, weight=0)
        self.power_pid_frame.grid_columnconfigure(1, weight=0)
        self.power_pid_frame.grid_columnconfigure(2, weight=1)

    def _setup_recording_controls(self):
        """Sets up the Start/Stop recording buttons."""
        # Recording Control Frame
        self.recording_control_frame = ttk.LabelFrame(self.right_column_frame, text="Recording Control")
        self.recording_control_frame.pack(pady=5, padx=5, fill="x")
        self.start_button = ttk.Button(self.recording_control_frame, text="Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, pady=5, padx=5, expand=True, fill="x")
        self.stop_button = ttk.Button(self.recording_control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, pady=5, padx=5, expand=True, fill="x")
    
    def _setup_steak_type_input(self):
        """Sets up the input field for steak type."""
        # Steak Type Input (Moved Here)
        self.steak_type_frame = ttk.LabelFrame(self.right_column_frame, text="Steak Type")
        self.steak_type_frame.pack(pady=5, padx=5, fill="x")

        self.steak_type_label = tk.Label(self.steak_type_frame, text="Type:")
        self.steak_type_label.pack(side=tk.LEFT, padx=5, pady=3)
        self.steak_type_entry = ttk.Entry(self.steak_type_frame, textvariable=self.steak_type_var)
        self.steak_type_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=3)
        
    def _setup_action_controls(self):
        """Sets up action buttons like 'Log Flip Event'."""
        # Action Frame for Flip Button
        self.action_frame = ttk.LabelFrame(self.right_column_frame, text="Actions")
        self.action_frame.pack(pady=5, padx=5, fill="x")
        self.flip_button = ttk.Button(self.action_frame, text="Log Flip Event", command=self.log_flip_event, state=tk.DISABLED)
        self.flip_button.pack(pady=5, padx=5, fill="x", expand=True)

    def _setup_timer_display(self):
        """Sets up the label for displaying elapsed recording time."""
        # Timer Display Frame
        self.timer_frame = ttk.LabelFrame(self.right_column_frame, text="Elapsed Time")
        self.timer_frame.pack(pady=5, padx=5, fill="x")
        self.time_label = tk.Label(self.timer_frame, text="00:00:00", font=("Helvetica", 32)) # Large font for time
        self.time_label.pack(pady=5, padx=5)

    def _setup_status_bar(self):
        """Sets up the status bar at the bottom of the UI."""
        # Parent is now self.right_column_frame, to be placed below other elements in this frame.
        self.status_label = tk.Label(self.right_column_frame, text="Status: Disconnected",
                                     relief=tk.SUNKEN, anchor=tk.W,
                                     wraplength=230,  # Adjusted for narrower column
                                     justify=tk.LEFT)
        # Pack it at the top of the remaining space in right_column_frame, below emergency_stop
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 10))
        
    def _setup_emergency_stop_controls(self):
        """Sets up the Emergency Stop button."""
        self.emergency_stop_frame = ttk.Frame(self.right_column_frame) # Parent is now the right_column_frame
        self.emergency_stop_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(10,5)) # Pack below timer in right column
        self.emergency_stop_button = tk.Button(self.emergency_stop_frame, text="EMERGENCY STOP", command=self.emergency_stop,
                                               bg="red", fg="white", font=("Helvetica", 10, "bold"), state=tk.DISABLED)
        self.emergency_stop_button.pack(fill=tk.X, expand=True, ipady=5)

    # --- Control Mode Logic ---
    def _handle_control_mode_change(self, event=None): # For Live Controls
        """Handles switching between control modes and updates UI accordingly."""
        selected_mode = self.control_mode_var.get()
        self.update_status(f"Control mode changed to: {selected_mode}")

        # Stop any active controllers if switching away from them.
        # Pass set_pwm_to_zero=False to allow for a seamless handover of PWM control
        # without turning the heater off momentarily.
        if selected_mode != "PID_PREHEAT" and self.preheating_active:
            self.stop_preheat(set_pwm_to_zero=False)
        if selected_mode != "PID_POWER" and self.power_pid_active:
            self.stop_power_pid(set_pwm_to_zero=False)
        if selected_mode != "CORRECTED_DIRECT_POWER" and self.corrected_direct_power_active:
            self.stop_corrected_direct_power(set_pwm_to_zero=False)

        # --- Enable/Disable Manual PWM Controls ---
        is_manual_mode = (selected_mode == "MANUAL_PWM")
        arduino_connected = self.arduino and self.arduino.running
        manual_pwm_button_state = tk.NORMAL if is_manual_mode and arduino_connected else tk.DISABLED
        # PWM entry is always enabled if connected, buttons depend on mode
        pwm_general_controls_state = tk.NORMAL if arduino_connected else tk.DISABLED
        self.pwm_entry.config(state=pwm_general_controls_state)
        self.set_pwm_button.config(state=manual_pwm_button_state)
        self.set_pwm_zero_button.config(state=manual_pwm_button_state)
        self.set_pwm_hundred_button.config(state=manual_pwm_button_state) # New: control state of "Set to 100%" button
        if hasattr(self, 'pwm_invert_checkbox'): # Ensure checkbox exists
            self.pwm_invert_checkbox.config(state=pwm_general_controls_state)

        if is_manual_mode and self.arduino and self.arduino.running:
            # When switching TO manual mode, update the entry box with the last known PWM value.
            # The handover is seamless because the stop_... functions were called with set_pwm_to_zero=False.
            last_pwm_percentage = round(self.last_active_pwm_0_255 / 255.0 * 100)
            self.pwm_percentage_var.set(last_pwm_percentage)
            self.current_pwm_setting_0_255 = self.last_active_pwm_0_255 # Also update the internal state for manual mode

        # --- Enable/Disable PID Preheat Controls ---
        is_preheat_mode = (selected_mode == "PID_PREHEAT")
        preheat_button_state = tk.NORMAL if is_preheat_mode and self.arduino and self.arduino.running else tk.DISABLED
        # PID entry fields are always enabled if connected, button depends on mode
        pid_entry_state = tk.NORMAL if self.arduino and self.arduino.running else tk.DISABLED
        self.pid_sensor_tm1_radio.config(state=pid_entry_state) # Sensor selection tied to entry state
        self.pid_sensor_th_radio.config(state=pid_entry_state)   # Sensor selection tied to entry state
        self.pid_setpoint_entry.config(state=pid_entry_state)
        self.pid_kp_entry.config(state=pid_entry_state)
        self.pid_ki_entry.config(state=pid_entry_state)
        self.pid_kd_entry.config(state=pid_entry_state)
        self.preheat_button.config(state=preheat_button_state)
        if not is_preheat_mode: # If not in preheat mode, ensure button text is "Start Preheat"
            self.preheat_button.config(text="Start Preheat")
            self.preheat_status_label.config(text="Preheat Status: Off")

        # --- Enable/Disable PID Power Controls ---
        is_power_mode = (selected_mode == "PID_POWER")
        power_button_state = tk.NORMAL if is_power_mode and self.arduino and self.arduino.running else tk.DISABLED
        self.power_pid_setpoint_entry.config(state=pid_entry_state) # Uses same pid_entry_state
        self.power_pid_kp_entry.config(state=pid_entry_state)
        self.power_pid_ki_entry.config(state=pid_entry_state)
        self.power_pid_kd_entry.config(state=pid_entry_state)
        self.power_pid_button.config(state=power_button_state)
        if not is_power_mode: # If not in power mode, ensure button text is "Start Power Ctrl"
            self.power_pid_button.config(text="Start Power Ctrl")
            self.power_pid_status_label.config(text="Power PID Status: Off")

        # --- Enable/Disable Corrected PID Power Controls ---
        is_corrected_direct_mode = (selected_mode == "CORRECTED_DIRECT_POWER")
        # Corrected Direct Power uses the Setpoint entry from Power PID. Kp, Ki, Kd are ignored.
        # Its button state depends on mode AND if the T-R file is loaded
        # The power_pid_button is reused for this mode.
        self.corrected_power_target_entry.config(state=pid_entry_state) # New target power entry
        # Kp, Ki, Kd entries from power_pid_frame remain enabled by pid_entry_state but are not used by this mode.
        # Equation parameter entries are controlled by pid_entry_state as well.
        self.eq_A_entry.config(state=pid_entry_state)
        self.eq_B_entry.config(state=pid_entry_state)
        self.eq_C_entry.config(state=pid_entry_state) # Manage state for Param C entry
        self.rds_on_entry.config(state=pid_entry_state) # Manage state for Rds(on) entry
        self.nominal_supply_voltage_entry.config(state=pid_entry_state) # Manage state for V_supply_max_pwm entry
        if is_corrected_direct_mode:
            self.corrected_power_button.config(state=tk.NORMAL if self.arduino and self.arduino.running else tk.DISABLED)
            if not self.corrected_direct_power_active: # If not active, ensure button text and status are reset
                self.corrected_power_button.config(text="Start Corr.Pwr")
                self.corrected_power_status_label.config(text="Corrected Pwr: Off")
        else: # If not in corrected power mode, disable its button and reset text/status
            self.corrected_power_button.config(state=tk.DISABLED, text="Start Corr.Pwr")
            self.corrected_power_status_label.config(text="Corrected Pwr: Off")

        # If switching to Manual PWM and Arduino is connected, ensure PWM is set (e.g. to 0 or last value)
        # For now, rely on PID stop setting PWM to 0, or user explicitly setting manual PWM.
        if is_manual_mode and self.arduino and self.arduino.running:
             # Optionally, apply current pwm_percentage_var or set to 0.
             # self.arduino.control_arduino(self.current_pwm_setting_0_255) # Applies last manual or 0 if PID was stopped
             pass

    # Removed Temperature-Resistance File Handling methods (load_temp_resistance_file, get_resistance_at_temp)

    # --- PWM Inversion Helper ---
    def _apply_pwm_inversion(self, pwm_value_0_255):
        if self.invert_pwm_var.get():
            return 255 - pwm_value_0_255
        return pwm_value_0_255

    # --- Core Functionality Methods ---
    def _calculate_pwm_actual(self, percentage):
        """Converts percentage (0-100) to actual PWM value (0-255)."""
        return int(round(percentage / 100.0 * 255))

    def log_flip_event(self):
        if self.recording:
            current_flip_time = self.elapsed_time
            self.flip_event_times.append(current_flip_time)
            self.update_status(f"Flip event logged at {current_flip_time}s.")

    def set_pwm_from_entry(self):
        """
        Sets the PWM value based on the input from the PWM entry box.
        Validates the input and sends the command to Arduino if connected.
        """
        try:
            percentage = self.pwm_percentage_var.get()
            if not (0 <= percentage <= 100):
                messagebox.showerror("PWM Error", "PWM percentage must be between 0 and 100.")
                return
        except tk.TclError:
            messagebox.showerror("PWM Error", "Invalid PWM percentage. Please enter a number.")
            return

        # Stop other active controllers before setting manual PWM
        if self.preheating_active: self.stop_preheat()
        if self.power_pid_active: self.stop_power_pid()
        if self.corrected_direct_power_active: self.stop_corrected_direct_power()

        if self.control_mode_var.get() != "MANUAL_PWM":
            self.control_mode_var.set("MANUAL_PWM")
            self._handle_control_mode_change()

        if self.arduino and self.arduino.running:
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(percentage)
            self.last_active_pwm_0_255 = self.current_pwm_setting_0_255 # Update last active PWM
            self.pwm_label_display.config(text=f"PWM: {percentage}%") # Update label
            final_pwm_to_send = self._apply_pwm_inversion(self.current_pwm_setting_0_255)
            self.arduino.control_arduino(final_pwm_to_send)
            status_msg = f"PWM set to {self.pwm_percentage_var.get()}% ({self.current_pwm_setting_0_255}). "
            status_msg += f"Sent to Arduino: {final_pwm_to_send}" if self.invert_pwm_var.get() else f"Sent to Arduino: {self.current_pwm_setting_0_255}"
            self.update_status(status_msg)
        else:
            messagebox.showwarning("PWM Error", "Arduino not connected. Cannot send PWM command.")

    def set_pwm_to_zero(self):
        """
        Sets the PWM value to 0 and sends the command to Arduino.
        Also stops PID preheat if it's active.
        """
        # Stop other active controllers before setting manual PWM to zero
        if self.preheating_active: self.stop_preheat()
        if self.power_pid_active: self.stop_power_pid()
        if self.corrected_direct_power_active: self.stop_corrected_direct_power()

        if self.control_mode_var.get() != "MANUAL_PWM":
            self.control_mode_var.set("MANUAL_PWM")
            self._handle_control_mode_change()

        self.pwm_percentage_var.set(0) # Update the Tkinter variable
        self.last_active_pwm_0_255 = 0 # Update last active PWM

        if self.arduino and self.arduino.running:
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(0)
            self.last_active_pwm_0_255 = self.current_pwm_setting_0_255
            self.pwm_label_display.config(text=f"PWM: 0%") # Update label
            final_pwm_to_send = self._apply_pwm_inversion(self.current_pwm_setting_0_255)
            self.arduino.control_arduino(final_pwm_to_send) # Send 0 (or 255 if inverted) to Arduino
            status_msg = f"PWM set to 0% (0). "
            status_msg += f"Sent to Arduino: {final_pwm_to_send}" if self.invert_pwm_var.get() else f"Sent to Arduino: {self.current_pwm_setting_0_255}"
            self.update_status(status_msg)
        else:
            # Still update local state even if not connected, so UI reflects 0
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(0)
            self.last_active_pwm_0_255 = self.current_pwm_setting_0_255
            self.pwm_label_display.config(text=f"PWM: 0%")
            self.update_status("PWM set to 0% (0) (Arduino not connected).")

    def set_pwm_to_hundred(self):
        """
        Sets the PWM value to 100% and sends the command to Arduino.
        Also stops any active PID controllers.
        """
        # Stop other active controllers before setting manual PWM to 100
        if self.preheating_active: self.stop_preheat()
        if self.power_pid_active: self.stop_power_pid()
        if self.corrected_direct_power_active: self.stop_corrected_direct_power()

        if self.control_mode_var.get() != "MANUAL_PWM":
            self.control_mode_var.set("MANUAL_PWM")
            self._handle_control_mode_change()

        self.pwm_percentage_var.set(100) # Update the Tkinter variable
        self.last_active_pwm_0_255 = self._calculate_pwm_actual(100) # Update last active PWM

        if self.arduino and self.arduino.running:
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(100)
            self.last_active_pwm_0_255 = self.current_pwm_setting_0_255
            self.pwm_label_display.config(text=f"PWM: 100%") # Update label
            final_pwm_to_send = self._apply_pwm_inversion(self.current_pwm_setting_0_255)
            self.arduino.control_arduino(final_pwm_to_send) # Send 255 (or 0 if inverted) to Arduino
            status_msg = f"PWM set to 100% (255). "
            status_msg += f"Sent to Arduino: {final_pwm_to_send}" if self.invert_pwm_var.get() else f"Sent to Arduino: {self.current_pwm_setting_0_255}"
            self.update_status(status_msg)
        else:
            # Still update local state even if not connected, so UI reflects 100
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(100)
            self.last_active_pwm_0_255 = self.current_pwm_setting_0_255
            self.pwm_label_display.config(text=f"PWM: 100%")
            self.update_status("PWM set to 0% (0) (Arduino not connected).")

    def reset_realtime_display(self):
        """Resets all real-time data display labels to their default '---' state."""
        self.update_realtime_display(voltage=None, current=None, temps=[None]*5)

    def update_status(self, message):
        """Updates the status bar message and prints it to the console."""
        self.status_label.config(text=f"Status: {message}")
        print(message) # Also print to console

    def connect_devices(self):
        """
        Attempts to connect to the voltage meter, current meter, and Arduino, or uses mock devices if in debug mode.
        Updates UI based on connection status.
        """
        if self.debug_mode_var.get():
            # --- DEBUG MODE CONNECTION ---
            self.update_status("Connecting in DEBUG MODE...")
            try:
                self.arduino = MockControlArduino(self)
                self.voltage_meter = MockControlMeter(self.arduino)
                self.current_meter = MockControlMeter(self.arduino)
                self.update_status("Mock devices created.")

                # Start pre-recording display
                self.stop_pre_recording_display_event.clear()
                if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                    self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                    self.pre_recording_display_thread.start()

                # Enable UI elements
                self.manual_pwm_radio.config(state=tk.NORMAL)
                self.pid_preheat_radio.config(state=tk.NORMAL)
                self.pid_power_radio.config(state=tk.NORMAL)
                self.corrected_direct_power_radio.config(state=tk.NORMAL)
                if hasattr(self, 'pwm_invert_checkbox'):
                    self.pwm_invert_checkbox.config(state=tk.NORMAL)

                self.update_status("Debug Mode: Devices 'connected'. Live display active.")
                self._handle_control_mode_change()
                self.emergency_stop_button.config(state=tk.NORMAL)
                self.start_button.config(state=tk.NORMAL)
                self.connect_button.config(state=tk.DISABLED)
                # Disable port/address entries
                self.arduino_port_combobox.config(state=tk.DISABLED)
                self.voltage_meter_address_combobox.config(state=tk.DISABLED)
                self.current_meter_address_combobox.config(state=tk.DISABLED)

            except Exception as e:
                self.close_devices()
                error_msg = f"Debug mode connection failed: {e}"
                self.update_status(error_msg)
                messagebox.showerror("Debug Error", error_msg)
                # Reset UI to disconnected state
                self.start_button.config(state=tk.DISABLED)
                self.connect_button.config(state=tk.NORMAL)
                self.arduino_port_combobox.config(state='readonly')
                self.voltage_meter_address_combobox.config(state='readonly')
                self.current_meter_address_combobox.config(state='readonly')
        else:
            # --- REAL HARDWARE CONNECTION ---
            self.update_status("Connecting...")
            try:
                # Connect to Voltage Meter
                vm_addr = self.voltage_meter_address_var.get()
                self.update_status(f"Connecting to Voltage Meter ({vm_addr})...")
                self.voltage_meter = cm.control_meter()
                self.voltage_meter.connect(vm_addr)
                self.voltage_meter.set_voltage_mode()
                self.update_status("Voltage Meter connected.")
                time.sleep(1)

                # Connect to Current Meter
                cm_addr = self.current_meter_address_var.get()
                self.update_status(f"Connecting to Current Meter ({cm_addr})...")
                self.current_meter = cm.control_meter()
                self.current_meter.connect(cm_addr)
                self.current_meter.set_current_mode()
                self.update_status("Current Meter connected.")
                time.sleep(1)
                
                # Connect to Arduino
                arduino_port = self.arduino_port_var.get()
                self.update_status(f"Connecting to Arduino ({arduino_port})...")
                self.arduino = ca.control_arduino(arduino_port, self.ARDUINO_BAUDRATE)
                if not self.arduino.running:
                    self.arduino = None # Ensure it's None if connection truly failed
                    raise Exception("Failed to connect to Arduino or start its read thread.")
                self.update_status("Arduino connected. Initializing live display...")

                # Start pre-recording display now that Arduino is up
                self.stop_pre_recording_display_event.clear()
                if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                    self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                    self.pre_recording_display_thread.start()

                # Enable UI elements
                self.manual_pwm_radio.config(state=tk.NORMAL)
                self.pid_preheat_radio.config(state=tk.NORMAL)
                self.pid_power_radio.config(state=tk.NORMAL)
                self.corrected_direct_power_radio.config(state=tk.NORMAL)

                if hasattr(self, 'pwm_invert_checkbox'):
                    self.pwm_invert_checkbox.config(state=tk.NORMAL)

                # Verify and set initial PWM to 0 (logical)
                self.update_status("Connect: Verifying zero power output...")
                power_confirmed_zero_on_connect = False
                if self.arduino and self.arduino.running:
                    for attempt in range(self.MAX_ZEROING_ATTEMPTS):
                        logical_pwm_value_for_zero = 0
                        self.pwm_percentage_var.set(0)
                        self.current_pwm_setting_0_255 = logical_pwm_value_for_zero
                        final_pwm_to_send = self._apply_pwm_inversion(self.current_pwm_setting_0_255)
                        self.arduino.control_arduino(final_pwm_to_send)
                        if hasattr(self, 'pwm_label_display') and self.pwm_label_display.winfo_exists():
                            self.pwm_label_display.config(text="PWM: 0%")
                        
                        self.update_status(f"Connect: Attempt {attempt + 1}/{self.MAX_ZEROING_ATTEMPTS} - Sent logical PWM 0 (Actual: {final_pwm_to_send}). Checking power...")
                        time.sleep(0.75) # Allow system to stabilize

                        measured_power_on_connect = float('inf')
                        voltage_val_conn, current_val_conn = None, None
                        power_read_successful_conn = False

                        if self.voltage_meter and self.current_meter:
                            try:
                                v_list = self.voltage_meter.read_voltage()
                                if v_list: voltage_val_conn = v_list[0]
                                c_list = self.current_meter.read_current()
                                if c_list: current_val_conn = c_list[0]

                                if voltage_val_conn is not None and current_val_conn is not None:
                                    measured_power_on_connect = float(voltage_val_conn) * float(current_val_conn)
                                    power_read_successful_conn = True
                                    self.update_status(f"Connect: Attempt {attempt + 1} - Measured power: {measured_power_on_connect:.2f}W")
                                else:
                                    self.update_status(f"Connect: Attempt {attempt + 1} - Could not get V/I readings.")
                            except pyvisa.errors.VisaIOError as ve:
                                self.update_status(f"Connect: Attempt {attempt + 1} - VISA error reading power: {ve}")
                            except Exception as e:
                                self.update_status(f"Connect: Attempt {attempt + 1} - Error reading power: {e}")
                        else:
                            self.update_status(f"Connect: Attempt {attempt + 1} - Meters not available to check power.")

                        if power_read_successful_conn and measured_power_on_connect < self.POWER_ZERO_THRESHOLD:
                            self.update_status(f"Connect: Power confirmed near zero ({measured_power_on_connect:.2f}W). PWM inversion is {self.invert_pwm_var.get()}.")
                            power_confirmed_zero_on_connect = True
                            break
                        else:
                            if attempt < self.MAX_ZEROING_ATTEMPTS - 1:
                                new_inversion_state = not self.invert_pwm_var.get()
                                self.invert_pwm_var.set(new_inversion_state)
                                self.update_status(f"Connect: Power not zero ({measured_power_on_connect:.2f}W). Flipped PWM inversion to {new_inversion_state}. Retrying...")
                    
                    if not power_confirmed_zero_on_connect:
                        warn_msg = f"Connect: Power NOT confirmed zero after {self.MAX_ZEROING_ATTEMPTS} attempts. Last: {measured_power_on_connect:.2f}W. Please check system."
                        self.update_status(warn_msg)
                        messagebox.showwarning("Connection Power Warning", warn_msg)
                
                self.update_status(f"Devices connected. Live display active. Ready to record. Initial power zeroing: {'OK' if power_confirmed_zero_on_connect else 'CHECK MANUALLY'}")

                self._handle_control_mode_change() # Update UI based on current (default) mode
                self.emergency_stop_button.config(state=tk.NORMAL) # Enable E-Stop
                time.sleep(1) # Short delay to allow first data to potentially arrive for display

                self.start_button.config(state=tk.NORMAL)
                self.connect_button.config(state=tk.DISABLED)
                # Disable port/address entries
                self.arduino_port_combobox.config(state=tk.DISABLED)
                self.voltage_meter_address_combobox.config(state=tk.DISABLED)
                self.current_meter_address_combobox.config(state=tk.DISABLED)

            except Exception as e:
                self.close_devices() # Ensure any partially opened devices are closed
                error_msg = f"Connection failed: {e}"
                self.update_status(error_msg)
                messagebox.showerror("Connection Error", error_msg)
                self.start_button.config(state=tk.DISABLED)
                self.preheat_button.config(state=tk.DISABLED)
                self.set_pwm_button.config(state=tk.DISABLED) # Disable PWM buttons on connection failure
                self.set_pwm_zero_button.config(state=tk.DISABLED)
                self.emergency_stop_button.config(state=tk.DISABLED) # Disable E-Stop
                # self.load_tr_file_button.config(state=tk.DISABLED) # Removed
                # Equation parameter entries are disabled by _handle_control_mode_change
                self.connect_button.config(state=tk.NORMAL) # Allow retry
                # Ensure port/address entries are enabled on failure (close_devices should handle this)
                self.arduino_port_combobox.config(state='readonly') # Re-enable to readonly
                self.voltage_meter_address_combobox.config(state='readonly') # Re-enable to readonly
                self.current_meter_address_combobox.config(state='readonly') # Re-enable to readonly


    def run_pre_recording_display_loop(self):
        """
        Thread loop to continuously update real-time data display
        when not actively recording or preheating.
        """
        self.update_status("Live display active (not recording/preheating).")
        # print("[PreRecDisp] Pre-recording display loop started.") # Debug

        while not self.stop_pre_recording_display_event.is_set() and self.arduino and self.arduino.running:
            voltage_val, current_val = None, None
            temps_to_display = np.array([-1.0]*5) # Default if Arduino read fails

            try:
                temps_to_display = self.arduino.return_temperature()

                try:
                    if self.voltage_meter:
                        voltage_list = self.voltage_meter.read_voltage()
                        if voltage_list: voltage_val = voltage_list[0]
                except pyvisa.errors.VisaIOError as ve_v:
                    print(f"[PreRecDisp_VISA_ERROR] Voltage meter: {ve_v}")
                    voltage_val = None # Ensure it's None on error
                except Exception as e_v:
                    print(f"[PreRecDisp_ERROR] Voltage meter: {e_v}")
                    voltage_val = None
                
                try:
                    if self.current_meter:
                        current_list = self.current_meter.read_current()
                        if current_list: current_val = current_list[0]
                except pyvisa.errors.VisaIOError as ve_c:
                    print(f"[PreRecDisp_VISA_ERROR] Current meter: {ve_c}")
                    current_val = None # Ensure it's None on error
                except Exception as e_c:
                    print(f"[PreRecDisp_ERROR] Current meter: {e_c}")
                    current_val = None

                if not self.stop_pre_recording_display_event.is_set(): # Check again before UI update
                    self.master.after(0, self.update_realtime_display, voltage_val, current_val, temps_to_display)

            except pyvisa.errors.VisaIOError as ve:
                print(f"[PreRecDisp] General VISA Error during pre-recording display (should be caught by specific handlers): {ve}") # Log error, continue
            except serial.SerialException as se:
                print(f"[PreRecDisp] Serial Error during pre-recording display: {se}") # Log error, loop might exit if arduino.running becomes false
            except Exception as e:
                print(f"[PreRecDisp] Generic Error during pre-recording display: {e}")
            time.sleep(self.pre_recording_display_interval)
        # print("[PreRecDisp] Pre-recording display loop finished.") # Debug

    def toggle_preheat(self):
        """
        Starts or stops the PID preheating process.
        """
        if not self.arduino or not self.arduino.running:
            messagebox.showerror("Error", "Arduino not connected.")
            return

        if self.preheating_active:
            # If it's already active, just stop it.
            self.stop_preheat()
        else:
            # If we are about to START it:
            # 1. Stop any other active controllers.
            if self.power_pid_active: self.stop_power_pid()
            if self.corrected_direct_power_active: self.stop_corrected_direct_power()

            # 2. Set the UI to the correct mode.
            if self.control_mode_var.get() != "PID_PREHEAT":
                self.control_mode_var.set("PID_PREHEAT")
                self._handle_control_mode_change()
            
            # 3. Start the controller.
            self.start_preheat()

    def start_preheat(self):
        """
        Initializes and starts the PID preheating thread and updates UI state.
        """
        try:
            setpoint = self.pid_setpoint_var.get()
            kp = self.pid_kp_var.get()
            ki = self.pid_ki_var.get()
            kd = self.pid_kd_var.get()
        except tk.TclError:
            messagebox.showerror("Input Error", "Invalid PID parameters or target temperature.")
            return

        print(f"[UI_Preheat] Starting Preheat. Target: {setpoint}°C, Kp: {kp}, Ki: {ki}, Kd: {kd}, Sensor: {self.pid_sensor_var.get()}")
        self.preheat_pid_controller = SimplePID(Kp=kp, Ki=ki, Kd=kd, setpoint=setpoint, sample_time=self.pid_control_interval)
        self.preheating_active = True
        self.stop_preheat_event.clear()

        self.preheat_thread = threading.Thread(target=self.run_preheat_loop, daemon=True)
        self.preheat_thread.start()

        # UI updates are now handled by _handle_control_mode_change
        self.preheat_button.config(text="Stop Preheat")
        self._handle_control_mode_change() # Refresh UI states

        self.update_status(f"PID Preheat ({self.pid_sensor_var.get()}) started. Target: {setpoint}°C")

    def stop_preheat(self, set_pwm_to_zero=True):
        """
        Stops the PID preheating thread, turns off the heater, and updates UI state.
        :param set_pwm_to_zero: If True, explicitly sets PWM to 0. If False, leaves PWM as is for seamless mode switching.
        """
        if self.preheat_thread and self.preheat_thread.is_alive():
            self.stop_preheat_event.set()
            self.preheat_thread.join(timeout=2.0)
            if self.preheat_thread.is_alive():
                print("[UI] WARNING: Preheat thread did not terminate in time!")
        self.preheat_thread = None
        self.preheating_active = False

        if set_pwm_to_zero:
            # Set PWM to a safe state (0) when stopping the controller.
            if self.arduino and self.arduino.running:
                final_pwm_to_send = self._apply_pwm_inversion(0)
                self.arduino.control_arduino(final_pwm_to_send)
                self.update_status("PID Preheat stopped. PWM set to 0.")
            else:
                self.update_status("PID Preheat stopped.")
        else:
            self.update_status("PID Preheat controller stopped for mode switch.")

        # UI updates are now handled by _handle_control_mode_change
        # Ensure the button text and status label are reset correctly if not switching mode
        if self.control_mode_var.get() == "PID_PREHEAT": # Only if still in this mode
            self.preheat_button.config(text="Start Preheat")
            self.preheat_status_label.config(text="Preheat Status: Off")
        self._handle_control_mode_change() # Refresh UI states

    def run_preheat_loop(self):
        """
        Thread loop for PID control during preheating.
        Reads selected sensor, calculates PID output, sends PWM to Arduino, and updates preheat status UI.
        """
        if not self.preheat_pid_controller:
            print("[PreheatLoop] PID controller not initialized.")
            return
        
        print(f"[PreheatLoop] Initializing. Target: {self.pid_setpoint_var.get()}°C, Kp: {self.pid_kp_var.get()}, Ki: {self.pid_ki_var.get()}, Kd: {self.pid_kd_var.get()}, Sensor: {self.pid_sensor_var.get()}")
        self.preheat_pid_controller.reset() # Reset PID state before starting
        self.preheat_pid_controller.set_setpoint(self.pid_setpoint_var.get()) # Ensure setpoint is current
        self.preheat_pid_controller.set_tunings(self.pid_kp_var.get(), self.pid_ki_var.get(), self.pid_kd_var.get())

        last_ui_update_time = time.time()
        ui_update_interval = self.pre_recording_display_interval 

        while not self.stop_preheat_event.is_set() and self.arduino and self.arduino.running:
            selected_sensor_name = self.pid_sensor_var.get()
            temps = self.arduino.return_temperature()
            actual_sensor_temp = None # Will store the temperature from the selected sensor

            if temps.size == 5: # Ensure we have data for all 5 sensors
                if selected_sensor_name == "TM_1" and temps[2] is not None and temps[2] > -1: # TM_1 is at index 2
                    actual_sensor_temp = temps[2]
                elif selected_sensor_name == "TH" and temps[0] is not None and temps[0] > -1: # TH is at index 0
                    actual_sensor_temp = temps[0]
            
            pwm_output_float = self.preheat_pid_controller.update(actual_sensor_temp)
            # print(f"[PreheatLoop] Sensor ({self.pid_sensor_var.get()}): {actual_sensor_temp}, PID Output (0-255): {pwm_output_float}")

            if pwm_output_float is not None: # pwm_output_float could be None if update interval not met
                pwm_to_send = int(round(pwm_output_float))
                self.last_active_pwm_0_255 = pwm_to_send # Update last active PWM
                final_pwm_value_sent = self._apply_pwm_inversion(pwm_to_send)
                self.arduino.control_arduino(final_pwm_value_sent)
                # print(f"[PreheatLoop] Sent PWM to Arduino: {pwm_to_send}")
                
                current_time = time.time()
                if current_time - last_ui_update_time >= ui_update_interval:
                    target_temp_str = f"Target ({selected_sensor_name}): {self.pid_setpoint_var.get():.1f}°C"
                    actual_temp_str = f"Actual ({selected_sensor_name}): {actual_sensor_temp if actual_sensor_temp is not None else 'N/A'}°C"
                    
                    pwm_output_percentage = int(round(pwm_output_float / 255 * 100))
                    raw_pwm_value = pwm_to_send
                    
                    pwm_info_str = f"PID PWM: {pwm_output_percentage}% ({raw_pwm_value})"
                    if self.invert_pwm_var.get():
                        pwm_info_str += f" -> Sent: {final_pwm_value_sent}"
                    status_text_to_display = f"{target_temp_str}\n{actual_temp_str}\n{pwm_info_str}"
                    self.master.after(0, self.preheat_status_label.config, {"text": status_text_to_display})
                    last_ui_update_time = current_time # Corrected variable name

            time.sleep(self.pid_control_interval / 2) # Shorter sleep, PID internal timing handles the actual update rate
        print("[PreheatLoop] Preheat loop finished.")

    def toggle_power_pid(self):
        """Starts or stops the PID power control process."""
        if not (self.arduino and self.arduino.running and self.voltage_meter and self.current_meter):
            messagebox.showerror("Error", "Arduino or Meters not connected.")
            return

        if self.power_pid_active:
            # If it's already active, just stop it.
            self.stop_power_pid()
        else:
            # If we are about to START it:
            # 1. Stop any other active controllers.
            if self.preheating_active: self.stop_preheat()
            if self.corrected_direct_power_active: self.stop_corrected_direct_power()

            # 2. Set the UI to the correct mode.
            if self.control_mode_var.get() != "PID_POWER":
                self.control_mode_var.set("PID_POWER")
                self._handle_control_mode_change()
            
            # 3. Start the controller.
            self.start_power_pid()

    def start_power_pid(self):
        """Initializes and starts the PID power control thread."""
        try:
            setpoint = self.power_pid_setpoint_var.get()
            kp = self.power_pid_kp_var.get()
            ki = self.power_pid_ki_var.get()
            kd = self.power_pid_kd_var.get()
        except tk.TclError:
            messagebox.showerror("Input Error", "Invalid Power PID parameters or target power.")
            return

        print(f"[UI_PowerPID] Starting Power PID. Target: {setpoint}W, Kp: {kp}, Ki: {ki}, Kd: {kd}")
        self.power_pid_controller = SimplePID(Kp=kp, Ki=ki, Kd=kd, setpoint=setpoint, sample_time=self.pid_control_interval)
        self.power_pid_active = True
        self.stop_power_pid_event.clear()

        self.power_pid_thread = threading.Thread(target=self.run_power_pid_loop, daemon=True)
        self.power_pid_thread.start()

        self.power_pid_button.config(text="Stop Power Ctrl")
        self._handle_control_mode_change() # Refresh UI states
        self.update_status(f"PID Power Control started. Target: {setpoint}W")

    def stop_power_pid(self, set_pwm_to_zero=True):
        """
        Stops the PID power control thread and turns off heater.
        :param set_pwm_to_zero: If True, explicitly sets PWM to 0. If False, leaves PWM as is for seamless mode switching.
        """
        if self.power_pid_thread and self.power_pid_thread.is_alive():
            self.stop_power_pid_event.set()
            self.power_pid_thread.join(timeout=2.0)
            if self.power_pid_thread.is_alive():
                print("[UI] WARNING: Power PID thread did not terminate in time!")
        self.power_pid_thread = None
        self.power_pid_active = False

        if set_pwm_to_zero:
            # Set PWM to a safe state (0) when stopping the controller.
            if self.arduino and self.arduino.running:
                final_pwm_to_send = self._apply_pwm_inversion(0)
                self.arduino.control_arduino(final_pwm_to_send)
                self.update_status("PID Power Control stopped. PWM set to 0.")
            else:
                self.update_status("PID Power Control stopped.")
        else:
            self.update_status("PID Power controller stopped for mode switch.")

        if self.control_mode_var.get() == "PID_POWER": # Only if still in this mode
            self.power_pid_button.config(text="Start Power Ctrl")
            self.power_pid_status_label.config(text="Power PID Status: Off")
        self._handle_control_mode_change() # Refresh UI states

    def run_power_pid_loop(self):
        """Thread loop for PID power control."""
        if not self.power_pid_controller:
            print("[PowerPIDLoop] Power PID controller not initialized.")
            return
        
        print(f"[PowerPIDLoop] Initializing. Target: {self.power_pid_setpoint_var.get()}W, Kp: {self.power_pid_kp_var.get()}, Ki: {self.power_pid_ki_var.get()}, Kd: {self.power_pid_kd_var.get()}")
        # --- Get current power to prime PID's _last_input after reset ---
        initial_power_for_pid = None
        if self.voltage_meter and self.current_meter:
            try:
                # This read is primarily for priming _last_input.
                voltage_list = self.voltage_meter.read_voltage() 
                current_list = self.current_meter.read_current()
                if voltage_list and current_list and voltage_list[0] is not None and current_list[0] is not None:
                    initial_power_for_pid = float(voltage_list[0]) * float(current_list[0])
            except pyvisa.errors.VisaIOError as ve:
                print(f"[PowerPIDLoop_INIT_VISA_ERROR] Error getting initial power for PID priming: {ve}")
            except Exception as e:
                print(f"[PowerPIDLoop_ERROR] Error getting initial power for PID priming: {e}")
        # --- End priming read ---



        self.power_pid_controller.reset()
        if initial_power_for_pid is not None:
            self.power_pid_controller._last_input = initial_power_for_pid # CRITICAL: Prime _last_input
            print(f"[PowerPIDLoop_INFO] Primed _last_input to {initial_power_for_pid:.2f}W")
        else:
            # This print was already here, just adding a tag
            print(f"[PowerPIDLoop_WARN] Could not prime _last_input. D-term might be incorrect on first step if power is high.")

        self.power_pid_controller.set_setpoint(self.power_pid_setpoint_var.get())
        self.power_pid_controller.set_tunings(self.power_pid_kp_var.get(), self.power_pid_ki_var.get(), self.power_pid_kd_var.get())

        last_ui_update_time = time.time()
        ui_update_interval = self.pre_recording_display_interval

        while not self.stop_power_pid_event.is_set() and self.arduino and self.arduino.running and self.voltage_meter and self.current_meter:
            current_power = None
            voltage_val, current_val = None, None

            try:
                if self.voltage_meter:
                    voltage_list_loop = self.voltage_meter.read_voltage()
                    if voltage_list_loop: voltage_val = voltage_list_loop[0]
            except pyvisa.errors.VisaIOError as ve_v:
                print(f"[PowerPIDLoop_VISA_ERROR] Voltage meter: {ve_v}")
            except Exception as e_v:
                print(f"[PowerPIDLoop_ERROR] Voltage meter: {e_v}")
            
            try:
                if self.current_meter:
                    current_list_loop = self.current_meter.read_current()
                    if current_list_loop: current_val = current_list_loop[0]
            except pyvisa.errors.VisaIOError as ve_c:
                print(f"[PowerPIDLoop_VISA_ERROR] Current meter: {ve_c}")
            except Exception as e_c:
                print(f"[PowerPIDLoop_ERROR] Current meter: {e_c}")

            if voltage_val is not None and current_val is not None:
                current_power = float(voltage_val) * float(current_val)
            pwm_output_float = self.power_pid_controller.update(current_power)
            print(f"[PowerPIDLoop] Current Power: {current_power}, PID Output (0-255): {pwm_output_float}")

            if pwm_output_float is not None:
                pwm_to_send = int(round(pwm_output_float))
                self.last_active_pwm_0_255 = pwm_to_send # Update last active PWM
                final_pwm_value_sent = self._apply_pwm_inversion(pwm_to_send)
                self.arduino.control_arduino(final_pwm_value_sent)
                # print(f"[PowerPIDLoop] Sent PWM to Arduino: {pwm_to_send}")

                if time.time() - last_ui_update_time >= ui_update_interval:
                    pwm_output_percentage = int(round(pwm_output_float / 255 * 100))
                    raw_pwm_value = pwm_to_send

                    pwm_info_str = f"PID PWM: {pwm_output_percentage}% ({raw_pwm_value})"
                    if self.invert_pwm_var.get():
                        pwm_info_str += f" -> Sent: {final_pwm_value_sent}"
                    status_text_to_display = f"Target: {self.power_pid_setpoint_var.get():.1f}W\nActual: {current_power:.2f}W\n{pwm_info_str}"
                    self.master.after(0, self.power_pid_status_label.config, {"text": status_text_to_display})
                    last_ui_update_time = time.time()

            time.sleep(self.pid_control_interval / 2)
        print("[PowerPIDLoop] Power PID loop finished.")

    # --- Corrected Direct Power Methods ---
    def toggle_corrected_direct_power(self): # This is effectively handled by toggle_power_pid
        # This method is now called by the dedicated corrected_power_button
        if not (self.arduino and self.arduino.running and self.voltage_meter and self.current_meter):
            messagebox.showerror("Error", "Arduino or Meters not connected.")
            return

        if self.corrected_direct_power_active:
            # If it's already active, just stop it.
            self.stop_corrected_direct_power()
        else:
            # If we are about to START it:
            # 1. Stop any other active controllers.
            if self.preheating_active: self.stop_preheat()
            if self.power_pid_active: self.stop_power_pid()

            # 2. Set the UI to the correct mode.
            if self.control_mode_var.get() != "CORRECTED_DIRECT_POWER":
                self.control_mode_var.set("CORRECTED_DIRECT_POWER")
                self._handle_control_mode_change()
            
            # 3. Start the controller.
            self.start_corrected_direct_power()
    # The start_corrected_direct_power and stop_corrected_direct_power methods
    # will now update self.corrected_power_button and self.corrected_power_status_label

    def start_corrected_direct_power(self):
        try:
            target_power = self.corrected_power_target_var.get() # Use dedicated target var
            param_A = self.equation_param_A_var.get()
            param_B = self.equation_param_B_var.get() # param_B can be zero or negative
            param_C = self.equation_param_C_var.get() 
            rds_on = self.rds_on_var.get()
            v_supply_max_pwm = self.nominal_supply_voltage_var.get()

            if target_power < 0:
                messagebox.showwarning("Input Warning", "Target power should be non-negative. Using absolute value.")
                target_power = abs(target_power)
                self.corrected_power_target_var.set(target_power) # Update dedicated var
            
            # param_C can be any real number
            if param_A <= 0: # A must be positive for R = A*exp(B*T)
                messagebox.showerror("Input Error", "Equation parameter A must be positive.")
                return
            
            if rds_on < 0:
                messagebox.showerror("Input Error", "MOSFET Rds(on) must be non-negative.")
                return
            
            if v_supply_max_pwm <= 0:
                messagebox.showerror("Input Error", "Nominal Supply Voltage (V_supply_max_pwm) must be positive.")
                return


        except tk.TclError:
            messagebox.showerror("Input Error", "Invalid target power or equation parameter value(s).")
            return

        # print(f"[UI_CorrectedDirectPwr] Starting. Target: {target_power}W, EqParams: A={param_A}, B={param_B}, C={param_C}")
        self.corrected_direct_power_active = True
        self.stop_corrected_direct_power_event.clear()

        self.corrected_direct_power_thread = threading.Thread(target=self.run_corrected_direct_power_loop, daemon=True)
        self.corrected_direct_power_thread.start()

        self.corrected_power_button.config(text="Stop Corr.Pwr") 
        self._handle_control_mode_change() # Refresh UI states (mainly for other controls)
        self.update_status(f"Corrected Direct Power Control started. Target: {target_power}W")

    def stop_corrected_direct_power(self, set_pwm_to_zero=True):
        """
        Stops the Corrected Direct Power thread and turns off heater.
        :param set_pwm_to_zero: If True, explicitly sets PWM to 0. If False, leaves PWM as is for seamless mode switching.
        """
        if self.corrected_direct_power_thread and self.corrected_direct_power_thread.is_alive():
            self.stop_corrected_direct_power_event.set()
            self.corrected_direct_power_thread.join(timeout=2.0)
            if self.corrected_direct_power_thread.is_alive():
                print("[UI] WARNING: Corrected Direct Power thread did not terminate in time!")
        self.corrected_direct_power_thread = None
        self.corrected_direct_power_active = False

        if set_pwm_to_zero:
            # Set PWM to a safe state (0) when stopping the controller.
            if self.arduino and self.arduino.running:
                final_pwm_to_send = self._apply_pwm_inversion(0)
                self.arduino.control_arduino(final_pwm_to_send)
                self.update_status("Corrected Direct Power Control stopped. PWM set to 0.")
            else:
                self.update_status("Corrected Direct Power Control stopped.")
        else:
            self.update_status("Corrected Direct Power controller stopped for mode switch.")

        # Update dedicated button and status label
        self.corrected_power_button.config(text="Start Corr.Pwr") 
        self.corrected_power_status_label.config(text="Corrected Pwr: Off") 
        
        self._handle_control_mode_change() # Refresh UI states (mainly for other controls)

    def run_corrected_direct_power_loop(self):
        last_ui_update_time = time.time()
        ui_update_interval = self.pre_recording_display_interval
        target_power_w = self.corrected_power_target_var.get() 
        # print(f"[CorrectedPwrLoop_DEBUG] Initial Target Power (P_target_w): {target_power_w} W")
        # print(f"[CorrectedPwrLoop_DEBUG] NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM: {self.NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM} V")
        # Get equation parameters once at the start of the loop
        try:
            param_A = self.equation_param_A_var.get()
            param_B = self.equation_param_B_var.get()
            param_C = self.equation_param_C_var.get()
            rds_on = self.rds_on_var.get() # Get Rds(on)
            v_supply_max_pwm = self.nominal_supply_voltage_var.get()

            print(f"[CorrectedPwrLoop_PARAMS] Initial Corrected Power Parameters: "
                  f"Target_Power: {target_power_w:.2f}W, A: {param_A:.4f}, B: {param_B:.6f}, C: {param_C:.4f}, "
                  f"Rds(on): {rds_on:.3f}Ω, V_supply_max_pwm: {v_supply_max_pwm:.2f}V")


            if param_A <= 0: # Should have been caught by start, but good to re-check
                print("[CorrectedDirectPwrLoop] Error: Parameter A is not positive. Stopping loop.")
                self.master.after(0, self.stop_corrected_direct_power) # Schedule stop on main thread
                # self.corrected_power_button.config(text="Start Corr.Pwr") # Reset button on error
                return
            if rds_on < 0: # Should have been caught by start
                print("[CorrectedDirectPwrLoop] Error: Rds(on) is negative. Stopping loop.")
                self.master.after(0, self.stop_corrected_direct_power)
                return
            if v_supply_max_pwm <= 0: # Should have been caught by start
                print("[CorrectedDirectPwrLoop] Error: V_supply_max_pwm is not positive. Stopping loop.")
                self.master.after(0, self.stop_corrected_direct_power)
                return
        except tk.TclError:
            # print(f"[CorrectedPwrLoop_DEBUG] Error reading parameters. Target Power: {target_power_w}")
            print("[CorrectedDirectPwrLoop] Error: Could not get equation parameters. Stopping loop.")
            self.master.after(0, self.stop_corrected_direct_power)
            return

        while not self.stop_corrected_direct_power_event.is_set() and self.arduino and self.arduino.running and self.voltage_meter and self.current_meter:
            voltage_val, current_val = None, None
            voltage_list, current_list = None, None # Keep for structure if needed, but use _val

            try:
                if self.voltage_meter:
                    voltage_list = self.voltage_meter.read_voltage()
                    if voltage_list: voltage_val = voltage_list[0]
            except pyvisa.errors.VisaIOError as ve_v:
                print(f"[CorrectedPwrLoop_VISA_ERROR] Timeout/Error reading voltage: {ve_v}")
            except Exception as e_v:
                print(f"[CorrectedPwrLoop_ERROR] Unexpected error reading voltage: {e_v}")

            try:
                if self.current_meter:
                    current_list = self.current_meter.read_current()
                    if current_list: current_val = current_list[0]
            except pyvisa.errors.VisaIOError as ve_c:
                print(f"[CorrectedPwrLoop_VISA_ERROR] Timeout/Error reading current: {ve_c}")
            except Exception as e_c:
                print(f"[CorrectedPwrLoop_ERROR] Unexpected error reading current: {e_c}")

            temps_arduino = self.arduino.return_temperature() # TM_1 is at index 2
            measured_power, R_heater, pwm_final_sent = None, 0.0, 0 # R_heater default to 0 if not calculable
            if voltage_val is not None and current_val is not None: # We have V and I readings
                measured_power = float(voltage_val) * float(current_val)
                # print(f"[CorrectedPwrLoop_DEBUG] Measured Total Power (V*I): {measured_power:.3f} W (V={voltage_list[0]:.3f}, I={current_list[0]:.3f})")

            if temps_arduino.size == 5 and temps_arduino[2] is not None and temps_arduino[2] > -1: # Valid TM_1
                current_temp_for_R = temps_arduino[2] # Using TM_1
                # print(f"[CorrectedPwrLoop_DEBUG] Current Temp for R (TM_1): {current_temp_for_R}°C")
                R_heater = self.get_resistance_from_equation(current_temp_for_R, param_A, param_B, param_C)
                # print(f"[CorrectedPwrLoop_DEBUG] Calculated R_heater: {R_heater} Ω (using A={param_A}, B={param_B}, C={param_C}, Rds(on)={rds_on})")

                if R_heater is not None and R_heater > 1e-3: # Ensure R_heater is valid and reasonably positive
                    if target_power_w >= 0 and v_supply_max_pwm > 0:
                        # Step 1: Calculate target current I_target for R_heater
                        I_target = np.sqrt(target_power_w / R_heater)
                        # print(f"[CorrectedPwrLoop_DEBUG] Step 1: I_target = sqrt({target_power_w} / {R_heater}) = {I_target:.4f} A")

                        # Step 2: Calculate total voltage V_total_required for (R_heater + rds_on)
                        V_total_required = I_target * (R_heater + rds_on)
                        # print(f"[CorrectedPwrLoop_DEBUG] Step 2: V_total_required = {I_target:.4f} * ({R_heater} + {rds_on}) = {V_total_required:.4f} V")
                        # Step 3: Calculate PWM based on V_total_required
                        pwm_calculated_float = (V_total_required / v_supply_max_pwm) * 255.0
                        # print(f"[CorrectedPwrLoop_DEBUG] Step 3: pwm_calculated_float = ({V_total_required:.4f} / {v_supply_max_pwm}) * 255.0 = {pwm_calculated_float:.4f}")
                        
                        pwm_final_sent = int(round(max(0, min(pwm_calculated_float, 255))))
                        self.last_active_pwm_0_255 = pwm_final_sent # Update last active PWM
                        
                        # Comprehensive debug print for this iteration
                        print(f"[CorrectedPwrLoop_ITER_DEBUG] "
                              f"TgtP_h: {target_power_w:.2f}W, MeasP_tot: {measured_power if measured_power is not None else 'N/A'}W, "
                              f"Temp_R: {current_temp_for_R:.2f}C, R_h: {R_heater:.3f}Ω, Rds: {rds_on:.3f}Ω, "
                              f"I_tgt: {I_target:.3f}A, V_tot_req: {V_total_required:.3f}V, PWM_calc: {pwm_calculated_float:.2f}, PWM_sent: {pwm_final_sent}")
                        actual_pwm_sent_to_arduino = self._apply_pwm_inversion(pwm_final_sent)
                        self.arduino.control_arduino(actual_pwm_sent_to_arduino)
                        # print(f"[CorrectedPwrLoop_DEBUG] PWM Sent: {pwm_final_sent}") # Made redundant by the comprehensive print above
                    else: # target power is negative, or V_supply is not set
                        # At this point, R_heater and current_temp_for_R should be valid
                        # print(f"[CorrectedPwrLoop_DEBUG_ITERATION_OFF] Cond: TgtP<0 or V_supply<=0. PWM=0. "
                        #       f"TgtP_h: {target_power_w:.2f}, MeasP_tot: {measured_power if measured_power is not None else 'N/A'}, "
                        #       f"Temp_R: {current_temp_for_R:.2f}C, R_h: {R_heater:.3f}Ω")
                        effective_zero_pwm = self._apply_pwm_inversion(0)
                        self.last_active_pwm_0_255 = 0 # Update last active PWM
                        self.arduino.control_arduino(effective_zero_pwm)
                        pwm_final_sent = 0
                else: # R_heater could not be calculated or is not positive enough
                    # current_temp_for_R was valid, R_heater is the issue
                    # print(f"[CorrectedPwrLoop_DEBUG_ITERATION_OFF] Cond: R_heater invalid/small ({R_heater}). PWM=0. "
                    #       f"MeasP_tot: {measured_power if measured_power is not None else 'N/A'}, Temp_R: {current_temp_for_R:.2f}C")
                    effective_zero_pwm = self._apply_pwm_inversion(0)
                    self.arduino.control_arduino(effective_zero_pwm)
                    self.last_active_pwm_0_255 = 0 # Update last active PWM
                    pwm_final_sent = 0
            else: # No valid temperature for R_heater
                # current_temp_for_R is the issue, R_heater would not be calculated
                # print(f"[CorrectedPwrLoop_DEBUG_ITERATION_OFF] Cond: Invalid Temp for R_heater. PWM=0. "
                #       f"MeasP_tot: {measured_power if measured_power is not None else 'N/A'}")
                effective_zero_pwm = self._apply_pwm_inversion(0)
                self.last_active_pwm_0_255 = 0 # Update last active PWM
                self.arduino.control_arduino(effective_zero_pwm) # Safety: turn off
                pwm_final_sent = 0

            if time.time() - last_ui_update_time >= ui_update_interval:
                # Prepare display strings
                measured_total_power_str = f"{measured_power:.1f}" if measured_power is not None else "--"
                temp_tm1_val = None
                # V_total_required might not be defined if pwm_final_sent is 0 due to an early exit condition
                current_V_total_required = locals().get('V_total_required', None)

                if temps_arduino.size == 5 and temps_arduino[2] is not None and temps_arduino[2] > -1:
                    temp_tm1_val = temps_arduino[2]
                temp_tm1_str = f"{temp_tm1_val:.1f}" if temp_tm1_val is not None else "--"
                
                r_heater_str = "--"
                if R_heater is not None and R_heater > 0: # R_heater could be float('inf')
                    if R_heater == float('inf'):
                        r_heater_str = "inf"
                    else:
                        r_heater_str = f"{R_heater:.2f}"

                pwm_perc_str = f"{int(round(pwm_final_sent/255*100))}%"
                rds_on_str = f"{rds_on:.3f}" 

                # Construct the 3-line status text
                line1 = f"P Tgt(Htr): {target_power_w:.1f}W Act(Tot): {measured_total_power_str}W"
                line2 = f"T1: {temp_tm1_str}°C Rh: {r_heater_str}Ω Rds: {rds_on_str}Ω"
                
                line3_parts = [f"PWM: {pwm_perc_str} ({pwm_final_sent})"]
                if current_V_total_required is not None and pwm_final_sent > 0 : # Show V_total_required if it was calculated and relevant
                    line3_parts.append(f"VReq: {current_V_total_required:.2f}V")
                if self.invert_pwm_var.get():
                    line3_parts.append(f"Sent: {self._apply_pwm_inversion(pwm_final_sent)}")
                line3 = " ".join(line3_parts)
                status_text_to_display = f"{line1}\n{line2}\n{line3}"
                self.master.after(0, self.corrected_power_status_label.config, {"text": status_text_to_display})
                last_ui_update_time = time.time()
            time.sleep(self.pid_control_interval / 2)
        # print("[CorrectedDirectPwrLoop] Corrected Direct Power loop finished.")

    def get_resistance_from_equation(self, temperature_celsius, param_A, param_B, param_C):
        """Calculates resistance using R(T) = A * exp(B*T) + C."""
        if temperature_celsius is None or param_A is None or param_B is None or param_C is None:
            return None # Cannot calculate if any input is missing
        try:
            resistance = param_A * np.exp(param_B * temperature_celsius) + param_C
            return resistance
        except OverflowError: # exp(B*T) might be too large
            print(f"[WARN_RES_EQ] OverflowError in resistance calculation with T={temperature_celsius}, A={param_A}, B={param_B}, C={param_C}")
            return float('inf') # Or handle appropriately, e.g., return None or a very large number
        except Exception as e: # Catch other potential math errors
            print(f"[ERROR_RES_EQ] Error in resistance calculation: {e}")
            return None

    def start_recording(self):
        """Starts the data recording process."""
        if not (self.voltage_meter and self.current_meter and self.arduino and self.arduino.running):
            messagebox.showerror("Error", "Devices are not connected properly.")
            return

        self.recording = True
        self.elapsed_time = 0
        self.data = [] # Clear previous data
        self.flip_event_times = [] # Clear previous flip events
        # Stop pre-recording display thread if it's running
        if self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive():
            self.stop_pre_recording_display_event.set()
            self.pre_recording_display_thread.join(timeout=2.0) 
            if self.pre_recording_display_thread.is_alive():
                print("[UI] WARNING: Pre-recording display thread did not terminate in time before recording!")
            self.pre_recording_display_thread = None 

        self.update_time_label() # Initialize time label

        # Start UI timer thread
        self.timer_thread = threading.Thread(target=self.run_ui_timer, daemon=True)
        self.timer_thread.start()

        # Start data collection thread
        self.data_collection_thread = threading.Thread(target=self.run_data_collection_loop)
        self.data_collection_thread.start()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.connect_button.config(state=tk.DISABLED) # Disable connect/disconnect during recording
        self.steak_type_entry.config(state=tk.DISABLED) # Disable steak type entry during recording

        self.flip_button.config(state=tk.NORMAL) # Enable flip button during recording
        self.update_status("Recording started...")

    def stop_recording(self):
        """Stops the data recording process, saves data, and resets UI state."""
        # print("[UI] stop_recording called") # Debug
        self.recording = False # Signal threads to stop
        # print(f"[UI] self.recording set to {self.recording}") # Debug

        if self.data_collection_thread and self.data_collection_thread.is_alive():
            self.update_status("Stopping recording, please wait for data processing...")
            # print("[UI] Attempting to join data_collection_thread...") # Debug
            self.data_collection_thread.join(timeout=10.0) # Wait for data collection to finish, with a 10-second timeout
            if self.data_collection_thread.is_alive():
                print("[UI] WARNING: Data collection thread did not terminate in time after stopping!") # Should ideally not happen
            # else: # Debug
                # print("[UI] Data collection thread joined successfully.") # Debug
        
        # UI timer thread is daemon, will stop.

        # Restart pre-recording display if devices are still connected
        if self.arduino and self.arduino.running: 
            self.stop_pre_recording_display_event.clear()
            if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                self.pre_recording_display_thread.start()

        # print("[UI] Proceeding after join attempt.") # Debug
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.steak_type_entry.config(state=tk.NORMAL) # Re-enable steak type entry
        self.flip_button.config(state=tk.DISABLED) # Disable flip button after recording
        self.connect_button.config(state=tk.NORMAL if not (self.voltage_meter and self.current_meter and self.arduino) else tk.DISABLED)

        if self.data:
            self.update_status("Recording stopped. Saving and plotting data...")
            self.save_and_plot_data()
            self.update_status(f"Data saved and plotted. Ready for new recording or disconnect.")
        else:
            self.update_status("Recording stopped. No data collected.")
        
        if not (self.arduino and self.arduino.running): # If Arduino died or was disconnected
            self.reset_realtime_display()

        # Re-enable connect button if devices are connected, otherwise it should stay as is.
        if self.voltage_meter and self.current_meter and self.arduino:
             self.connect_button.config(state=tk.DISABLED) # Still connected
             # Keep port entries disabled
             self.arduino_port_combobox.config(state=tk.DISABLED)
             self.voltage_meter_address_combobox.config(state=tk.DISABLED)
             self.current_meter_address_combobox.config(state=tk.DISABLED)
        else:
             self.connect_button.config(state=tk.NORMAL) # Not connected, allow reconnect

    def run_ui_timer(self):
        """Thread loop to update the elapsed time label every second during recording."""
        while self.recording:
            time.sleep(1)
            if not self.recording: # Check again in case stop_recording was called
                break
            self.elapsed_time += 1
            # Schedule UI update on main thread
            self.master.after(0, self.update_time_label)

    def update_time_label(self):
        """Updates the elapsed time label in the UI."""
        if not hasattr(self, 'master') or not self.master.winfo_exists():
            return
        hours = self.elapsed_time // 3600
        minutes = (self.elapsed_time % 3600) // 60
        seconds = self.elapsed_time % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        try:
            if hasattr(self, 'time_label') and self.time_label.winfo_exists():
                self.time_label.config(text=time_str)
        except tk.TclError:
            print("TclError in update_time_label, likely widget destroyed.")

    def update_realtime_display(self, voltage, current, temps):
        """
        Updates all real-time data labels in the UI.
        Calculates power and resistance if voltage and current are available.
        """
        if not hasattr(self, 'master') or not self.master.winfo_exists():
            return
        try:
            v_text = f"Voltage: {voltage:.3f} V" if voltage is not None else "Voltage: --- V"
            c_text = f"Current: {current:.3f} A" if current is not None else "Current: --- A"
            if voltage is not None and current is not None:
                power = voltage * current
                p_text = f"Power: {power:.2f} W"
            else:
                p_text = "Power: --- W"

            th_text = f"TH: {temps[0]:.2f} °C" if temps[0] is not None and temps[0] > -1 else "TH: --- °C"
            tt_text = f"Ttest: {temps[1]:.2f} °C" if temps[1] is not None and temps[1] > -1 else "Ttest: --- °C"
            tm1_text = f"TM_1: {temps[2]:.2f} °C" if temps[2] is not None and temps[2] > -1 else "TM_1: --- °C"
            tm2_text = f"TM_2: {temps[3]:.2f} °C" if temps[3] is not None and temps[3] > -1 else "TM_2: --- °C"
            tm3_text = f"TM_3: {temps[4]:.2f} °C" if temps[4] is not None and temps[4] > -1 else "TM_3: --- °C"

            if self.power_display_label.winfo_exists(): self.power_display_label.config(text=p_text)
            if self.voltage_display_label.winfo_exists(): self.voltage_display_label.config(text=v_text)
            if self.current_display_label.winfo_exists(): self.current_display_label.config(text=c_text)
            if self.th_display_label.winfo_exists(): self.th_display_label.config(text=th_text)
            if self.ttest_display_label.winfo_exists(): self.ttest_display_label.config(text=tt_text)
            if self.tm1_display_label.winfo_exists(): self.tm1_display_label.config(text=tm1_text)
            if self.tm2_display_label.winfo_exists(): self.tm2_display_label.config(text=tm2_text)
            if self.tm3_display_label.winfo_exists(): self.tm3_display_label.config(text=tm3_text)
        except tk.TclError:
            print("TclError in update_realtime_display, likely widget destroyed.")
        except Exception as e:
            print(f"Error updating real-time display: {e}")

    def run_data_collection_loop(self):
        """
        Main data collection loop running in a separate thread.
        Controls Arduino PWM, reads data from meters and Arduino, stores data, and updates UI.
        """
        loop_start_time = time.time()
        self.update_status("Data collection thread started.") # Removed PWM display from here
        # print("[DCL] Data Collection Loop started") # Debug

        while self.recording:
            current_loop_time = time.time()
            
            try:                
                # The active control loop (PID or manual button) is responsible for sending PWM commands.
                # This loop's only job is to read and record data.
                voltage_val, current_val = None, None
                temperature_array = np.array([-1.0]*5) # Default if Arduino read fails or not connected

                try:
                    if self.voltage_meter:
                        voltage_list_loop = self.voltage_meter.read_voltage()
                        if voltage_list_loop: voltage_val = voltage_list_loop[0]
                except pyvisa.errors.VisaIOError as ve_v:
                    print(f"[DCL_VISA_ERROR] Voltage meter: {ve_v}")
                except Exception as e_v:
                    print(f"[DCL_ERROR] Voltage meter: {e_v}")

                try:
                    if self.current_meter:
                        current_list_loop = self.current_meter.read_current()
                        if current_list_loop: current_val = current_list_loop[0]
                except pyvisa.errors.VisaIOError as ve_c:
                    print(f"[DCL_VISA_ERROR] Current meter: {ve_c}")
                except Exception as e_c:
                    print(f"[DCL_ERROR] Current meter: {e_c}")
                
                if self.arduino and self.arduino.running:
                    temperature_array = self.arduino.return_temperature()

                if voltage_val is not None and current_val is not None and temperature_array.size == 5:
                    self.data.append([
                        current_loop_time - loop_start_time,
                        float(voltage_val), # This is the voltage value
                        float(current_val),
                        temperature_array[0], # TH
                        temperature_array[1], # Ttest
                        temperature_array[2], # TM_1
                        temperature_array[3], # TM_2
                        temperature_array[4],  # TM_3
                        self.control_mode_var.get() # Current control mode
                    ])
                    # Schedule real-time display update
                    self.master.after(0, self.update_realtime_display, voltage_val, current_val, temperature_array)
                else:
                    print("[DCL] Warning: Incomplete data from meter or Arduino in a cycle.")

            except pyvisa.errors.VisaIOError as ve:
                print(f"[DCL] VISA Error during data collection cycle: {ve}")
                # Consider if you want to stop recording or try to reinitialize on certain VISA errors
            except serial.SerialException as se:
                print(f"[DCL] Serial Error during data collection cycle: {se}")
            except Exception as e:
                print(f"[DCL] Generic Error during data collection cycle: {e}")
            
            time.sleep(self.sample_interval) # self.sample_interval is 0.01s
        
        # print(f"[DCL] Loop finished. self.recording = {self.recording}") # Debug
        self.update_status("Data collection thread finished.")

    def save_and_plot_data(self):
        """
        Saves the collected data to a CSV file and generates plots.
        Filename includes timestamp, steak type, duration, and average power.
        """
        if not self.data:
            self.update_status("No data to save.")
            return

        data_np = np.array(self.data)
        if data_np.shape[0] == 0 or data_np.shape[1] < 9: # Check for expected number of columns
            self.update_status("No data to save after numpy conversion.")
            return

        # Calculate average power for the filename
        try:
            # Voltage is at index 1, Current is at index 2
            average_power = np.nanmean(data_np[:, 1].astype(float) * data_np[:, 2].astype(float))
        except (IndexError, ValueError) as e:
            print(f"Could not calculate average power for filename: {e}")
            average_power = 0.0 # Default if calculation fails

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Define subdirectory for saving data
        directory = "data_collection/data" 
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename_base = f"{timestamp}_{self.steak_type_var.get()}_{self.elapsed_time}S_{average_power:.2f}W.csv"
        filename = os.path.join(directory, filename_base)
        
        try:
            # Prepare data for saving, including flip events
            header = "Time,voltage,current,TH,Ttest,TM_1,TM_2,TM_3,Control_Mode,FlipEvent"
            
            # Create a new column for flip events, initialized to 0
            flip_column = np.zeros((data_np.shape[0], 1))
            
            # Mark flip events in the new column
            # Convert recorded flip times (which are elapsed_time) to approximate row indices
            # This assumes self.sample_interval is the time step for each row in data_np
            if self.flip_event_times:
                time_column_float = data_np[:, 0].astype(float)
                for t_flip in self.flip_event_times:
                    # Find the closest row index for the flip time
                    # This can be tricky if sample_interval isn't perfectly regular or if t_flip doesn't align
                    # A more robust way is to find the index where data_np[:, 0] is closest to t_flip
                    try:
                        # Find the index of the time value closest to t_flip
                        closest_time_index = np.abs(time_column_float - t_flip).argmin()
                        flip_column[closest_time_index, 0] = 1 # Mark as 1 for flip
                    except IndexError:
                        print(f"Warning: Could not accurately map flip time {t_flip}s to data row.")
            
            # Concatenate the flip column to the main data
            data_to_save = np.concatenate((data_np, flip_column), axis=1)
            
            # Define custom format for columns, ensuring Control_Mode is string and FlipEvent is integer
            # Original 8 columns (Time, V, I, T1-T5) are float, Control_Mode is string, FlipEvent is int
            # We need to handle the string column. np.savetxt is not ideal for mixed types.
            # Using pandas DataFrame to_csv is more robust for mixed types.
            import pandas as pd
            df_to_save = pd.DataFrame(data_to_save, columns=header.split(','))
            df_to_save['FlipEvent'] = df_to_save['FlipEvent'].astype(float).astype(int) # Ensure FlipEvent is int

            df_to_save.to_csv(filename, index=False, float_format='%.5f')
            self.update_status(f"Data (including flips) saved to {filename}")

            # Plotting (using data_np for numerical plotting, as it's already numerical)
            # Ensure data_np is converted to float for plotting if it contains strings
            data_for_plot = data_np[:, :8].astype(float) # Take only numerical columns for plotting

            plt.figure(figsize=(12, 9)) # Adjusted figure size

            plt.subplot(2, 1, 1)
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 1], label="Voltage (V)")
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 2] * 10, label="Current (A x10)") # Assuming current is multiplied by 10 for plotting
            plt.plot(data_for_plot[:, 0], np.multiply(data_for_plot[:, 1], data_for_plot[:, 2]), label="Power (W)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.title("Voltage, Current, and Power")
            plt.legend()
            plt.grid(True)
            # Add vertical lines for flip events
            for t_flip in self.flip_event_times:
                plt.axvline(x=t_flip, color='r', linestyle='--', linewidth=0.8, label='Flip' if t_flip == self.flip_event_times[0] else None)

            plt.subplot(2, 1, 2)
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 3], label="TH (°C)")
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 4], label="Ttest (°C)")
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 5], label="TM_1 (°C)")
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 6], label="TM_2 (°C)")
            plt.plot(data_for_plot[:, 0], data_for_plot[:, 7], label="TM_3 (°C)")
            plt.xlabel("Time (s)")
            plt.ylabel("Temperature (°C)")
            plt.title("Temperatures")
            plt.legend()
            plt.grid(True)
            for t_flip in self.flip_event_times:
                plt.axvline(x=t_flip, color='r', linestyle='--', linewidth=0.8, label='Flip' if t_flip == self.flip_event_times[0] else None)

            plt.tight_layout()
            plt.show() # This will block until the plot window is closed.
        except Exception as e:
            error_msg = f"Error during saving or plotting: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Save/Plot Error", error_msg)


    def close_devices(self):
        """
        Closes connections to all devices (Arduino, meters) and stops related threads.
        """
        self.update_status("Closing device connections...")

        # Stop preheat thread if active
        if self.preheating_active:
            self.stop_preheat() # This will also set PWM to 0
        if self.power_pid_active:
            self.stop_power_pid()
        if self.corrected_direct_power_active:
            self.stop_corrected_direct_power()

        # Stop pre-recording display thread first
        if self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive():
            # print("[UI] Stopping pre-recording display thread from close_devices...") # Debug
            self.stop_pre_recording_display_event.set()
            self.pre_recording_display_thread.join(timeout=2.0)
            if self.pre_recording_display_thread.is_alive():
                print("[UI] WARNING: Pre-recording display thread did not terminate in time during close_devices!")
        self.pre_recording_display_thread = None

        if self.arduino:
            try:
                if self.arduino.running: # Check if it was running before trying to send command
                    # Explicitly set logical PWM to 0 considering UI inversion
                    logical_pwm_off = 0 # This is the 0-255 value for "off" before inversion
                    pwm_signal_to_send_off = self._apply_pwm_inversion(logical_pwm_off)
                    self.arduino.control_arduino(pwm_signal_to_send_off)
                    self.current_pwm_setting_0_255 = 0 # Update internal state
                    self.pwm_percentage_var.set(0)    # Update UI var
                    if hasattr(self, 'pwm_label_display') and self.pwm_label_display.winfo_exists():
                        self.pwm_label_display.config(text="PWM: 0%")
                    print(f"[UI_CloseDevices] Logical PWM set to 0 (Sent to Arduino: {pwm_signal_to_send_off}) before closing port.")
                
                self.arduino.close()
                self.update_status("Arduino disconnected.")
            except Exception as e:
                self.update_status(f"Error closing Arduino: {e}")
            self.arduino = None
        if self.voltage_meter:
            try:
                self.voltage_meter.close()
                self.update_status("Voltage meter disconnected.")
            except Exception as e:
                self.update_status(f"Error closing voltage meter: {e}")
            self.voltage_meter = None
        if self.current_meter:
            try:
                self.current_meter.close()
                self.update_status("Current meter disconnected.")
            except Exception as e:
                self.update_status(f"Error closing current meter: {e}")
            self.current_meter = None
        self.update_status("All specified devices closed.")
        self.reset_realtime_display()

        # Disable control mode radio buttons and all mode-specific controls
        self.manual_pwm_radio.config(state=tk.DISABLED)
        self.pid_preheat_radio.config(state=tk.DISABLED)
        self.pid_power_radio.config(state=tk.DISABLED)
        self.corrected_direct_power_radio.config(state=tk.DISABLED)
        self.corrected_power_button.config(state=tk.DISABLED) # Disable new button
        self.corrected_power_target_entry.config(state=tk.DISABLED) # Disable new entry
        self.eq_A_entry.config(state=tk.DISABLED)
        self.eq_B_entry.config(state=tk.DISABLED)
        self.eq_C_entry.config(state=tk.DISABLED) # Disable Param C entry
        self.rds_on_entry.config(state=tk.DISABLED) # Disable Rds(on) entry
        self.nominal_supply_voltage_entry.config(state=tk.DISABLED) # Disable V_supply_max_pwm entry
        if hasattr(self, 'pwm_invert_checkbox'): self.pwm_invert_checkbox.config(state=tk.DISABLED)

        self.control_mode_var.set("MANUAL_PWM") # Reset to default
        self.emergency_stop_button.config(state=tk.DISABLED)
        
        # Re-enable port/address entries as devices are closed
        self.arduino_port_combobox.config(state='readonly')
        self.voltage_meter_address_combobox.config(state='readonly')
        self.current_meter_address_combobox.config(state='readonly')
        self.connect_button.config(state=tk.NORMAL) # Re-enable connect button
        self._handle_control_mode_change() # This will disable all specific controls based on new default

    def emergency_stop(self):
        """Immediately halts all heating, PID loops, and recording."""
        self.update_status("EMERGENCY STOP ACTIVATED!")

        # 1. Stop PID Loops (ensure controllers are stopped first)
        if self.preheating_active:
            self.stop_preheat() # This also sets PWM to 0 and updates its UI
        if self.power_pid_active:
            self.stop_power_pid() # This also sets PWM to 0 and updates its UI
        if self.corrected_direct_power_active:
            self.stop_corrected_direct_power()

        # 2. Stop Heating: In debug mode, just send 0. In normal mode, verify power is off.
        if self.debug_mode_var.get():
            self.update_status("E-STOP: [DEBUG] Setting PWM to 0.")
            if self.arduino and self.arduino.running:
                self.arduino.control_arduino(self._apply_pwm_inversion(0))
                self.pwm_percentage_var.set(0)
                if hasattr(self, 'pwm_label_display') and self.pwm_label_display.winfo_exists():
                    self.pwm_label_display.config(text="PWM: 0%")
        elif self.arduino and self.arduino.running:
            self.update_status("E-STOP: Attempting to set PWM to zero and verify power...")
            
            power_confirmed_zero = False

            for attempt in range(self.MAX_ZEROING_ATTEMPTS):
                if not (self.arduino and self.arduino.running):
                    self.update_status("E-STOP: Arduino disconnected during zeroing attempt. Aborting further PWM changes.")
                    break

                # Send logical PWM 0 command
                logical_pwm_value_for_zero = 0
                pwm_signal_to_send = self._apply_pwm_inversion(logical_pwm_value_for_zero)
                self.arduino.control_arduino(pwm_signal_to_send)
                
                # Update internal state and UI to reflect logical 0% PWM
                self.current_pwm_setting_0_255 = logical_pwm_value_for_zero
                self.pwm_percentage_var.set(0)
                if hasattr(self, 'pwm_label_display') and self.pwm_label_display.winfo_exists():
                    self.pwm_label_display.config(text="PWM: 0%")
                
                self.update_status(f"E-STOP: Attempt {attempt + 1}/{self.MAX_ZEROING_ATTEMPTS} - Sent logical PWM 0 (Actual: {pwm_signal_to_send}). Checking power...")
                time.sleep(0.75)  # Allow system to stabilize and meters to respond

                measured_power = float('inf')
                voltage_val, current_val = None, None
                power_read_successful = False

                if self.voltage_meter and self.current_meter:
                    try:
                        voltage_list = self.voltage_meter.read_voltage()
                        if voltage_list: voltage_val = voltage_list[0]

                        current_list = self.current_meter.read_current()
                        if current_list: current_val = current_list[0]

                        if voltage_val is not None and current_val is not None:
                            measured_power = float(voltage_val) * float(current_val)
                            power_read_successful = True
                            self.update_status(f"E-STOP: Attempt {attempt + 1} - Measured power: {measured_power:.2f}W")
                        else:
                            self.update_status(f"E-STOP: Attempt {attempt + 1} - Could not get valid V/I readings.")
                    except pyvisa.errors.VisaIOError as ve:
                        self.update_status(f"E-STOP: Attempt {attempt + 1} - VISA error reading power: {ve}")
                    except Exception as e:
                        self.update_status(f"E-STOP: Attempt {attempt + 1} - Error reading power: {e}")
                else:
                    self.update_status(f"E-STOP: Attempt {attempt + 1} - Meters not available to check power.")

                if power_read_successful and measured_power < self.POWER_ZERO_THRESHOLD:
                    self.update_status(f"E-STOP: Power confirmed near zero ({measured_power:.2f}W). PWM inversion is {self.invert_pwm_var.get()}.")
                    power_confirmed_zero = True
                    break  # Power is off
                else:
                    if attempt < self.MAX_ZEROING_ATTEMPTS - 1:
                        current_inversion_state = self.invert_pwm_var.get()
                        new_inversion_state = not current_inversion_state
                        self.invert_pwm_var.set(new_inversion_state) # This will also update the checkbox
                        self.update_status(f"E-STOP: Power not zero ({measured_power:.2f}W). Flipped PWM inversion to {new_inversion_state}. Retrying...")
                    # If it's the last attempt, the loop will end, and the final warning will be shown outside.
            
            if not power_confirmed_zero:
                warning_message = f"E-STOP: Power NOT confirmed zero after {self.MAX_ZEROING_ATTEMPTS} attempts. "
                if power_read_successful:
                    warning_message += f"Last measured: {measured_power:.2f}W. "
                else:
                    warning_message += "Could not read power. "
                warning_message += f"Final PWM inversion state: {self.invert_pwm_var.get()}. Please check system manually!"
                self.update_status(warning_message)
                messagebox.showwarning("E-Stop Warning", warning_message)

        elif not (self.arduino and self.arduino.running):
            self.update_status("E-STOP: Arduino not connected. Cannot send PWM commands or verify power via PWM.")

        # 3. Handle Recording (if active)
        was_recording = self.recording
        self.recording = False # Signal data collection & UI timer threads to stop

        if was_recording:
            # Wait for data collection thread to finish if it was running
            if self.data_collection_thread and self.data_collection_thread.is_alive():
                self.data_collection_thread.join(timeout=3.0) # Shorter timeout for E-Stop
                if self.data_collection_thread.is_alive():
                    print("[UI] WARNING: Data collection thread did not terminate cleanly after E-Stop.")
            
            # Save data if recording was active and data exists
            if self.data:
                self.save_and_plot_data()
            
            self.elapsed_time = 0 # Reset elapsed time
            self.update_time_label() # Update UI

        # 4. Determine connection status *after* all stop actions
        is_connected = self.arduino and self.arduino.running
        controls_state = tk.NORMAL if is_connected else tk.DISABLED
        radio_button_state = tk.NORMAL if is_connected else tk.DISABLED

        # 5. Reset UI elements related to recording and general controls
        self.stop_button.config(state=tk.DISABLED)
        self.flip_button.config(state=tk.DISABLED)
        self.steak_type_entry.config(state=tk.NORMAL) # Allow editing for next run

        # Set state of control mode radio buttons based on connection status
        radio_button_names = [
            'manual_pwm_radio', 'pid_preheat_radio', 'pid_power_radio',
            'corrected_direct_power_radio'
        ]
        for name in radio_button_names:
            widget = getattr(self, name, None)
            if widget: # Check if the attribute exists and is a widget
                widget.config(state=radio_button_state)

        # Other buttons' states will be set by _handle_control_mode_change based on actual connection.
        self.start_button.config(state=controls_state)
        self.connect_button.config(state=tk.DISABLED if is_connected else tk.NORMAL)

        # Port/address entries state
        port_combobox_state = tk.DISABLED if is_connected else 'readonly'
        self.arduino_port_combobox.config(state=port_combobox_state)
        self.voltage_meter_address_combobox.config(state=port_combobox_state)
        self.current_meter_address_combobox.config(state=port_combobox_state)
        
        self._handle_control_mode_change() # Refresh states of live control panels

        # 5. Restart pre-recording display if devices are still connected
        if is_connected:
            self.stop_pre_recording_display_event.clear() # Ensure event is clear
            if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                self.pre_recording_display_thread.start()
        else:
            self.reset_realtime_display()

        self.update_status("EMERGENCY STOP COMPLETE. System idle. Check connections if needed.")
        
    def on_closing(self):
        """Handles the event when the main UI window is closed."""
        if self.recording:
            if messagebox.askyesno("Confirm Exit", "Recording is in progress. Stop recording and exit?"):
                self.stop_recording() # This will also join the data_collection_thread
            else:
                return # User chose not to exit

        self.emergency_stop() # Perform a full stop before closing

        self.close_devices() # This will also handle stopping the pre-recording display thread
        try:
            plt.close('all') # Close all Matplotlib figures
        except Exception as e:
            print(f"Error closing matplotlib figures: {e}")
            
        self.master.destroy()

if __name__ == "__main__":
    print("Start")
    root = tk.Tk()
    app = DataCollectionUI(root)
    root.mainloop()
