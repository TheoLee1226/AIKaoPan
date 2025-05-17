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
import serial

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

    def update(self, current_value):
        now = time.time()
        time_change = now - self._last_time

        # print(f"[PID_UPDATE] Called. current_value={current_value}, time_change={time_change:.4f}s, sample_time={self.sample_time}")

        if current_value is None: # Cannot compute if current_value is invalid
            # print("[PID_UPDATE] current_value is None, returning last output.")
            return self.output # Return last known good output or 0

        if time_change >= self.sample_time:
            # print(f"[PID_UPDATE] Time to update. Last update was {time_change:.4f}s ago.")
            error = float(self.setpoint) - float(current_value)
            print(f"[PID_UPDATE] Setpoint={self.setpoint}, CurrentValue={current_value}, Error={error:.4f}")

            # Proportional term
            p_term = self.Kp * error
            # print(f"[PID_UPDATE] P_Term (Kp={self.Kp} * Error={error:.4f}) = {p_term:.4f}")

            # Integral term (with anti-windup)
            integral_change = self.Ki * error * time_change # Use actual time_change
            self._integral += self.Ki * error * time_change # Use actual time_change
            # print(f"[PID_UPDATE] Integral_Change (Ki={self.Ki} * Error={error:.4f} * TC={time_change:.4f}) = {integral_change:.4f}")
            # print(f"[PID_UPDATE] Integral before clamp: {self._integral:.4f}")
            self._integral = max(self.integral_min, min(self._integral, self.integral_max))
            # print(f"[PID_UPDATE] Integral after clamp ({self.integral_min}, {self.integral_max}): {self._integral:.4f}")
            i_term = self._integral
            # print(f"[PID_UPDATE] I_Term = {i_term:.4f}")

            # Derivative term (on measurement to reduce derivative kick)
            input_change = float(current_value) - self._last_input
            # print(f"[PID_UPDATE] InputChange (Current={current_value} - LastInput={self._last_input}) = {input_change:.4f}")
            d_term = 0.0
            if time_change > 0:
                d_term = -self.Kd * (input_change / time_change)
                # print(f"[PID_UPDATE] D_Term (-Kd={-self.Kd} * InChg={input_change:.4f} / TC={time_change:.4f}) = {d_term:.4f}")
            else:
                # print(f"[PID_UPDATE] D_Term = 0 (time_change <= 0)")
                pass # Added pass to make the else block valid
            self.output = p_term + i_term + d_term
            # print(f"[PID_UPDATE] Output before clamp (P={p_term:.4f} + I={i_term:.4f} + D={d_term:.4f}) = {self.output:.4f}")
            self.output = max(self.output_min, min(self.output, self.output_max))
            # print(f"[PID_UPDATE] Output after clamp ({self.output_min}, {self.output_max}): {self.output:.4f}")

            self._last_error = error
            self._last_time = now
            self._last_input = float(current_value)
            return self.output
        # print(f"[PID_UPDATE] Not time to update yet (time_change={time_change:.4f}s < sample_time={self.sample_time}). Returning last output: {self.output}")
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
    VOLTAGE_METER_ADDRESS = "ASRL11::INSTR"
    CURRENT_METER_ADDRESS = "ASRL4::INSTR"
    ARDUINO_COM_PORT = "COM3"
    ARDUINO_BAUDRATE = 9600

    # --- Default UI Values ---
    DEFAULT_STEAK_TYPE = "test_steak"
    DEFAULT_PWM_PERCENTAGE = 0
    DEFAULT_PID_SETPOINT = 60.0 # Target temperature for preheat (Temperature PID - Unchanged)
    DEFAULT_PID_KP = 10.0
    DEFAULT_PID_KI = 0.1
    DEFAULT_PID_KD = 0.5
    DEFAULT_POWER_PID_SETPOINT = 20.0 # Target Watts (Power PID)
    DEFAULT_POWER_PID_KP = 4.5      # Slightly reduced Kp
    DEFAULT_POWER_PID_KI = 0.3     # Slightly reduced Ki
    DEFAULT_POWER_PID_KD = 0.1     # Significantly increased Kd to counteract undershoot
    DEFAULT_R_REF = 1.0 # Default reference resistance if file not loaded
    NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM = 24.0 # Assumed V_out at PWM=255. Adjust as needed.    
    DEFAULT_EQ_PARAM_A = 22.0 # Example: R0 for R(T) = A * exp(B*T)
    DEFAULT_EQ_PARAM_B = 0.0039 # Example: temperature coefficient for copper/nichrome like
    DEFAULT_EQ_PARAM_C = 0.0 # Example: Constant offset for R(T) = A * exp(B*T) + C
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
        self.recording_control_mode_var = tk.StringVar(value="MANUAL_PWM") 
        self.active_recording_control_mode = "MANUAL_PWM" # Stores the mode active when recording started        
        # Removed self.temp_resist_filename_var
        self.pwm_percentage_var = tk.IntVar(value=self.DEFAULT_PWM_PERCENTAGE)
        self.steak_type_var = tk.StringVar(value=self.DEFAULT_STEAK_TYPE)

        # Internal state for PWM
        self.current_pwm_setting_0_255 = self._calculate_pwm_actual(self.pwm_percentage_var.get()) # Actual PWM value (0-255) for Arduino

        # Configuration
        self.pre_recording_display_interval = 0.5 # seconds for display update when not recording or preheating
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
        self._setup_recording_control_mode_selection() # New: Recording Control Mode
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
        self._handle_recording_control_mode_change() # Set initial UI state for recording controls
        self.update_status("Disconnected. Press 'Connect Devices'.")

    # --- UI Setup Helper Methods ---
    def _setup_connection_controls(self):
        """Sets up the device connection button."""
        # Connection Frame
        self.connection_frame = ttk.LabelFrame(self.left_column_frame, text="Device Connection")
        self.connection_frame.pack(pady=(0, 5), padx=5, fill="x")
        self.connect_button = tk.Button(self.connection_frame, text="Connect Devices", command=self.connect_devices)
        self.connect_button.pack(pady=5, padx=5, fill="x", expand=True)

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

        self.set_pwm_zero_button = ttk.Button(self.pwm_frame, text="Set to 0", command=self.set_pwm_to_zero, state=tk.DISABLED)
        self.set_pwm_zero_button.grid(row=0, column=4, padx=5, pady=3, sticky="ew")

        # Configure column weights for pwm_frame grid to allow expansion if needed
        self.pwm_frame.grid_columnconfigure(0, weight=0) # Label column
        self.pwm_frame.grid_columnconfigure(1, weight=0) # Entry/Button column
        self.pwm_frame.grid_columnconfigure(2, weight=0) # Range Label/Button column
        self.pwm_frame.grid_columnconfigure(3, weight=1) # Button column
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
        
        # Button and Status Label to the right, similar to PID control blocks
        self.corrected_power_button = ttk.Button(self.res_eq_frame, text="Start Corr.Pwr", command=self.toggle_corrected_direct_power, state=tk.DISABLED)
        self.corrected_power_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky="ewns") # Span 1 row

        self.corrected_power_status_label = tk.Label(self.res_eq_frame, text="Corrected Pwr: Off", wraplength=220, justify=tk.LEFT) # Increased wraplength
        self.corrected_power_status_label.grid(row=1, column=2, rowspan=3, padx=10, pady=2, sticky="nsew") # Starts at row 1, spans 3 rows
        self.res_eq_frame.grid_columnconfigure(2, weight=1)


    def _setup_power_pid_controls(self):
        """Sets up the PID power control elements."""
        self.power_pid_frame = ttk.LabelFrame(self.middle_column_frame, text="PID Power Control (Watts)")
        self.power_pid_frame.pack(pady=5, padx=5, fill="x")

        # Row 0: Target Power
        ttk.Label(self.power_pid_frame, text="Target Power (W):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_setpoint_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_setpoint_var, width=7)
        self.power_pid_setpoint_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 1: Kp
        ttk.Label(self.power_pid_frame, text="Kp:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_kp_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_kp_var, width=7)
        self.power_pid_kp_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 2: Ki
        ttk.Label(self.power_pid_frame, text="Ki:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_ki_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_ki_var, width=7)
        self.power_pid_ki_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)

        # Row 3: Kd
        ttk.Label(self.power_pid_frame, text="Kd:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.power_pid_kd_entry = ttk.Entry(self.power_pid_frame, textvariable=self.power_pid_kd_var, width=7)
        self.power_pid_kd_entry.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)

        # Column 2: Button and Status
        self.power_pid_button = ttk.Button(self.power_pid_frame, text="Start Power Ctrl", command=self.toggle_power_pid, state=tk.DISABLED)
        self.power_pid_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky="ewns") 
        
        self.power_pid_status_label = tk.Label(self.power_pid_frame, text="Power PID Status: Off", wraplength=200, justify=tk.LEFT)
        self.power_pid_status_label.grid(row=1, column=2, rowspan=3, padx=10, pady=2, sticky="nsew") # Starts at row 1, spans 3 rows

        self.power_pid_frame.grid_columnconfigure(0, weight=0)
        self.power_pid_frame.grid_columnconfigure(1, weight=0)
        self.power_pid_frame.grid_columnconfigure(2, weight=1)

    def _setup_recording_control_mode_selection(self):
        """Sets up radio buttons for selecting control mode during recording."""
        self.recording_mode_frame = ttk.LabelFrame(self.right_column_frame, text="Recording Control Mode")
        self.recording_mode_frame.pack(pady=(0, 5), padx=5, fill="x")

        radio_button_container = ttk.Frame(self.recording_mode_frame)
        radio_button_container.pack(pady=2)

        self.rec_manual_pwm_radio = ttk.Radiobutton(radio_button_container, text="Manual PWM", variable=self.recording_control_mode_var, # Default
                                                 value="MANUAL_PWM", command=self._handle_recording_control_mode_change, state=tk.DISABLED)
        self.rec_manual_pwm_radio.pack(side=tk.LEFT, padx=5, pady=2)

        self.rec_pid_power_radio = ttk.Radiobutton(radio_button_container, text="PID Power", variable=self.recording_control_mode_var,
                                                value="PID_POWER", command=self._handle_recording_control_mode_change, state=tk.DISABLED)
        self.rec_pid_power_radio.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.rec_corrected_direct_power_radio = ttk.Radiobutton(radio_button_container, text="Corrected Power (Eqn)", variable=self.recording_control_mode_var,
                                                        value="CORRECTED_DIRECT_POWER", command=self._handle_recording_control_mode_change, state=tk.DISABLED)
        self.rec_corrected_direct_power_radio.pack(side=tk.LEFT, padx=5, pady=2)

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

        # Stop any active PID controllers if switching away from them
        if selected_mode != "PID_PREHEAT" and self.preheating_active:
            self.stop_preheat()
        if selected_mode != "PID_POWER" and self.power_pid_active:
            self.stop_power_pid()
        if selected_mode != "CORRECTED_DIRECT_POWER" and self.corrected_direct_power_active:
            self.stop_corrected_direct_power()

        # --- Enable/Disable Manual PWM Controls ---
        is_manual_mode = (selected_mode == "MANUAL_PWM")
        manual_pwm_button_state = tk.NORMAL if is_manual_mode and self.arduino and self.arduino.running else tk.DISABLED
        # PWM entry is always enabled if connected, buttons depend on mode
        self.pwm_entry.config(state=tk.NORMAL if self.arduino and self.arduino.running else tk.DISABLED)
        self.set_pwm_button.config(state=manual_pwm_button_state)
        self.set_pwm_zero_button.config(state=manual_pwm_button_state)

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

    def _handle_recording_control_mode_change(self, event=None):
        """Handles changes in the recording control mode selection."""
        selected_recording_mode = self.recording_control_mode_var.get()
        # This function currently doesn't need to do much beyond updating a variable,
        # as the actual mode switch happens in start_recording.
        # self.update_status(f"Recording mode set to: {selected_recording_mode}") # Optional: for debugging
    # Removed Temperature-Resistance File Handling methods (load_temp_resistance_file, get_resistance_at_temp)

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

        if self.control_mode_var.get() != "MANUAL_PWM":
            self.control_mode_var.set("MANUAL_PWM")
            self._handle_control_mode_change() # This will stop other PIDs

        if self.arduino and self.arduino.running:
            # PIDs are already stopped by _handle_control_mode_change if mode was switched

            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(percentage)
            self.pwm_label_display.config(text=f"PWM: {percentage}%") # Update label
            self.arduino.control_arduino(self.current_pwm_setting_0_255)
            self.update_status(f"PWM set to {self.pwm_percentage_var.get()}% ({self.current_pwm_setting_0_255})")
        else:
            messagebox.showwarning("PWM Error", "Arduino not connected. Cannot send PWM command.")

    def set_pwm_to_zero(self):
        """
        Sets the PWM value to 0 and sends the command to Arduino.
        Also stops PID preheat if it's active.
        """
        if self.control_mode_var.get() != "MANUAL_PWM":
            self.control_mode_var.set("MANUAL_PWM")
            self._handle_control_mode_change() # This will stop other PIDs

        self.pwm_percentage_var.set(0) # Update the Tkinter variable, which updates the Entry box

        if self.arduino and self.arduino.running:
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(0)
            self.pwm_label_display.config(text=f"PWM: 0%") # Update label
            self.arduino.control_arduino(self.current_pwm_setting_0_255) # Send 0 to Arduino
            self.update_status(f"PWM set to 0% (0)")
        else:
            # Still update local state even if not connected, so UI reflects 0
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(0)
            self.pwm_label_display.config(text=f"PWM: 0%")
            self.update_status("PWM set to 0% (Arduino not connected).")

    def reset_realtime_display(self):
        """Resets all real-time data display labels to their default '---' state."""
        self.update_realtime_display(voltage=None, current=None, temps=[None]*5)

    def update_status(self, message):
        """Updates the status bar message and prints it to the console."""
        self.status_label.config(text=f"Status: {message}")
        print(message) # Also print to console

    def connect_devices(self):
        """
        Attempts to connect to the voltage meter, current meter, and Arduino.
        Updates UI based on connection status.
        """
        self.update_status("Connecting...")
        try:
            # Connect to Voltage Meter
            self.update_status(f"Connecting to Voltage Meter ({self.VOLTAGE_METER_ADDRESS})...")
            self.voltage_meter = cm.control_meter()
            self.voltage_meter.connect(self.VOLTAGE_METER_ADDRESS)
            self.voltage_meter.set_voltage_mode()
            self.update_status("Voltage Meter connected.")
            time.sleep(1)

            # Connect to Current Meter
            self.update_status(f"Connecting to Current Meter ({self.CURRENT_METER_ADDRESS})...")
            self.current_meter = cm.control_meter()
            self.current_meter.connect(self.CURRENT_METER_ADDRESS)
            self.current_meter.set_current_mode()
            self.update_status("Current Meter connected.")
            time.sleep(1)
            
            # Connect to Arduino
            self.update_status(f"Connecting to Arduino ({self.ARDUINO_COM_PORT})...")
            self.arduino = ca.control_arduino(self.ARDUINO_COM_PORT, self.ARDUINO_BAUDRATE)
            if not self.arduino.running:
                self.arduino = None # Ensure it's None if connection truly failed
                raise Exception("Failed to connect to Arduino or start its read thread.")
            self.update_status("Arduino connected. Initializing live display...")

            # Start pre-recording display now that Arduino is up
            self.stop_pre_recording_display_event.clear()
            if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                self.pre_recording_display_thread.start()

            # Enable control mode radio buttons
            self.manual_pwm_radio.config(state=tk.NORMAL)
            self.pid_preheat_radio.config(state=tk.NORMAL)
            self.pid_power_radio.config(state=tk.NORMAL)
            self.corrected_direct_power_radio.config(state=tk.NORMAL)
            # Equation parameter entries are enabled by _handle_control_mode_change
            # self.load_tr_file_button.config(state=tk.NORMAL) # Removed

            # Enable recording control mode radio buttons
            self.rec_manual_pwm_radio.config(state=tk.NORMAL)
            self.rec_corrected_direct_power_radio.config(state=tk.NORMAL)
            self.rec_pid_power_radio.config(state=tk.NORMAL)

            self._handle_control_mode_change() # Update UI based on current (default) mode
            self.emergency_stop_button.config(state=tk.NORMAL) # Enable E-Stop
            time.sleep(1) # Short delay to allow first data to potentially arrive for display

            self.start_button.config(state=tk.NORMAL)
            self.connect_button.config(state=tk.DISABLED)
            self.update_status("Devices connected. Live display active. Ready to record.")

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

                if self.voltage_meter:
                    voltage_list = self.voltage_meter.read_voltage()
                    if voltage_list: voltage_val = voltage_list[0]
                
                if self.current_meter:
                    current_list = self.current_meter.read_current()
                    if current_list: current_val = current_list[0]

                if not self.stop_pre_recording_display_event.is_set(): # Check again before UI update
                    self.master.after(0, self.update_realtime_display, voltage_val, current_val, temps_to_display)

            except pyvisa.errors.VisaIOError as ve:
                print(f"[PreRecDisp] VISA Error during pre-recording display: {ve}") # Log error, continue
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

        if self.recording:
            messagebox.showerror("Error", "Recording in progress. Please stop recording first.")
            return

        if self.control_mode_var.get() != "PID_PREHEAT":
            self.control_mode_var.set("PID_PREHEAT")
            self._handle_control_mode_change() # This will stop other PIDs/manual
            # UI for preheat controls will be enabled by _handle_control_mode_change
        
        # Ensure Corrected Direct Power is stopped if it was active
        if self.corrected_direct_power_active:
            self.stop_corrected_direct_power()

        if self.preheating_active:
            self.stop_preheat()
        else:
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

    def stop_preheat(self):
        """
        Stops the PID preheating thread, turns off the heater, and updates UI state.
        """
        if self.preheat_thread and self.preheat_thread.is_alive():
            self.stop_preheat_event.set()
            self.preheat_thread.join(timeout=2.0)
            if self.preheat_thread.is_alive():
                print("[UI] WARNING: Preheat thread did not terminate in time!")
        self.preheat_thread = None
        self.preheating_active = False

        # REMOVED: Do not automatically set PWM to 0 when stopping preheat.
        # if self.arduino and self.arduino.running:
        #     self.arduino.control_arduino(0) # Turn off heater by setting PWM to 0

        # UI updates are now handled by _handle_control_mode_change
        # Ensure the button text and status label are reset correctly if not switching mode
        if self.control_mode_var.get() == "PID_PREHEAT": # Only if still in this mode
            self.preheat_button.config(text="Start Preheat")
            self.preheat_status_label.config(text="Preheat Status: Off")
        self._handle_control_mode_change() # Refresh UI states

        self.update_status("PID Preheat stopped.")

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
                self.arduino.control_arduino(pwm_to_send)
                # print(f"[PreheatLoop] Sent PWM to Arduino: {pwm_to_send}")
                
                current_time = time.time()
                if current_time - last_ui_update_time >= ui_update_interval:
                    target_temp_str = f"Target ({selected_sensor_name}): {self.pid_setpoint_var.get():.1f}°C"
                    actual_temp_str = f"Actual ({selected_sensor_name}): {actual_sensor_temp if actual_sensor_temp is not None else 'N/A'}°C"
                    pwm_str = f"PID PWM: {int(round(pwm_output_float/255*100))}% ({pwm_to_send})"
                    status_text = f"{target_temp_str}\n{actual_temp_str}\n{pwm_str}" # Display on multiple lines
                    self.master.after(0, self.preheat_status_label.config, {"text": status_text})
                    last_ui_update_time = current_time # Corrected variable name

            time.sleep(self.pid_control_interval / 2) # Shorter sleep, PID internal timing handles the actual update rate
        print("[PreheatLoop] Preheat loop finished.")

    def toggle_power_pid(self):
        """Starts or stops the PID power control process."""
        if not (self.arduino and self.arduino.running and self.voltage_meter and self.current_meter):
            messagebox.showerror("Error", "Arduino or Meters not connected.")
            return

        if self.recording:
            messagebox.showerror("Error", "Recording in progress. Please stop recording first.")
            return

        if self.control_mode_var.get() != "PID_POWER":
            self.control_mode_var.set("PID_POWER")
            self._handle_control_mode_change()
        
        # Ensure Corrected Direct Power is stopped if it was active
        if self.corrected_direct_power_active:
            self.stop_corrected_direct_power()

        if self.power_pid_active:
            self.stop_power_pid()
        else:
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

    def stop_power_pid(self):
        """Stops the PID power control thread and turns off heater."""
        if self.power_pid_thread and self.power_pid_thread.is_alive():
            self.stop_power_pid_event.set()
            self.power_pid_thread.join(timeout=2.0)
            if self.power_pid_thread.is_alive():
                print("[UI] WARNING: Power PID thread did not terminate in time!")
        self.power_pid_thread = None
        self.power_pid_active = False

        # REMOVED: Do not automatically set PWM to 0 when stopping power PID.
        # if self.arduino and self.arduino.running:
        #     self.arduino.control_arduino(0) # Turn off heater

        if self.control_mode_var.get() == "PID_POWER": # Only if still in this mode
            self.power_pid_button.config(text="Start Power Ctrl")
            self.power_pid_status_label.config(text="Power PID Status: Off")
        self._handle_control_mode_change() # Refresh UI states
        self.update_status("PID Power Control stopped.")

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
                    initial_power_for_pid = voltage_list[0] * current_list[0]
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
            voltage_list = self.voltage_meter.read_voltage()
            current_list = self.current_meter.read_current()

            if voltage_list and current_list and voltage_list[0] is not None and current_list[0] is not None:
                current_power = voltage_list[0] * current_list[0]

            pwm_output_float = self.power_pid_controller.update(current_power)
            print(f"[PowerPIDLoop] Current Power: {current_power}, PID Output (0-255): {pwm_output_float}")

            if pwm_output_float is not None:
                pwm_to_send = int(round(pwm_output_float))
                self.arduino.control_arduino(pwm_to_send)
                # print(f"[PowerPIDLoop] Sent PWM to Arduino: {pwm_to_send}")

                if time.time() - last_ui_update_time >= ui_update_interval:
                    status_text = f"Target: {self.power_pid_setpoint_var.get():.1f}W\nActual: {current_power:.2f}W\nPID PWM: {int(round(pwm_output_float/255*100))}% ({pwm_to_send})"
                    self.master.after(0, self.power_pid_status_label.config, {"text": status_text})
                    last_ui_update_time = time.time()
            time.sleep(self.pid_control_interval / 2)
        print("[PowerPIDLoop] Power PID loop finished.")

    # --- Corrected Direct Power Methods ---
    def toggle_corrected_direct_power(self): # This is effectively handled by toggle_power_pid
        # This method is now called by the dedicated corrected_power_button
        if not (self.arduino and self.arduino.running and self.voltage_meter and self.current_meter):
            messagebox.showerror("Error", "Arduino or Meters not connected.")
            return
        if self.recording:
            messagebox.showerror("Error", "Recording in progress. Please stop recording first.")
            return
        
        # Ensure we are in the correct mode (though UI should enforce this)
        if self.control_mode_var.get() != "CORRECTED_DIRECT_POWER":
            self.control_mode_var.set("CORRECTED_DIRECT_POWER")
            self._handle_control_mode_change() # This will stop other PIDs

        if self.corrected_direct_power_active:
            self.stop_corrected_direct_power()
        else:
            self.start_corrected_direct_power()
    # The start_corrected_direct_power and stop_corrected_direct_power methods
    # will now update self.corrected_power_button and self.corrected_power_status_label

    def start_corrected_direct_power(self):
        try:
            target_power = self.corrected_power_target_var.get() # Use dedicated target var
            param_A = self.equation_param_A_var.get()
            param_B = self.equation_param_B_var.get() # param_B can be zero or negative
            param_C = self.equation_param_C_var.get() 

            if target_power < 0:
                messagebox.showwarning("Input Warning", "Target power should be non-negative. Using absolute value.")
                target_power = abs(target_power)
                self.corrected_power_target_var.set(target_power) # Update dedicated var
            
            # param_C can be any real number, so no specific validation here unless R_actual must be > 0
            if param_A <= 0: # A must be positive for R = A*exp(B*T)
                messagebox.showerror("Input Error", "Equation parameter A must be positive.")
                return

        except tk.TclError:
            messagebox.showerror("Input Error", "Invalid target power or equation parameter value(s).")
            return

        print(f"[UI_CorrectedDirectPwr] Starting. Target: {target_power}W, EqParams: A={param_A}, B={param_B}, C={param_C}")
        self.corrected_direct_power_active = True
        self.stop_corrected_direct_power_event.clear()

        self.corrected_direct_power_thread = threading.Thread(target=self.run_corrected_direct_power_loop, daemon=True)
        self.corrected_direct_power_thread.start()

        self.corrected_power_button.config(text="Stop Corr.Pwr") 
        self._handle_control_mode_change() # Refresh UI states (mainly for other controls)
        self.update_status(f"Corrected Direct Power Control started. Target: {target_power}W")

    def stop_corrected_direct_power(self):
        if self.corrected_direct_power_thread and self.corrected_direct_power_thread.is_alive():
            self.stop_corrected_direct_power_event.set()
            self.corrected_direct_power_thread.join(timeout=2.0)
            if self.corrected_direct_power_thread.is_alive():
                print("[UI] WARNING: Corrected Direct Power thread did not terminate in time!")
        self.corrected_direct_power_thread = None
        self.corrected_direct_power_active = False

        # Update dedicated button and status label
        self.corrected_power_button.config(text="Start Corr.Pwr") 
        self.corrected_power_status_label.config(text="Corrected Pwr: Off") 
        
        self._handle_control_mode_change() # Refresh UI states (mainly for other controls)
        self.update_status("Corrected Direct Power Control stopped.")

    def run_corrected_direct_power_loop(self):
        last_ui_update_time = time.time()
        ui_update_interval = self.pre_recording_display_interval
        target_power_w = self.corrected_power_target_var.get() # Use dedicated target var
        # Get equation parameters once at the start of the loop
        try:
            param_A = self.equation_param_A_var.get()
            param_B = self.equation_param_B_var.get()
            param_C = self.equation_param_C_var.get()
            if param_A <= 0: # Should have been caught by start, but good to re-check
                print("[CorrectedDirectPwrLoop] Error: Parameter A is not positive. Stopping loop.")
                self.master.after(0, self.stop_corrected_direct_power) # Schedule stop on main thread
                # self.corrected_power_button.config(text="Start Corr.Pwr") # Reset button on error
                return
        except tk.TclError:
            print("[CorrectedDirectPwrLoop] Error: Could not get equation parameters. Stopping loop.")
            self.master.after(0, self.stop_corrected_direct_power)
            return

        while not self.stop_corrected_direct_power_event.is_set() and self.arduino and self.arduino.running and self.voltage_meter and self.current_meter:
            voltage_list = self.voltage_meter.read_voltage()
            current_list = self.current_meter.read_current()
            temps_arduino = self.arduino.return_temperature() # TM_1 is at index 2
            
            measured_power, R_actual, pwm_final_sent = None, 0.0, 0 # R_actual default to 0 if not calculable

            if voltage_list and current_list: # We have V and I readings
                measured_power = voltage_list[0] * current_list[0]

            if temps_arduino.size == 5 and temps_arduino[2] is not None and temps_arduino[2] > -1: # Valid TM_1
                current_temp_for_R = temps_arduino[2] # Using TM_1
                R_actual = self.get_resistance_from_equation(current_temp_for_R, param_A, param_B, param_C)
                # print(f"[CorrectedDirectPwrLoop] Temp: {current_temp_for_R}, A: {param_A}, B: {param_B}, R_actual: {R_actual}") # Debug

                if R_actual is not None and R_actual > 0:
                    if target_power_w >= 0 and self.NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM > 0:
                        V_required = np.sqrt(target_power_w * R_actual)
                        pwm_calculated_float = (V_required / self.NOMINAL_SUPPLY_VOLTAGE_AT_MAX_PWM) * 255.0
                        pwm_final_sent = int(round(max(0, min(pwm_calculated_float, 255))))
                        self.arduino.control_arduino(pwm_final_sent)
                    else: # target power is negative, or V_supply is not set
                        self.arduino.control_arduino(0) 
                        pwm_final_sent = 0
                else: # R_actual could not be calculated or is not positive
                    self.arduino.control_arduino(0) 
                    pwm_final_sent = 0
            else: # No valid temperature for R_actual
                self.arduino.control_arduino(0) # Safety: turn off
                pwm_final_sent = 0

            if time.time() - last_ui_update_time >= ui_update_interval:
                # Prepare display strings
                measured_power_str = f"{measured_power:.1f}" if measured_power is not None else "--"
                
                temp_tm1_val = None
                if temps_arduino.size == 5 and temps_arduino[2] is not None and temps_arduino[2] > -1:
                    temp_tm1_val = temps_arduino[2]
                temp_tm1_str = f"{temp_tm1_val:.1f}" if temp_tm1_val is not None else "--"
                
                r_actual_str = "--"
                if R_actual is not None and R_actual > 0: # R_actual could be float('inf')
                    if R_actual == float('inf'):
                        r_actual_str = "inf"
                    else:
                        r_actual_str = f"{R_actual:.2f}"

                pwm_perc_str = f"{int(round(pwm_final_sent/255*100))}%"

                # Construct the 3-line status text
                line1 = f"P Tgt: {target_power_w:.1f}W Act: {measured_power_str}W"
                line2 = f"T1: {temp_tm1_str}°C R: {r_actual_str}Ω"
                line3 = f"PWM: {pwm_perc_str} ({pwm_final_sent})"
                status_text_to_display = f"{line1}\n{line2}\n{line3}"
                self.master.after(0, self.corrected_power_status_label.config, {"text": status_text_to_display})
                last_ui_update_time = time.time()
            time.sleep(self.pid_control_interval / 2)
        print("[CorrectedDirectPwrLoop] Corrected Direct Power loop finished.")

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
        
        self.active_recording_control_mode = self.recording_control_mode_var.get()
        self.update_status(f"Starting recording with mode: {self.active_recording_control_mode}")

        if self.active_recording_control_mode == "MANUAL_PWM":
            if self.preheating_active: self.stop_preheat()
            if self.power_pid_active: self.stop_power_pid()
            if self.corrected_direct_power_active: self.stop_corrected_direct_power()
            self.control_mode_var.set("MANUAL_PWM") # Sync live mode display
            try:
                percentage = self.pwm_percentage_var.get()
                if not (0 <= percentage <= 100):
                    messagebox.showerror("PWM Error", "Manual PWM for recording: Percentage must be 0-100.")
                    return
                self.current_pwm_setting_0_255 = self._calculate_pwm_actual(percentage)
                self.pwm_label_display.config(text=f"PWM: {percentage}%")
                if self.arduino and self.arduino.running:
                    self.arduino.control_arduino(self.current_pwm_setting_0_255)
                else:
                    messagebox.showerror("Error", "Arduino not connected for Manual PWM recording.")
                    return
            except tk.TclError:
                messagebox.showerror("PWM Error", "Manual PWM for recording: Invalid percentage.")
                return
        elif self.active_recording_control_mode == "PID_POWER":
            if self.preheating_active: self.stop_preheat()
            if self.corrected_direct_power_active: self.stop_corrected_direct_power()
            self.control_mode_var.set("PID_POWER") # Sync live mode display
            if not self.power_pid_active:
                self.start_power_pid()
            if not self.power_pid_active: # If start_power_pid failed
                messagebox.showerror("Error", "Failed to start PID Power Control for recording.")
                return
        elif self.active_recording_control_mode == "CORRECTED_DIRECT_POWER":            
            try: # Validate equation parameters before starting recording in this mode
                param_A_val = self.equation_param_A_var.get()
                _ = self.equation_param_C_var.get() # Just to check if it's a valid float
                if param_A_val <= 0:
                    messagebox.showerror("Error", "Corrected Direct Power: Equation parameter A must be positive.")
                    return
            except tk.TclError:
                messagebox.showerror("Error", "Corrected Direct Power: Invalid equation parameter(s).")
                return
            if self.preheating_active: self.stop_preheat()
            if self.power_pid_active: self.stop_power_pid() # Stop regular PID power if active
            self.control_mode_var.set("CORRECTED_DIRECT_POWER")
            if not self.corrected_direct_power_active:
                self.start_corrected_direct_power() 
            if not self.corrected_direct_power_active: # If start failed
                messagebox.showerror("Error", "Failed to start Corrected Direct Power Control for recording.")
                return
        # Removed PID_PREHEAT as a recording mode option
        # else: # Should not happen if UI is correctly configured
        #     messagebox.showerror("Error", f"Unknown recording control mode: {self.active_recording_control_mode}")
        #     return
        
        self._handle_control_mode_change() # Update UI based on the (possibly new) live mode

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
        self.connect_button.config(state=tk.DISABLED) # Disable connect during recording
        self.steak_type_entry.config(state=tk.DISABLED) # Disable steak type entry during recording
        
        # Disable all live control inputs and mode selections
        self.manual_pwm_radio.config(state=tk.DISABLED)
        self.pid_preheat_radio.config(state=tk.DISABLED)
        self.pid_power_radio.config(state=tk.DISABLED)
        self.corrected_direct_power_radio.config(state=tk.DISABLED)
        self.pwm_entry.config(state=tk.DISABLED)
        self.set_pwm_button.config(state=tk.DISABLED)
        self.set_pwm_zero_button.config(state=tk.DISABLED)
        # PID controls are already handled by _handle_control_mode_change, but ensure they are off if not the active recording mode
        self.preheat_button.config(state=tk.DISABLED)
        self.power_pid_button.config(state=tk.DISABLED) # Standard PID power button
        self.corrected_power_button.config(state=tk.DISABLED) # New Corrected power button
        # Corrected Direct Power button (reused power_pid_button) would also be disabled by _handle_control_mode_change
        self.rec_manual_pwm_radio.config(state=tk.DISABLED) 
        # self.rec_pid_preheat_radio.config(state=tk.DISABLED) # Removed
        self.rec_pid_power_radio.config(state=tk.DISABLED) 

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
        
        # If manual PWM was used for recording, set PWM to 0. PID modes continue.
        if self.active_recording_control_mode == "MANUAL_PWM":
            if self.arduino and self.arduino.running:
                self.arduino.control_arduino(0)
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

        # Re-enable control mode radio buttons if devices are connected
        radio_state = tk.NORMAL if self.arduino and self.arduino.running else tk.DISABLED
        self.manual_pwm_radio.config(state=radio_state)
        self.pid_preheat_radio.config(state=radio_state)
        self.pid_power_radio.config(state=radio_state)
        self.corrected_direct_power_radio.config(state=radio_state)

        # Re-enable recording control mode radio buttons if devices are connected
        self.rec_manual_pwm_radio.config(state=radio_state)
        self.rec_corrected_direct_power_radio.config(state=radio_state)
        self.rec_pid_power_radio.config(state=radio_state)
        
        self._handle_control_mode_change() # Update UI elements based on current mode

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
                if self.active_recording_control_mode == "MANUAL_PWM":
                    if self.arduino and self.arduino.running:
                         self.arduino.control_arduino(self.current_pwm_setting_0_255) # Send current PWM value
                voltage_list = self.voltage_meter.read_voltage()
                current_list = self.current_meter.read_current()
                temperature_array = self.arduino.return_temperature() # numpy array

                if voltage_list and current_list and temperature_array.size == 5:
                    # Assuming read_voltage/current return list of one float
                    voltage_val = voltage_list[0]
                    current_val = current_list[0]
                    
                    self.data.append([
                        current_loop_time - loop_start_time,
                        voltage_val,
                        current_val,
                        temperature_array[0], # TH
                        temperature_array[1], # Ttest
                        temperature_array[2], # TM_1
                        temperature_array[3], # TM_2
                        temperature_array[4]  # TM_3
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
        if data_np.shape[0] == 0:
            self.update_status("No data to save after numpy conversion.")
            return

        # Calculate average power for the filename
        try:
            # Voltage is at index 1, Current is at index 2
            average_power = np.mean(data_np[:, 1] * data_np[:, 2])
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
            header = "Time,voltage,current,TH,Ttest,TM_1,TM_2,TM_3,FlipEvent"
            
            # Create a new column for flip events, initialized to 0
            flip_column = np.zeros((data_np.shape[0], 1))
            
            # Mark flip events in the new column
            # Convert recorded flip times (which are elapsed_time) to approximate row indices
            # This assumes self.sample_interval is the time step for each row in data_np
            if self.flip_event_times:
                for t_flip in self.flip_event_times:
                    # Find the closest row index for the flip time
                    # This can be tricky if sample_interval isn't perfectly regular or if t_flip doesn't align
                    # A more robust way is to find the index where data_np[:, 0] is closest to t_flip
                    try:
                        # Find the index of the time value closest to t_flip
                        closest_time_index = np.abs(data_np[:, 0] - t_flip).argmin()
                        flip_column[closest_time_index, 0] = 1 # Mark as 1 for flip
                    except IndexError:
                        print(f"Warning: Could not accurately map flip time {t_flip}s to data row.")
            
            # Concatenate the flip column to the main data
            data_to_save = np.concatenate((data_np, flip_column), axis=1)
            
            # Define custom format for columns, ensuring FlipEvent is integer
            # Original 8 columns are float, last one is int
            formats = ['%.5f'] * 8 + ['%d'] 

            np.savetxt(filename, data_to_save, delimiter=",",
                       header=header,
                       comments="", fmt=formats)
            self.update_status(f"Data (including flips) saved to {filename}")

            # Plotting
            plt.figure(figsize=(12, 9)) # Adjusted figure size

            plt.subplot(2, 1, 1)
            plt.plot(data_np[:, 0], data_np[:, 1], label="Voltage (V)")
            plt.plot(data_np[:, 0], data_np[:, 2] * 10, label="Current (A x10)") # Assuming current is multiplied by 10 for plotting
            plt.plot(data_np[:, 0], np.multiply(data_np[:, 1], data_np[:, 2]), label="Power (W)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.title("Voltage, Current, and Power")
            plt.legend()
            plt.grid(True)
            # Add vertical lines for flip events
            for t_flip in self.flip_event_times:
                plt.axvline(x=t_flip, color='r', linestyle='--', linewidth=0.8, label='Flip' if t_flip == self.flip_event_times[0] else None)

            plt.subplot(2, 1, 2)
            plt.plot(data_np[:, 0], data_np[:, 3], label="TH (°C)")
            plt.plot(data_np[:, 0], data_np[:, 4], label="Ttest (°C)")
            plt.plot(data_np[:, 0], data_np[:, 5], label="TM_1 (°C)")
            plt.plot(data_np[:, 0], data_np[:, 6], label="TM_2 (°C)")
            plt.plot(data_np[:, 0], data_np[:, 7], label="TM_3 (°C)")
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

        self.rec_manual_pwm_radio.config(state=tk.DISABLED)
        self.rec_corrected_direct_power_radio.config(state=tk.DISABLED)
        self.rec_pid_power_radio.config(state=tk.DISABLED) 

        self.control_mode_var.set("MANUAL_PWM") # Reset to default
        self._handle_recording_control_mode_change()
        self.emergency_stop_button.config(state=tk.DISABLED)
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

        # 2. Stop Heating (explicitly set PWM to 0 after PIDs are handled)
        # This ensures PWM is set to 0 even if no PID was active, or as a final confirmation.
        if self.arduino and self.arduino.running:
            self.arduino.control_arduino(0) # Send PWM 0 command to Arduino
            self.current_pwm_setting_0_255 = 0 # Update internal PWM state
            self.pwm_percentage_var.set(0) # Update UI variable for PWM percentage
            if hasattr(self, 'pwm_label_display') and self.pwm_label_display.winfo_exists(): # Update UI display
                 self.pwm_label_display.config(text="PWM: 0%")

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

        # 4. Reset UI elements related to recording and general controls
        self.stop_button.config(state=tk.DISABLED)
        self.flip_button.config(state=tk.DISABLED)
        self.steak_type_entry.config(state=tk.NORMAL) # Allow editing for next run

        # After E-Stop, always allow mode switching.
        # Specific mode operation buttons will still be governed by device connection status
        # handled within _handle_control_mode_change.
        for radio_btn in [self.manual_pwm_radio, self.pid_preheat_radio, self.pid_power_radio,
                          self.corrected_direct_power_radio, self.rec_manual_pwm_radio,
                          self.rec_pid_power_radio, self.rec_corrected_direct_power_radio]:
            if hasattr(self, radio_btn.winfo_name()):
                radio_btn.config(state=tk.NORMAL)

        # Other buttons' states will be set by _handle_control_mode_change based on actual connection.
        is_connected = self.arduino and self.arduino.running
        controls_state = tk.NORMAL if is_connected else tk.DISABLED
        self.start_button.config(state=controls_state)
        self.connect_button.config(state=tk.DISABLED if is_connected else tk.NORMAL)
        
        self._handle_control_mode_change() # Refresh states of live control panels

        # 5. Restart pre-recording display if devices are still connected
        if is_connected:
            self.stop_pre_recording_display_event.clear()
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
    root = tk.Tk()
    app = DataCollectionUI(root)
    root.mainloop()
