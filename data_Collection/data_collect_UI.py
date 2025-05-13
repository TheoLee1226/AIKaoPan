import tkinter as tk
from tkinter import messagebox, ttk # Added ttk for better Entry validation potentially
import time
import threading
import os
import numpy as np
import matplotlib.pyplot as plt

import pyvisa # For specific VISA error handling
import control_meter as cm
import control_arduino as ca
import serial

# --- PID Controller Class ---
class SimplePID:
    def __init__(self, Kp, Ki, Kd, setpoint, sample_time, output_limits=(0, 255), integral_limits=(-1000, 1000)):
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

        if current_value is None: # Cannot compute if current_value is invalid
            return self.output # Return last known good output or 0

        if time_change >= self.sample_time:
            error = float(self.setpoint) - float(current_value)

            # Proportional term
            p_term = self.Kp * error

            # Integral term (with anti-windup)
            self._integral += self.Ki * error * time_change # Use actual time_change
            self._integral = max(self.integral_min, min(self._integral, self.integral_max))
            i_term = self._integral

            # Derivative term (on measurement to reduce derivative kick)
            input_change = float(current_value) - self._last_input
            d_term = 0.0
            if time_change > 0:
                d_term = -self.Kd * (input_change / time_change)
            
            self.output = p_term + i_term + d_term
            self.output = max(self.output_min, min(self.output, self.output_max))

            self._last_error = error
            self._last_time = now
            self._last_input = float(current_value)
            return self.output
        return self.output # Not time to update yet, return last output

    def set_tunings(self, Kp, Ki, Kd):
        if Kp is not None: self.Kp = float(Kp)
        if Ki is not None: self.Ki = float(Ki)
        if Kd is not None: self.Kd = float(Kd)

    def set_setpoint(self, setpoint):
        self.setpoint = float(setpoint)
        # Optional: Reset integral when setpoint changes significantly
        # self._integral = 0.0 
        # self._last_error = 0.0 # Reset error as well

    def reset(self):
        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0
        self._last_input = 0.0
        self.output = 0.0
        # print("PID Reset")

# --- End PID Controller Class ---
class DataCollectionUI:
    def __init__(self, master):
        self.master = master
        master.title("Data Collection UI")
        master.geometry("380x820") # Adjusted window size for more frames

        self.voltage_meter = None
        self.current_meter = None
        self.arduino = None

        self.data = []
        self.recording = False
        self.flip_event_times = [] # To store timestamps of flip events
        # self.initial_power_for_filename = 0.0 # No longer needed, will calculate average power
        self.elapsed_time = 0
        self.timer_thread = None
        self.data_collection_thread = None
        self.pre_recording_display_thread = None
        self.stop_pre_recording_display_event = threading.Event()
        self.pre_recording_display_interval = 0.5 # seconds for display update when not recording or preheating

        # PID Preheat related
        self.pid_controller = None
        self.preheating_active = False
        self.preheat_thread = None
        self.stop_preheat_event = threading.Event()
        self.pid_control_interval = 0.2 # seconds for PID loop
        self.pid_setpoint_tm1_var = tk.DoubleVar(value=60.0) # Target TM_1 temp
        self.pid_kp_var = tk.DoubleVar(value=10.0) # Placeholder Kp
        self.pid_ki_var = tk.DoubleVar(value=0.1)  # Placeholder Ki
        self.pid_kd_var = tk.DoubleVar(value=0.5)  # Placeholder Kd

        # Configuration (can be made into UI inputs later)
        self.pwm_percentage_var = tk.IntVar(value=0) # PWM percentage (0-100), default to 0
        self.current_pwm_setting_0_255 = self._calculate_pwm_actual(self.pwm_percentage_var.get()) # Actual PWM value (0-255) for Arduino
        self.steak_type_var = tk.StringVar(value="test_steak") # Variable for steak type
        self.sample_interval = 0.01 # seconds

        # UI Elements
        # Connection Frame
        self.connection_frame = tk.LabelFrame(master, text="Device Connection")
        self.connection_frame.pack(pady=5, padx=10, fill="x")
        self.connect_button = tk.Button(self.connection_frame, text="Connect Devices", command=self.connect_devices)
        self.connect_button.pack(pady=5, padx=5)

        # Recording Control Frame
        self.recording_control_frame = tk.LabelFrame(master, text="Recording Control")
        self.recording_control_frame.pack(pady=5, padx=10, fill="x")
        self.start_button = tk.Button(self.recording_control_frame, text="Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, pady=5, padx=5, expand=True)
        self.stop_button = tk.Button(self.recording_control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, pady=5, padx=5, expand=True)


        # Steak Type Input
        self.steak_type_frame = tk.LabelFrame(master, text="Steak Type")
        self.steak_type_frame.pack(pady=5, padx=10, fill="x")

        self.steak_type_label = tk.Label(self.steak_type_frame, text="Type:")
        self.steak_type_label.pack(side=tk.LEFT, padx=5)
        self.steak_type_entry = tk.Entry(self.steak_type_frame, textvariable=self.steak_type_var)
        self.steak_type_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # PWM Control
        self.pwm_frame = tk.LabelFrame(master, text="PWM Control")
        self.pwm_frame.pack(pady=5, padx=10, fill="x")

        self.pwm_label_display = tk.Label(self.pwm_frame, text=f"PWM: {self.pwm_percentage_var.get()}%")
        self.pwm_label_display.pack(side=tk.LEFT, padx=5)

        self.pwm_entry = tk.Entry(self.pwm_frame, textvariable=self.pwm_percentage_var, width=5)
        self.pwm_entry.pack(side=tk.LEFT, padx=5)
        
        self.pwm_range_label = tk.Label(self.pwm_frame, text="(0-100)")
        self.pwm_range_label.pack(side=tk.LEFT, padx=(0, 5)) # Add a little padding to the right
        # self.pwm_percentage_var.trace_add("write", self.pwm_entry_changed) # Optional: for live validation or label update

        self.set_pwm_button = tk.Button(self.pwm_frame, text="Set PWM", command=self.set_pwm_from_entry)
        self.set_pwm_button.pack(side=tk.LEFT, padx=5)

        self.set_pwm_zero_button = tk.Button(self.pwm_frame, text="Set to 0", command=self.set_pwm_to_zero)
        self.set_pwm_zero_button.pack(side=tk.LEFT, padx=5)

        # PID Preheat Control Frame
        self.preheat_frame = tk.LabelFrame(master, text="PID Preheat Control (TM_1)")
        self.preheat_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(self.preheat_frame, text="Target Temp (°C):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.pid_setpoint_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_setpoint_tm1_var, width=7)
        self.pid_setpoint_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(self.preheat_frame, text="Kp:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W) # Already English
        self.pid_kp_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_kp_var, width=7)
        self.pid_kp_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(self.preheat_frame, text="Ki:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W) # Already English
        self.pid_ki_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_ki_var, width=7)
        self.pid_ki_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(self.preheat_frame, text="Kd:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W) # Already English
        self.pid_kd_entry = ttk.Entry(self.preheat_frame, textvariable=self.pid_kd_var, width=7)
        self.pid_kd_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)

        self.preheat_button = tk.Button(self.preheat_frame, text="Start Preheat", command=self.toggle_preheat, state=tk.DISABLED)
        self.preheat_button.grid(row=0, column=2, rowspan=1, padx=10, pady=5, sticky=tk.EW)
        
        self.preheat_status_label = tk.Label(self.preheat_frame, text="Preheat Status: Off", wraplength=200, justify=tk.LEFT)
        self.preheat_status_label.grid(row=1, column=2, rowspan=3, padx=10, pady=2, sticky=tk.W)

        # Configure column weights for preheat_frame grid
        self.preheat_frame.grid_columnconfigure(0, weight=0)
        self.preheat_frame.grid_columnconfigure(1, weight=0)
        self.preheat_frame.grid_columnconfigure(2, weight=1)
        # Real-time data display
        self.realtime_data_frame = tk.LabelFrame(master, text="Real-time Data")
        self.realtime_data_frame.pack(pady=10, padx=10, fill="x")

        self.voltage_display_label = tk.Label(self.realtime_data_frame, text="Voltage: --- V")
        self.voltage_display_label.pack(anchor=tk.W)
        self.current_display_label = tk.Label(self.realtime_data_frame, text="Current: --- A")
        self.current_display_label.pack(anchor=tk.W)
        self.power_display_label = tk.Label(self.realtime_data_frame, text="Power: --- W") # Power display
        self.power_display_label.pack(anchor=tk.W)
        self.th_display_label = tk.Label(self.realtime_data_frame, text="TH: --- °C")
        self.th_display_label.pack(anchor=tk.W)
        self.ttest_display_label = tk.Label(self.realtime_data_frame, text="Ttest: --- °C")
        self.ttest_display_label.pack(anchor=tk.W)
        self.tm1_display_label = tk.Label(self.realtime_data_frame, text="TM_1: --- °C")
        self.tm1_display_label.pack(anchor=tk.W)
        self.tm2_display_label = tk.Label(self.realtime_data_frame, text="TM_2: --- °C")
        self.tm2_display_label.pack(anchor=tk.W)
        self.tm3_display_label = tk.Label(self.realtime_data_frame, text="TM_3: --- °C")
        self.tm3_display_label.pack(anchor=tk.W)
        
        # Action Frame for Flip Button
        self.action_frame = tk.LabelFrame(master, text="Actions")
        self.action_frame.pack(pady=5, padx=10, fill="x")
        self.flip_button = tk.Button(self.action_frame, text="Log Flip Event", command=self.log_flip_event, state=tk.DISABLED)
        self.flip_button.pack(pady=5, padx=5)

        # Timer Display Frame
        self.timer_frame = tk.LabelFrame(master, text="Elapsed Time")
        self.timer_frame.pack(pady=5, padx=10, fill="x")
        self.time_label = tk.Label(self.timer_frame, text="00:00:00", font=("Helvetica", 64))
        self.time_label.pack(pady=5, padx=5)

        self.status_label = tk.Label(master, text="Status: Disconnected", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_realtime_display() # Initialize display
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_status("Disconnected. Press 'Connect Devices'.")

    def _calculate_pwm_actual(self, percentage):
        """Converts percentage (0-100) to actual PWM value (0-255)."""
        return int(round(percentage / 100.0 * 255))

    def log_flip_event(self):
        if self.recording:
            current_flip_time = self.elapsed_time
            self.flip_event_times.append(current_flip_time)
            self.update_status(f"Flip event logged at {current_flip_time}s.")

    # def update_pwm_from_scale_percentage(self, percentage_str): # No longer needed with Entry
    #     percentage = int(percentage_str)
    #
    #     if self.preheating_active:
    #         # If user moves slider during preheat, stop preheat and switch to manual PWM
    #         print("[UI] Manual PWM override: Stopping PID preheat.")
    #         self.stop_preheat() # This will set self.preheating_active to False
    #
    #     self.current_pwm_setting_0_255 = self._calculate_pwm_actual(percentage)
    #     self.pwm_label_display.config(text=f"PWM: {percentage}")

    def set_pwm_from_entry(self):
        try:
            percentage = self.pwm_percentage_var.get()
            if not (0 <= percentage <= 100):
                messagebox.showerror("PWM Error", "PWM percentage must be between 0 and 100.")
                # Optionally reset to a valid value or last known good
                # self.pwm_percentage_var.set(min(100, max(0, percentage))) # Clamp
                return
        except tk.TclError:
            messagebox.showerror("PWM Error", "Invalid PWM percentage. Please enter a number.")
            return

        if self.arduino and self.arduino.running:
            if self.preheating_active:
                print("[UI] Manual PWM set: Stopping PID preheat.")
                self.stop_preheat()
            
            self.current_pwm_setting_0_255 = self._calculate_pwm_actual(percentage)
            self.pwm_label_display.config(text=f"PWM: {percentage}%") # Update label
            self.arduino.control_arduino(self.current_pwm_setting_0_255)
            self.update_status(f"PWM set to {self.pwm_percentage_var.get()}% ({self.current_pwm_setting_0_255})")
        else:
            messagebox.showwarning("PWM Error", "Arduino not connected. Cannot send PWM command.")

    def set_pwm_to_zero(self):
        self.pwm_percentage_var.set(0) # Update the Tkinter variable, which updates the Entry box

        if self.arduino and self.arduino.running:
            if self.preheating_active:
                print("[UI] PWM set to 0: Stopping PID preheat.")
                self.stop_preheat()
            
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
        self.update_realtime_display(None, None, [None]*5)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        print(message) # Also print to console

    def connect_devices(self):
        self.update_status("Connecting...")
        try:
            self.update_status("Connecting to Voltage Meter (ASRL11)...")
            self.voltage_meter = cm.control_meter()
            self.voltage_meter.connect("ASRL11::INSTR")
            self.voltage_meter.set_voltage_mode()
            self.update_status("Voltage Meter connected.")
            time.sleep(1)

            self.update_status("Connecting to Current Meter (ASRL4)...")
            self.current_meter = cm.control_meter()
            self.current_meter.connect("ASRL4::INSTR")
            self.current_meter.set_current_mode()
            self.update_status("Current Meter connected.")
            time.sleep(1)

            self.update_status("Connecting to Arduino (COM3)...")
            self.arduino = ca.control_arduino("COM3", 9600)
            if not self.arduino.running:
                self.arduino = None # Ensure it's None if connection truly failed
                raise Exception("Failed to connect to Arduino or start its read thread.")
            self.update_status("Arduino connected. Initializing live display...")

            # Start pre-recording display now that Arduino is up
            self.stop_pre_recording_display_event.clear()
            if not (self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive()):
                self.pre_recording_display_thread = threading.Thread(target=self.run_pre_recording_display_loop, daemon=True)
                self.pre_recording_display_thread.start()

            self.preheat_button.config(state=tk.NORMAL)
            time.sleep(1) # Short delay to allow first data to potentially arrive for display

            self.start_button.config(state=tk.NORMAL)
            self.connect_button.config(state=tk.DISABLED)
            self.update_status("Devices connected. Live display active. Ready to record.")

        except Exception as e:
            error_msg = f"Connection failed: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Connection Error", error_msg)
            self.close_devices() # Ensure any partially opened devices are closed
            self.start_button.config(state=tk.DISABLED)
            # self.reset_realtime_display() # close_devices will call this
            self.preheat_button.config(state=tk.DISABLED)
            self.connect_button.config(state=tk.NORMAL) # Allow retry

    def run_pre_recording_display_loop(self):
        self.update_status("Live display active (not recording).")
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
        if not self.arduino or not self.arduino.running:
            messagebox.showerror("Error", "Arduino not connected.")
            return

        if self.recording:
            messagebox.showerror("Error", "Recording in progress. Please stop recording first.")
            return

        if self.preheating_active:
            self.stop_preheat()
        else:
            self.start_preheat()
    def start_preheat(self):
        try:
            setpoint = self.pid_setpoint_tm1_var.get()
            kp = self.pid_kp_var.get()
            ki = self.pid_ki_var.get()
            kd = self.pid_kd_var.get()
        except tk.TclError:
            messagebox.showerror("Input Error", "Invalid PID parameters or target temperature.")
            return

        self.pid_controller = SimplePID(Kp=kp, Ki=ki, Kd=kd, setpoint=setpoint, sample_time=self.pid_control_interval)
        self.preheating_active = True
        self.stop_preheat_event.clear()

        self.preheat_thread = threading.Thread(target=self.run_preheat_loop, daemon=True)
        self.preheat_thread.start()

        self.preheat_button.config(text="Stop Preheat")
        # self.start_button.config(state=tk.DISABLED) # Allow recording to be started, which will stop preheat
        # self.pwm_scale.config(state=tk.DISABLED) # PWM scale remains active
        self.pid_setpoint_entry.config(state=tk.DISABLED)
        self.pid_kp_entry.config(state=tk.DISABLED)
        self.pid_ki_entry.config(state=tk.DISABLED)
        self.pid_kd_entry.config(state=tk.DISABLED)
        self.pwm_entry.config(state=tk.DISABLED) # Disable manual PWM entry
        self.set_pwm_button.config(state=tk.DISABLED) # Disable manual Set PWM button
        self.update_status(f"PID Preheat started. Target TM_1: {setpoint}°C")

    def stop_preheat(self):
        if self.preheat_thread and self.preheat_thread.is_alive():
            self.stop_preheat_event.set()
            self.preheat_thread.join(timeout=2.0)
            if self.preheat_thread.is_alive():
                print("[UI] WARNING: Preheat thread did not terminate in time!")
        self.preheat_thread = None
        self.preheating_active = False

        if self.arduino and self.arduino.running:
            self.arduino.control_arduino(0) # Turn off heater

        self.preheat_button.config(text="Start Preheat")
        if self.arduino and self.arduino.running: # Only enable if connected
            self.start_button.config(state=tk.NORMAL)
        # self.pwm_scale.config(state=tk.NORMAL) # PWM scale remains active
        self.pid_setpoint_entry.config(state=tk.NORMAL)
        self.pid_kp_entry.config(state=tk.NORMAL)
        self.pid_ki_entry.config(state=tk.NORMAL)
        self.pid_kd_entry.config(state=tk.NORMAL)
        self.pwm_entry.config(state=tk.NORMAL) # Re-enable manual PWM entry
        self.set_pwm_button.config(state=tk.NORMAL) # Re-enable manual Set PWM button
        self.preheat_status_label.config(text="Preheat Status: Off")
        self.update_status("PID Preheat stopped.")

    def run_preheat_loop(self):
        if not self.pid_controller:
            print("[PreheatLoop] PID controller not initialized.")
            return
        
        self.pid_controller.reset() # Reset PID state before starting
        self.pid_controller.set_setpoint(self.pid_setpoint_tm1_var.get()) # Ensure setpoint is current
        self.pid_controller.set_tunings(self.pid_kp_var.get(), self.pid_ki_var.get(), self.pid_kd_var.get())

        while not self.stop_preheat_event.is_set() and self.arduino and self.arduino.running:
            temps = self.arduino.return_temperature()
            tm1_current = None
            if temps.size == 5 and temps[2] is not None and temps[2] > -1: # TM_1 is at index 2
                tm1_current = temps[2]
            
            # Update PID controller and get PWM output
            # The PID update method itself handles the sample_time logic
            pwm_output_float = self.pid_controller.update(tm1_current) 

            if pwm_output_float is not None: # pwm_output_float could be None if update interval not met
                pwm_to_send = int(round(pwm_output_float))
                self.arduino.control_arduino(pwm_to_send)
                # Update UI (use master.after for thread safety)
                target_temp_str = f"Target: {self.pid_setpoint_tm1_var.get():.1f}°C"
                tm1_str = f"TM_1: {tm1_current if tm1_current is not None else 'N/A'}°C"
                pwm_str = f"PID PWM: {int(round(pwm_output_float/255*100))}% ({pwm_to_send})"
                status_text = f"{target_temp_str}\n{tm1_str}\n{pwm_str}"
                self.master.after(0, self.preheat_status_label.config, {"text": status_text})

            time.sleep(self.pid_control_interval / 2) # Shorter sleep, PID internal timing handles the actual update rate
        print("[PreheatLoop] Preheat loop finished.")

    def start_recording(self):
        if not (self.voltage_meter and self.current_meter and self.arduino and self.arduino.running):
            messagebox.showerror("Error", "Devices are not connected properly.")
            return

        self.recording = True
        self.elapsed_time = 0

        if self.preheating_active:
            # messagebox.showinfo("Info", "Preheating is active. It will be stopped before starting the recording.") # 移除彈窗
            self.stop_preheat() # Stop preheating before recording

        # Send current PWM setting from UI to Arduino when recording starts
        if self.arduino and self.arduino.running:
            print(f"[UI] Sending initial PWM {self.pwm_percentage_var.get()}% ({self.current_pwm_setting_0_255}) at start of recording.")
            self.arduino.control_arduino(self.current_pwm_setting_0_255)

        # Initial power for filename is no longer calculated here. Average power will be calculated at save.
        self.data = [] # Clear previous data
        self.flip_event_times = [] # Clear previous flip events
        
        # Stop pre-recording display thread
        if self.pre_recording_display_thread and self.pre_recording_display_thread.is_alive():
            # print("[UI] Stopping pre-recording display thread for recording...") # Debug
            self.stop_pre_recording_display_event.set()
            self.pre_recording_display_thread.join(timeout=2.0) 
            if self.pre_recording_display_thread.is_alive():
                print("[UI] WARNING: Pre-recording display thread did not terminate in time before recording!")
            self.pre_recording_display_thread = None 
            # print("[UI] Pre-recording display thread stopped.") # Debug

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
        # self.pwm_scale.config(state=tk.DISABLED) # PWM scale remains active
        self.steak_type_entry.config(state=tk.DISABLED) # Disable steak type entry during recording
        self.preheat_button.config(state=tk.DISABLED) # Disable preheat button during recording
        self.flip_button.config(state=tk.NORMAL) # Enable flip button during recording
        self.update_status("Recording started...")

    def stop_recording(self):
        # print("[UI] stop_recording called") # Debug
        self.recording = False # Signal threads to stop
        # print(f"[UI] self.recording set to {self.recording}") # Debug

        if self.data_collection_thread and self.data_collection_thread.is_alive():
            self.update_status("Stopping recording, please wait for data processing...")
            # print("[UI] Attempting to join data_collection_thread...") # Debug
            self.data_collection_thread.join(timeout=10.0) # Wait for data collection to finish, with a 10-second timeout
            if self.data_collection_thread.is_alive():
                print("[UI] WARNING: Data collection thread did not terminate in time after stopping!")
                # At this point, the thread is still stuck.
            # else: # Debug
                # print("[UI] Data collection thread joined successfully.") # Debug
        
        # Send PWM 0 to Arduino when recording stops
        if self.arduino and self.arduino.running:
            print("[UI] Sending PWM 0 at stop of recording.")
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
        # self.pwm_scale.config(state=tk.NORMAL) # PWM scale remains active
        self.steak_type_entry.config(state=tk.NORMAL) # Re-enable steak type entry
        self.preheat_button.config(state=tk.NORMAL if self.arduino and self.arduino.running else tk.DISABLED)
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
        else:
             self.connect_button.config(state=tk.NORMAL) # Not connected, allow reconnect

    def run_ui_timer(self):
        while self.recording:
            time.sleep(1)
            if not self.recording: # Check again in case stop_recording was called
                break
            self.elapsed_time += 1
            # Schedule UI update on main thread
            self.master.after(0, self.update_time_label)

    def update_time_label(self):
        # 檢查主視窗和標籤是否存在，防止在銷毀後訪問
        # Check if the main window and label exist to prevent access after destruction
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
            # 小概率事件：控件正在被銷毀過程中
            print("TclError in update_time_label, likely widget destroyed.")

    def update_realtime_display(self, voltage, current, temps):
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
        loop_start_time = time.time()
        self.update_status("Data collection thread started.") # Removed PWM display from here
        # print("[DCL] Data Collection Loop started") # Debug

        while self.recording:
            current_loop_time = time.time()
            
            try:
                self.arduino.control_arduino(self.current_pwm_setting_0_255) # Use the actual 0-255 PWM value
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
        
        # Save data to a 'data' subdirectory relative to the script
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

            plt.tight_layout()
            plt.show() # This will block until the plot window is closed.
        except Exception as e:
            error_msg = f"Error during saving or plotting: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Save/Plot Error", error_msg)


    def close_devices(self):
        self.update_status("Closing device connections...")

        # Stop preheat thread if active
        if self.preheating_active:
            self.stop_preheat() # This will also set PWM to 0

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

    def on_closing(self):
        if self.recording:
            if messagebox.askyesno("Confirm Exit", "Recording is in progress. Stop recording and exit?"):
                self.stop_recording() # This will also join the data_collection_thread
            else:
                return # Don't close

        if self.preheating_active: # Ensure preheating is stopped before closing
            # self.stop_preheat() # Already called in close_devices if active
            pass

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
