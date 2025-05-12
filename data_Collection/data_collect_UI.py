import tkinter as tk
from tkinter import messagebox
import time
import threading
import os
import numpy as np
import matplotlib.pyplot as plt

import pyvisa # For specific VISA error handling
import control_meter as cm
import control_arduino as ca
import serial

class DataCollectionUI:
    def __init__(self, master):
        self.master = master
        master.title("Data Collection UI")
        master.geometry("350x520") # Adjusted window size for PWM and power display

        self.voltage_meter = None
        self.current_meter = None
        self.arduino = None

        self.data = []
        self.recording = False
        self.elapsed_time = 0
        self.timer_thread = None
        self.data_collection_thread = None

        # Configuration (can be made into UI inputs later)
        self.pwm_var = tk.IntVar(value=254) # Variable for PWM Scale
        self.PWM_power = self.pwm_var.get() # Initial PWM power
        self.steak_type = "ribeye"
        self.sample_interval = 0.01 # seconds

        # UI Elements
        self.connect_button = tk.Button(master, text="Connect Devices", command=self.connect_devices)
        self.connect_button.pack(pady=5)

        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # PWM Control
        self.pwm_frame = tk.LabelFrame(master, text="PWM 控制")
        self.pwm_frame.pack(pady=5, padx=10, fill="x")

        self.pwm_label_display = tk.Label(self.pwm_frame, text=f"PWM: {self.pwm_var.get()}")
        self.pwm_label_display.pack(side=tk.LEFT, padx=5)

        self.pwm_scale = tk.Scale(self.pwm_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                  variable=self.pwm_var, command=self.update_pwm_from_scale)
        self.pwm_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Real-time data display
        self.realtime_data_frame = tk.LabelFrame(master, text="即時數據 (Real-time Data)")
        self.realtime_data_frame.pack(pady=10, padx=10, fill="x")

        self.voltage_display_label = tk.Label(self.realtime_data_frame, text="電壓: --- V")
        self.voltage_display_label.pack(anchor=tk.W)
        self.current_display_label = tk.Label(self.realtime_data_frame, text="電流: --- A")
        self.current_display_label.pack(anchor=tk.W)
        self.power_display_label = tk.Label(self.realtime_data_frame, text="功率: --- W") # Power display
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

        self.time_label = tk.Label(master, text="00:00:00", font=("Helvetica", 16))
        self.time_label.pack(pady=10)

        self.status_label = tk.Label(master, text="Status: Disconnected", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_realtime_display() # Initialize display
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_status("Disconnected. Press 'Connect Devices'.")

    def update_pwm_from_scale(self, value):
        self.PWM_power = int(value)
        self.pwm_label_display.config(text=f"PWM: {self.PWM_power}")

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
                raise Exception("Failed to connect to Arduino or start its read thread.")
            self.update_status("Arduino connected.")
            time.sleep(5) # Allow Arduino to stabilize

            self.start_button.config(state=tk.NORMAL)
            self.connect_button.config(state=tk.DISABLED)
            self.update_status("All devices connected successfully. Ready to record.")

        except Exception as e:
            error_msg = f"Connection failed: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Connection Error", error_msg)
            self.close_devices() # Ensure any partially opened devices are closed
            self.start_button.config(state=tk.DISABLED)
            self.reset_realtime_display()
            self.connect_button.config(state=tk.NORMAL) # Allow retry

    def start_recording(self):
        if not (self.voltage_meter and self.current_meter and self.arduino and self.arduino.running):
            messagebox.showerror("Error", "Devices are not connected properly.")
            return

        self.recording = True
        self.elapsed_time = 0
        self.data = [] # Clear previous data
        
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
        self.pwm_scale.config(state=tk.DISABLED) # Disable PWM scale during recording
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
                print("[UI] WARNING: Data collection thread did not terminate after 10 seconds!")
                # At this point, the thread is still stuck.
            # else: # Debug
                # print("[UI] Data collection thread joined successfully.") # Debug
        
        # UI timer thread is daemon, will stop.
        # print("[UI] Proceeding after join attempt.") # Debug
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pwm_scale.config(state=tk.NORMAL) # Re-enable PWM scale
        self.connect_button.config(state=tk.NORMAL if not (self.voltage_meter and self.current_meter and self.arduino) else tk.DISABLED)


        if self.data:
            self.update_status("Recording stopped. Saving and plotting data...")
            self.save_and_plot_data()
            self.update_status(f"Data saved and plotted. Ready for new recording or disconnect.")
        else:
            self.update_status("Recording stopped. No data collected.")
        
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
            v_text = f"電壓: {voltage:.3f} V" if voltage is not None else "電壓: --- V"
            c_text = f"電流: {current:.3f} A" if current is not None else "電流: --- A"
            if voltage is not None and current is not None:
                power = voltage * current
                p_text = f"功率: {power:.2f} W"
            else:
                p_text = "功率: --- W"
            th_text = f"TH: {temps[0]:.2f} °C" if temps[0] is not None else "TH: --- °C"
            tt_text = f"Ttest: {temps[1]:.2f} °C" if temps[1] is not None else "Ttest: --- °C"
            tm1_text = f"TM_1: {temps[2]:.2f} °C" if temps[2] is not None else "TM_1: --- °C"
            tm2_text = f"TM_2: {temps[3]:.2f} °C" if temps[3] is not None else "TM_2: --- °C"
            tm3_text = f"TM_3: {temps[4]:.2f} °C" if temps[4] is not None else "TM_3: --- °C"

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
                self.arduino.control_arduino(self.PWM_power) # Use the potentially updated PWM_power
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

        try:
            initial_current = data_np[0, 2]
            initial_voltage = data_np[0, 1]
            power = initial_current * initial_voltage
        except IndexError:
            self.update_status("Not enough data to calculate initial power for filename.")
            power = 0 # Default power for filename

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save data to a 'data' subdirectory relative to the script
        directory = "data_collection/data" 
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename_base = f"{timestamp}_{self.steak_type}_{self.elapsed_time}S_{power:.2f}W.csv"
        filename = os.path.join(directory, filename_base)

        try:
            np.savetxt(filename, data_np, delimiter=",", 
                       header="Time,voltage,current,TH,Ttest,TM_1,TM_2,TM_3", 
                       comments="", fmt="%.5f")
            self.update_status(f"Data saved to {filename}")

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

        self.close_devices()
        try:
            plt.close('all') # 關閉所有 Matplotlib 圖窗
        except Exception as e:
            print(f"Error closing matplotlib figures: {e}")
            
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectionUI(root)
    root.mainloop()
