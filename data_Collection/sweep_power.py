import control_arduino as ca
import control_meter as cm
import numpy as np
import time
import pandas as pd # For CSV saving
import matplotlib.pyplot as plt # For plotting
import os # For path operations

# --- 設備連接參數 ---
VOLTAGE_METER_ADDRESS = "ASRL11::INSTR" # Example address, please adjust according to your setup
CURRENT_METER_ADDRESS = "ASRL4::INSTR"  # Example address, please adjust according to your setup
ARDUINO_COM_PORT = "COM3"               # Example COM port, please adjust according to your setup
ARDUINO_BAUDRATE = 9600

# --- PWM 控制參數 ---
INVERT_PWM_OUTPUT = True # True: PWM output inverted (0 becomes 255, 255 becomes 0), False: normal output

# --- 全局設備句柄 ---
# These will be initialized by connect_all_devices
arduino_dev = None
voltage_meter_dev = None
current_meter_dev = None

def connect_all_devices():
    """Connects all necessary devices."""
    global arduino_dev, voltage_meter_dev, current_meter_dev
    print("Connecting devices...")
    try:
        # 連接到電壓表
        print(f"Connecting to Voltage Meter ({VOLTAGE_METER_ADDRESS})...")
        voltage_meter_dev = cm.control_meter()
        voltage_meter_dev.connect(VOLTAGE_METER_ADDRESS)
        voltage_meter_dev.set_voltage_mode()
        print("電壓表已連接。")
        time.sleep(0.5)

        # 連接到電流表
        print(f"Connecting to Current Meter ({CURRENT_METER_ADDRESS})...")
        current_meter_dev = cm.control_meter()
        current_meter_dev.connect(CURRENT_METER_ADDRESS)
        current_meter_dev.set_current_mode()
        print("Current Meter connected.")
        time.sleep(0.5)
        
        # 連接到Arduino
        print(f"Connecting to Arduino ({ARDUINO_COM_PORT})...")
        arduino_dev = ca.control_arduino(ARDUINO_COM_PORT, ARDUINO_BAUDRATE)
        if not arduino_dev.running:
            arduino_dev.close() # Ensure it's closed, even if not fully running
            arduino_dev = None 
            raise Exception("Failed to connect to Arduino or start its read thread.")
        print("Arduino connected.")
        time.sleep(1) # Wait for Arduino to stabilize
        return True

    except Exception as e:
        print(f"Error connecting devices: {e}")
        # 清理部分連接的設備
        if arduino_dev:
            arduino_dev.close()
            arduino_dev = None
        if voltage_meter_dev:
            voltage_meter_dev.close()
            voltage_meter_dev = None
        if current_meter_dev:
            current_meter_dev.close()
            current_meter_dev = None
        return False

def close_all_devices():
    """Closes all device connections."""
    global arduino_dev, voltage_meter_dev, current_meter_dev
    print("Closing all devices...")
    if arduino_dev:
        # 安全起見，在關閉前將PWM設為0
        try:
            if arduino_dev.running:
                 logical_pwm_off = 0
                 pwm_signal_to_send_off = 255 - logical_pwm_off if INVERT_PWM_OUTPUT else logical_pwm_off
                 arduino_dev.control_arduino(pwm_signal_to_send_off)
                 print(f"Logical PWM set to 0 (Sent to Arduino: {pwm_signal_to_send_off}).")
        except Exception as e:
            print(f"Error setting PWM to 0 before closing: {e}")
        arduino_dev.close()
        print("Arduino closed.")
    if voltage_meter_dev:
        voltage_meter_dev.close()
        print("Voltage Meter closed.")
    if current_meter_dev:
        current_meter_dev.close()
        print("Current Meter closed.")

# --- 主腳本 ---
if __name__ == "__main__":
    collected_data_list = [] 
    start_time_script = None # Will be set when data collection starts

    if connect_all_devices():
        print("Starting PWM sweep to collect data...")
        pwm_signal_to_send = 255 if INVERT_PWM_OUTPUT else 0
        arduino_dev.control_arduino(pwm_signal_to_send)

        time.sleep(1)
        if INVERT_PWM_OUTPUT:
            print("Note: PWM output is set to inverted mode.")
        start_time_script = time.time() # Record start time for elapsed time calculation
        try:
            for i in range(256): # i 是邏輯PWM值 (0 到 255)
                if not arduino_dev or not arduino_dev.running:
                    print("Arduino disconnected during sweep. Aborting.")
                    break
                
                # 根據INVERT_PWM_OUTPUT決定實際發送給Arduino的PWM信號
                pwm_signal_to_send = 255 - i if INVERT_PWM_OUTPUT else i
                arduino_dev.control_arduino(pwm_signal_to_send)
                
                voltage_reading_list = [None] 
                current_reading_list = [None] 
                # Short delay to ensure PWM setting takes effect and circuit stabilizes
                time.sleep(0.5) # This delay can be adjusted as needed
                
                # Inner loop to take multiple readings for the same PWM setting
                for _ in range(20):
                    if voltage_meter_dev:
                        try:
                            voltage_reading_list = voltage_meter_dev.read_voltage()
                        except Exception as e:
                            print(f"Error reading voltage at logical PWM {i} (sent {pwm_signal_to_send}): {e}")
                    else:
                        print(f"Voltage meter not available at logical PWM {i}")

                    if current_meter_dev:
                        try:
                            current_reading_list = current_meter_dev.read_current()
                        except Exception as e:
                            print(f"Error reading current at logical PWM {i} (sent {pwm_signal_to_send}): {e}")
                    else:
                        print(f"Current meter not available at logical PWM {i}")

                    v_val = voltage_reading_list[0] if voltage_reading_list and voltage_reading_list[0] is not None else np.nan
                    c_val = current_reading_list[0] if current_reading_list and current_reading_list[0] is not None else np.nan

                    current_loop_time = time.time()
                    elapsed_time_seconds = current_loop_time - start_time_script
                    # Store elapsed time, logical PWM value i, voltage, and current
                    collected_data_list.append([elapsed_time_seconds, i, v_val, c_val])
                    print(f"Time: {elapsed_time_seconds:.2f}s, Logical PWM: {i} (Sent: {pwm_signal_to_send}), Voltage: {v_val if not np.isnan(v_val) else 'N/A'} V, Current: {c_val if not np.isnan(c_val) else 'N/A'} A")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("PWM sweep interrupted by user.")
        except Exception as e:
            print(f"Error during PWM sweep: {e}")
        finally:
            # 確保在掃描結束或發生錯誤時將PWM設置為0並關閉設備
            close_all_devices()
    else:
        print("Could not connect devices. Exiting.")

    # --- 處理並繪製數據 ---
    if collected_data_list:
        data_np = np.array(collected_data_list)
        # Extract columns: Time, PWM, Voltage, Current
        time_values = data_np[:, 0]
        pwm_values = data_np[:, 1] # pwm_values are still logical PWM
        voltage_values = data_np[:, 2]
        current_values = data_np[:, 3]
        
        # 計算功率，處理潛在的NaN值
        power_values = np.multiply(voltage_values, current_values)

        # --- 繪圖 ---
        print("Plotting data...")
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('PWM Sweep Device Characteristics (PWM values are logical)', fontsize=16) # Updated title to reflect PWM is logical

        # 電壓 vs PWM
        axs[0].plot(pwm_values, voltage_values, marker='.', linestyle='-', color='blue', label='Voltage')
        axs[0].set_ylabel('Voltage (V)')
        axs[0].set_title('Voltage vs. Logical PWM')
        axs[0].grid(True)
        axs[0].legend()

        # 電流 vs PWM
        axs[1].plot(pwm_values, current_values, marker='.', linestyle='-', color='red', label='Current')
        axs[1].set_ylabel('Current (A)')
        axs[1].set_title('Current vs. Logical PWM')
        axs[1].grid(True)
        axs[1].legend()

        # 功率 vs PWM
        axs[2].plot(pwm_values, power_values, marker='.', linestyle='-', color='green', label='Power')
        axs[2].set_ylabel('Power (W)')
        axs[2].set_xlabel('Logical PWM Value (0-255)')
        axs[2].set_title('Power vs. Logical PWM')
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局為suptitle留出空間

        # 保存圖表
        plot_filename_base = f"pwm_char_plot_{'inverted_' if INVERT_PWM_OUTPUT else ''}{time.strftime('%Y%m%d_%H%M%S')}.png"
        save_dir = os.path.join("data_collection", "data_fit_power")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        full_plot_path = os.path.join(save_dir, plot_filename_base)

        try:
            plt.savefig(full_plot_path)
            print(f"Plot saved to {full_plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        
        # 設定Matplotlib以支持中文顯示 (如果需要且系統支持)
        # 以下字體列表是常見的選擇，您可能需要根據您的系統進行調整
        try:
            # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS'] # Keep if Chinese characters are still needed elsewhere
            plt.rcParams['axes.unicode_minus'] = False 
        except Exception as e:
            print(f"Error setting font properties (this does not affect plot generation, but non-ASCII characters might not display correctly): {e}")
        
        plt.show()


        # --- 將數據保存到CSV ---
        print("Saving data to CSV...")
        df = pd.DataFrame({
            'Time (s)': time_values,
            'Logical_PWM': pwm_values,
            'Voltage (V)': voltage_values,
            'Current (A)': current_values,
            'Power (W)': power_values
        })
        
        csv_filename_base = f"pwm_char_data_{'inverted_' if INVERT_PWM_OUTPUT else ''}{time.strftime('%Y%m%d_%H%M%S')}.csv"
        full_csv_path = os.path.join(save_dir, csv_filename_base)

        try:
            df.to_csv(full_csv_path, index=False, float_format='%.5f')
            print(f"Data saved to {full_csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            
    else:
        print("No data collected to plot or save.")

    
