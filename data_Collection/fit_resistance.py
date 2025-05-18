import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the resistance-temperature model function
def resistance_model(T, A, B, C):
    """
    Resistance-temperature model: R(T) = A * exp(B * T) + C
    T: Temperature in Celsius
    A, B, C: Coefficients to be fitted
    """
    return A * np.exp(B * T) + C

# File path (assuming the file is in the same directory or provide full path)
file_path = 'G:\我的雲端硬碟\Research\StartUp\AIKaoPan\AIKaoPan\data_collection\Temp_resistance.csv' # 使用者上傳的檔案名稱

try:
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Rename columns for easier access if needed, and select relevant ones
    # Header: "Time,voltage,current,TH,Ttest,TM_1,TM_2,TM_3,FlipEvent"
    data.columns = data.columns.str.strip() # Remove any leading/trailing whitespace from column names

    # Calculate Resistance R = V/I
    # Add a small epsilon to current to prevent division by zero if current is exactly 0,
    # though we will filter out zero current later.
    epsilon = 1e-9
    data['Resistance'] = data['voltage'] / (data['current'] + epsilon)

    # Select TM_1 as the temperature source
    data['Temperature'] = data['TH']

    # --- Data Filtering ---
    # 1. Filter out rows where current is very close to zero (e.g., <= 0.01 A)
    #    as this can lead to unrealistic resistance values or errors.
    #    Also, resistance should be positive.
    filtered_data = data[data['current'] > 0.01]

    # 2. Filter out rows with invalid temperature readings (e.g., TH <= 0 or placeholder like -1)
    #    Assuming temperatures should be above 0 Celsius for this heating application.
    #    The original UI code checks for > -1, let's use a slightly more restrictive T > 0 for physical sense.
    #    Or, stick to TM_1 > -1 if that's the known invalid marker.
    #    Given the data typically starts from room temperature, T > 10 might be even better to avoid noise at low power.
    filtered_data = filtered_data[filtered_data['Temperature'] > 10] # TH in Celsius

    # 3. Ensure resistance is positive
    filtered_data = filtered_data[filtered_data['Resistance'] > 0]
    
    # Remove rows with NaN values in Temperature or Resistance that might have resulted from previous steps
    filtered_data = filtered_data.dropna(subset=['Temperature', 'Resistance'])

    if filtered_data.empty:
        print("數據篩選後沒有剩餘數據點，請檢查 CSV 檔案內容或篩選條件。")
    else:
        print(f"數據篩選後剩餘 {len(filtered_data)} 個數據點。")
        # Prepare data for curve fitting
        T_data = filtered_data['Temperature'].values
        R_data = filtered_data['Resistance'].values

        # --- Curve Fitting ---
        # Initial guesses for parameters (A, B, C)
        # From previous thought process: A=14.8, B=0.0026, C=0 or small
        # Let's try with C starting at 0 or a small positive value.
        initial_guesses = [15.0, 0.003, 1.0] # A, B, C

        # Define bounds for the parameters to guide the optimizer
        # A > 0, B can be small positive, C >= 0 (usually resistance doesn't drop below a certain base)
        # B is typically small for metals (e.g., 0.001 to 0.005)
        lower_bounds = [0.1, 0.0001, 0]
        upper_bounds = [100, 0.05, 50] # Adjusted upper bound for A and C

        try:
            params, covariance = curve_fit(resistance_model, T_data, R_data,
                                           p0=initial_guesses, bounds=(lower_bounds, upper_bounds),
                                           maxfev=5000) # Increased maxfev

            A_opt, B_opt, C_opt = params
            print("\n擬合得到的最佳係數：")
            print(f"A = {A_opt:.4f}")
            print(f"B = {B_opt:.6f}") # B is often very small, so more decimal places
            print(f"C = {C_opt:.4f}")

            # Calculate R_squared for goodness of fit
            residuals = R_data - resistance_model(T_data, A_opt, B_opt, C_opt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((R_data - np.mean(R_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R-squared (擬合優度) = {r_squared:.4f}")


            # --- Configure Matplotlib for Chinese characters ---
            # You might need to change 'Microsoft JhengHei' to a font available on your system
            # that supports Chinese characters (e.g., 'SimHei', 'Noto Sans CJK TC')
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Or 'SimHei' for Simplified Chinese
            plt.rcParams['axes.unicode_minus'] = False  # Resolve the minus sign display issue

            # --- Plotting the results ---
            plt.figure(figsize=(10, 6))
            plt.scatter(T_data, R_data, label='實驗數據 (TH)', s=10, color='blue', alpha=0.5)
            
            # Generate T values for the fitted curve for a smoother plot
            T_smooth = np.linspace(T_data.min(), T_data.max(), 500)
            R_fitted = resistance_model(T_smooth, A_opt, B_opt, C_opt)
            
            plt.plot(T_smooth, R_fitted, label=f'擬合曲線\nR(T) = {A_opt:.2f}*exp({B_opt:.4f}*T) + {C_opt:.2f}\n$R^2={r_squared:.4f}$', color='red')

            plt.xlabel('溫度 TH (°C)')
            plt.ylabel('電阻 (Ohms)')
            plt.title('電阻 vs. 溫度 (TH) 擬合結果')
            plt.legend()
            plt.grid(True)
            plt.show()

        except RuntimeError:
            print("\n無法找到最佳擬合參數。請檢查：")
            print("1. 初始猜測值 (initial_guesses) 是否合理。")
            print("2. 參數邊界 (bounds) 是否合適。")
            print("3. 數據本身是否適合這個模型。")
            print("4. 嘗試調整 maxfev (最大迭代次數)。")
        except Exception as e:
            print(f"\n擬合過程中發生錯誤：{e}")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path}'。請確認檔案路徑是否正確。")
except Exception as e:
    print(f"處理檔案時發生錯誤：{e}")