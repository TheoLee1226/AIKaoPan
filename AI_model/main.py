import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime
import Find_parameter as Fp
import Train_IC as TI
import MPC_PINN as MP

import control_arduino as CA

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或其他存在的中文字型名稱
plt.rcParams['axes.unicode_minus'] = False
# 禁用 Eager Execution (TensorFlow 1.x 兼容模式)
tf.disable_eager_execution()
# # 取得現在時間並格式化（例如 YYYYMMDD-HHMMSS）
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 設定隨機種子，確保結果可重現
np.random.seed(1234)
tf.set_random_seed(1234)

# # # 創建輸出目錄
output_dir = "PINN_PDE_output_20250201-231645"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# # # 新的輸出目錄：帶上時間戳
# output_dir = "PINN_PDE_output_" + timestamp

# # 若資料夾不存在則建立
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

    
#%% Classaicl_MPC
class ModelPredictiveControl(object):
    def __init__(self, A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal,
                 u_lb, u_ub):
        """
        A, B, C: 離散時間系統矩陣
        f: 預測步數 (H_p)
        v: 控制步數 (H_c)
        W3, W4: MPC 權重矩陣
        x0: 初始狀態
        desiredControlTrajectoryTotal: 期望的輸出軌跡（向量）
        u_lb, u_ub: 控制輸入上下界
        """
        self.A = A 
        self.B = B
        self.C = C
        self.f = f      # 預測時域長度
        self.v = v      # 控制時域長度
        self.W3 = W3 
        self.W4 = W4
        self.desiredControlTrajectoryTotal = desiredControlTrajectoryTotal

        self.u_min = u_lb  
        self.u_max = u_ub  

        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]

        self.currentTimeStep = 0

        self.states = [x0]    # 保存所有狀態
        self.inputs = []      # 保存施加過的控制輸入
        self.outputs = []     # 保存輸出 y = C*x

        # 形成提升矩陣
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()

    def formLiftedMatrices(self):
        f, v, r, n, m = self.f, self.v, self.r, self.n, self.m
        A, B, C = self.A, self.B, self.C

        # O: 輸出提升矩陣
        O = np.zeros((f * r, n))
        for i in range(f):
            powA = np.linalg.matrix_power(A, i+1)
            O[i*r:(i+1)*r, :] = C @ powA

        # M: 控制輸入對預測輸出之影響矩陣
        M = np.zeros((f * r, v * m))
        for i in range(f):
            for j in range(min(i+1, v)):
                powA = np.linalg.matrix_power(A, i - j)
                M[i*r:(i+1)*r, j*m:(j+1)*m] = C @ (powA @ B)

        # 利用權重構造最終增益矩陣
        tmp1 = M.T @ (self.W4 @ M)
        tmp2 = np.linalg.inv(tmp1 + self.W3)
        gainMatrix = tmp2 @ (M.T @ self.W4)
        return O, M, gainMatrix

    def propagateDynamics(self, controlInput, state):
        xkp1 = self.A @ state + self.B @ controlInput
        yk = self.C @ xkp1
        return xkp1, yk

    def computeControlInputs(self):
        # 取得未來 f 步的目標輸出
        desiredControlTrajectory = self.desiredControlTrajectoryTotal[
            self.currentTimeStep : self.currentTimeStep + self.f
        ]
        # 形成向量 S = 目標輸出 - 當前狀態預測的輸出
        vectorS = desiredControlTrajectory - (self.O @ self.states[self.currentTimeStep])
        # 利用增益矩陣計算整個控制序列
        inputSequenceComputed = self.gainMatrix @ vectorS  # shape (v*m,1)
        # 僅施加第一個控制輸入
        inputApplied = np.zeros((self.m, 1))
        u_first = inputSequenceComputed[0, 0]
        # 飽和處理
        if (self.u_min is not None) and (self.u_max is not None):
            u_first = np.clip(u_first, self.u_min, self.u_max)
        inputApplied[0, 0] = u_first

        # 狀態更新
        state_kp1, output_k = self.propagateDynamics(inputApplied, self.states[self.currentTimeStep])
        self.states.append(state_kp1)
        self.outputs.append(output_k)
        self.inputs.append(inputApplied)
        self.currentTimeStep += 1

#%% 主程式
if __name__ == "__main__":
    #%% PINN找參數
    hH= 1
    hM= 1
    CM= 1
    CH= 1
    hM_CM = hM/CM      # 參數 a
    hH_CH = hH/CH     # 參數 b
    Ta = 25.0    # 環境參數 TA
    t_train_min = 0
    t_train_max = 2
    t_number = 50 # 最大MPC預測範圍t_number
    # 產生訓練數據
    t_train = np.linspace(t_train_min, t_train_max, t_number).reshape(-1, 1)
    TM0_const = 25
    TH0_const = 25
    u_const  = 10
    TM0_train = np.full_like(t_train, TM0_const)
    TH0_train = np.full_like(t_train, TH0_const)
    u_train  = np.full_like(t_train, u_const)
    t_star = t_train       # 或自行定義新的時間點
    TM0_star = TM0_train
    TH0_star = TH0_train
    u_star  = u_train
    # # 利用已知解析解 (阻尼系統 + 常數控制) 生成 x, vx
    lambda_val = np.sqrt(hM_CM**2 - hM_CM*hH_CH + hH_CH**2)
    lambda1 = -(hM_CM + hH_CH) + lambda_val
    lambda2 = -(hM_CM + hH_CH) - lambda_val
    TM_eq = Ta + u_train/(3*hH_CH)
    TH_eq = Ta + 2*u_train/(3*hH_CH)
    M = TM0_train - TM_eq   # shape: (N_trajectories, 1)
    H = TH0_train - TH_eq   # shape: (N_trajectories, 1)
    C1 = ((2*hM_CM+lambda2)*M-hM_CM*H)/(lambda2-lambda1)
    C2 = M - C1
    TM_train = TM_eq + C1 * np.exp(lambda1 * t_train) + C2 * np.exp(lambda2 * t_train)  
    TH_train = TH_eq + (C1 * (2*hH_CH + lambda1)/hH_CH) * np.exp(lambda1 * t_train) + (C2 * (2*hH_CH + lambda2)/hH_CH) * np.exp(lambda2 * t_train)
    
    # %% Train IC
    # 在建立新模型之前，先清空預設圖形
    tf.compat.v1.reset_default_graph()
    #%% PINN初始化、輸入訓練
    # t_train_min = 0
    t_train_max = 5
    TM0_train_min = 20
    TM0_train_max = 50
    TH0_train_min = 20
    TH0_train_max = 50
    u_train_min = -50
    u_train_max = 50
    t_number = 50 # 最大MPC預測範圍t_number
    
    # 產生訓練數據
    t_train = np.linspace(t_train_min, t_train_max, t_number).reshape(-1, 1)
    t_train = np.tile(t_train, (t_number, 1))  # 變成 (50,50)
    
    TM0_train = np.random.uniform(TM0_train_min, TM0_train_max, (t_number, 1))
    TH0_train = np.random.uniform(TH0_train_min, TH0_train_max, (t_number, 1))
    u_train = np.random.uniform(u_train_min, u_train_max, (t_number, 1))
    # 讓每組 (x0, v0, u) 對應 50 個時間點，因此擴展 (50,1) -> (2500,1)
    TM0_train = np.repeat(TM0_train, t_number, axis=0)  # (2500, 1)
    TH0_train = np.repeat(TH0_train, t_number, axis=0)  # (2500, 1)
    u_train = np.repeat(u_train, t_number, axis=0)  # (2500, 1)
    
    # 利用已知解析解 (阻尼系統 + 常數控制) 生成 x, vx
    lambda_val = np.sqrt(hM_CM**2 - hM_CM*hH_CH + hH_CH**2)
    lambda1 = -(hM_CM + hH_CH) + lambda_val
    lambda2 = -(hM_CM + hH_CH) - lambda_val
    TM_eq = Ta + u_train/(3*hH_CH)
    TH_eq = Ta + 2*u_train/(3*hH_CH)
    M = TM0_train - TM_eq   # shape: (N_trajectories, 1)
    H = TH0_train - TH_eq   # shape: (N_trajectories, 1)
    C1 = ((2*hM_CM+lambda2)*M-hM_CM*H)/(lambda2-lambda1)
    C2 = M - C1
    TM_train = TM_eq + C1 * np.exp(lambda1 * t_train) + C2 * np.exp(lambda2 * t_train)  
    TH_train = TH_eq + (C1 * (2*hH_CH + lambda1)/hH_CH) * np.exp(lambda1 * t_train) + (C2 * (2*hH_CH + lambda2)/hH_CH) * np.exp(lambda2 * t_train)

    # 物理參數
    # d, m, k, coff_u = d_learned, m_learned, k_learned, coff_u_learned
    hH, CH, hM_CM, coff_u,Ta = 1, 1, 1, 1, 25
    layers = [4, 15, 15, 15, 2]

    model = TI. PhysicsInformedNN(
        t_train, TM0_train, TH0_train, u_train,
        TM_train, TH_train,
        layers, hH, CH, hM_CM, coff_u,Ta
    )

    # 如需重新訓練:
    # model.train(300000)
    # model.save_model(output_dir, "my_custom_model.ckpt")

    # 載入已訓練模型
    model.load_model(output_dir, "my_custom_model.ckpt")
    #%% 畫圖確認PINN模型
    # 設定測試數據範圍
    num_samples = 2  # 取 5 組隨機測試條件
    t_test = np.linspace(t_train_min, t_train_max, 100).reshape(-1, 1)
    
    # 隨機產生 num_samples 組 (x₀, v₀, u)
    random_cases = [
        (np.random.uniform(TM0_train_min, TM0_train_max),
         np.random.uniform(TH0_train_min, TH0_train_max),
         np.random.uniform(u_train_min, u_train_max))
        for _ in range(num_samples)
    ]
    
    # # 手動加入固定條件
    # fixed_cases = [
    #     (0.0, 0.0, 0.0),
    #     (0, 0, 5),
    # ]
    # # 手動加入固定條件
    fixed_cases = [
        
        (25, 25, 50),
    ]   
    # 合併所有測試條件
    test_cases = fixed_cases + random_cases
    # test_cases = fixed_cases     
    
    plt.figure(figsize=(12, 6))
    
    for i, (TM0_test, TH0_test, u_test) in enumerate(test_cases):
        TM0_test_array = np.full((100, 1), TM0_test)
        TH0_test_array = np.full((100, 1), TH0_test)
        u_test_array = np.full((100, 1), u_test)
    
        # **計算 PINN 預測解**
        TM_pred, TH_pred = model.predict(
            t_test, TM0_test_array, TH0_test_array, u_test_array)     
        
        TM_eq = Ta + u_test/(3*hH_CH)
        TH_eq = Ta + 2*u_test/(3*hH_CH)
        M = TM0_test - TM_eq   # shape: (N_trajectories, 1)
        H = TH0_test - TH_eq   # shape: (N_trajectories, 1)
        C1 = ((2*hM_CM+lambda2)*M-hM_CM*H)/(lambda2-lambda1)
        C2 = M - C1
        TM_true = TM_eq + C1 * np.exp(lambda1 * t_test) + C2 * np.exp(lambda2 * t_test)  
        TH_true = TH_eq + (C1 * (2*hH_CH + lambda1)/hH_CH) * np.exp(lambda1 * t_test) + (C2 * (2*hH_CH + lambda2)/hH_CH) * np.exp(lambda2 * t_test)
        
        plt.plot(t_test, TH_true, 'r-', linewidth=1,
                 label="True TH" if i == 0 else "")
        plt.plot(t_test, TH_pred, 'g--', linewidth=1,
                  label="PINN TH" if i == 0 else "")
        plt.plot(t_test,TM_true, 'y-', linewidth=1,
                 label="True TM" if i == 0 else "")
        plt.plot(t_test, TM_pred, 'b--', linewidth=1,
                  label="PINN TM" if i == 0 else "")
        
    plt.xlabel("$t$")
    plt.ylabel("$v_x$")
    plt.grid()
    plt.title("PINN vs True Solution for v_x(t)")
    plt.legend()
    plt.show()
    
    
#%% PINN_MPC
    timeSteps = 70
    time1 = np.arange(timeSteps)
    
    # 初始化軌跡
    # desiredTrajectory = Ta*np.ones((timeSteps, 1), dtype=np.float32)
    desiredTrajectory1 = Ta*np.ones((timeSteps, 1), dtype=np.float32)
    
    # 設定 SIN 函數範圍
    desiredTrajectory1[21:50] = 50 * np.sin(np.linspace(0, np.pi , 29))[:, np.newaxis] + Ta 

    iterations = 1000
    H_p = 5
    H_c = 3
    Q = 10
    R = 0.0005
    lr = 0.01
    dt = 0.5  # 取樣時間
    mpc_PINN = MP.MPC_Controller_TF1_HpHc(model=model, sess=model.sess, H_p=H_p, H_c=H_c,
                                    u_lb=u_train_min-50, u_ub=u_train_max+100, Q =Q , R =R, lr=lr ,dt =dt, Ta=Ta, hM_CM=hM_CM, hH_CH=hH_CH)
    
    total_steps = timeSteps - H_p   
    TM_history = []
    u_history = []
    
    # 初始狀態
    TMk = Ta
    THk = Ta
    coff_u_real =1
    # 記錄開始時間
    start_time = time.time()

    arduino = CA.control_arduino("COM7", 9600)  
    
    for step in range(total_steps):

        current_time = time.time() - start_time  # 計算目前已運行的時間
        print(f"Step {step+1}/{total_steps}, 已執行時間: {current_time:.2f} 秒")
    
        TM_ref_seg = desiredTrajectory1[step: step+H_p].reshape(H_p, 1)
    
        TM0_np = np.array([[TMk]], dtype=np.float32)
        TH0_np = np.array([[THk]], dtype=np.float32)
    
        # 1) 優化 => shape(H_c,1)
        u_seq_opt = mpc_PINN.optimize_control(TM0_np, TH0_np, TM_ref_seg, iterations,dt)
    
        # 2) 只施加第一個
        u_k = u_seq_opt[0, 0]
    
        # 3) **用 "真實 Plant" 模擬**
        # TMk, THk = MP.real_plant_dynamics(Ta, TMk, THk, u_k, dt, coff_u_real, hM_CM, hH_CH)

        # 3) 真實控制Arduino
        TMk = arduino.control_arduino_and_return_temp(u_k)
        THk = arduino.control_arduino_and_return_temp(u_k)

        # 4) 紀錄
        TM_history.append(TMk)
        u_history.append(u_k)

    TM_history.append(TMk)
    
    # 繪圖
    plt.figure()
    plt.plot(TM_history, 'r-o', label="TM (PINN-MPC)")
    plt.plot(desiredTrajectory1[:total_steps], 'b--', label="DesiredTrajectory")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.step(range(total_steps), u_history, where='post', label="u")
    plt.title("Control Input (PINN-MPC)")
    plt.show()