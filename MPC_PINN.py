import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或其他存在的中文字型名稱
plt.rcParams['axes.unicode_minus'] = False
# 禁用 Eager Execution (TensorFlow 1.x 兼容模式)
tf.disable_eager_execution()

# 設定隨機種子，確保結果可重現
np.random.seed(1234)
tf.set_random_seed(1234)

#%% PINN_MPC
class MPC_Controller_TF1_HpHc:
    def __init__(self, model, sess, H_p, H_c, u_lb, u_ub, Q, R, lr,dt, Ta, hM_CM, hH_CH):
        self.model = model
        self.sess = sess  # 共享模型的 Session
        self.H_p = H_p
        self.H_c = H_c
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.Q = Q
        self.R = R
        self.lr = lr
        
        self.Ta = Ta
        self.hM_CM = hM_CM
        self.hH_CH = hH_CH
        
        self.t_seq = tf.constant(np.arange(dt, (self.H_p+1)*dt, dt, dtype=np.float32).reshape(-1,1))
        
        with tf.variable_scope("MPC"):
            self.u_lb_tf = tf.constant(u_lb, dtype=tf.float32)
            self.u_ub_tf = tf.constant(u_ub, dtype=tf.float32)
            # 定義一個無限制的變數
            self.u_seq_unconstrained = tf.Variable(tf.zeros([H_c, 1], dtype=tf.float32))
            # 利用 sigmoid 重新參數化，強制 u_seq 落在 [u_lb, u_ub]
            self.u_seq = self.u_lb_tf + (self.u_ub_tf - self.u_lb_tf) * tf.sigmoid(self.u_seq_unconstrained)
            self.TM0_tf = tf.placeholder(tf.float32, shape=[1,1])
            self.TH0_tf = tf.placeholder(tf.float32, shape=[1,1])
            self.TM_ref_tf = tf.placeholder(tf.float32, shape=[H_p,1])
            self.state_cost, self.control_cost, self.loss = self.build_cost()
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[self.u_seq_unconstrained])
        
        # 初始化 MPC 範圍下的新變數（不重新初始化模型中已有的變數）
        mpc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MPC")
        self.sess.run(tf.variables_initializer(mpc_vars))

    def build_cost(self):
    # 建立 move-blocking => (H_p,1)
        tail_num = self.H_p - self.H_c
        last_u = self.u_seq[-1,:]  # shape(1,)
        tail_u = tf.tile(tf.reshape(last_u,[1,1]), [tail_num,1])
        u_extended = tf.concat([self.u_seq, tail_u], axis=0)  # shape(H_p,1)
    
        # tile x0,v0 => shape(H_p,1)
        TM0_tiled = tf.tile(self.TM0_tf, [self.H_p,1])
        TH0_tiled = tf.tile(self.TH0_tf, [self.H_p,1])
    
        # 用 net_output => x_pred, vx_pred
        TM_pred, TH_pred = self.model.net_output(self.t_seq, TM0_tiled, TH0_tiled, u_extended)
    
        # cost = Q*(x_pred-x_ref)^2 + R*(u_seq^2) (只對前H_c)
        state_cost = self.Q * tf.reduce_sum(tf.square(TM_pred - self.TM_ref_tf))
        
        # 加入 u = -alpha vx 這個條件
        alpha_TH = -5  # 這裡你可以自行調整 alpha 值
        alpha_TM = -5  # 這裡你可以自行調整 alpha 值    
        
        control_cost = self.R * tf.reduce_sum(tf.square(self.u_seq + alpha_TH * TH_pred[:self.H_c, :]+ alpha_TM*TM_pred[:self.H_c, :]+50))
    
        return state_cost, control_cost, state_cost + control_cost


    def optimize_control(self, TM0_np, TH0_np, TM_ref_np, iterations,dt):
        dt_val = dt         # 這裡假設 dt = 1（與 t_seq 取值一致）
        coff_u_real_val = 1.00 # 假設真實系統中使用的 coff_u_real 與訓練時一致
        for it in range(iterations):
            feed_dict = {
                self.TM0_tf: TM0_np,
                self.TH0_tf: TH0_np,
                self.TM_ref_tf: TM_ref_np
            }
            # 執行一次優化步驟
            self.sess.run(self.train_op, feed_dict=feed_dict)
        
            # Clip 控制輸入，並更新變量
            # clipped = tf.clip_by_value(self.u_seq, self.u_lb, self.u_ub)
            # self.sess.run(self.u_seq.assign(clipped))
        
            # 每隔一定次數打印 cost 值與 x_ref, x_pred, t_seq 數值，以及用 real_plant_dynamics 更新的狀態
            if it % iterations == 0:
        
                
                #-------------------
                # 在 optimize_control 的 if it % 10 == 0 分支中，增加如下代码：
                # 形成延伸的控制序列 u_extended (move-blocking)
                tail_num = self.H_p - self.H_c
                last_u = self.u_seq[-1, :]  # shape (1,)
                tail_u = tf.tile(tf.reshape(last_u, [1, 1]), [tail_num, 1])
                u_extended = tf.concat([self.u_seq, tail_u], axis=0)  # shape (H_p,1)
                
                # tile x0, v0 => shape (H_p,1)
                TM0_tiled = tf.tile(self.TM0_tf, [self.H_p, 1])
                TH0_tiled = tf.tile(self.TH0_tf, [self.H_p, 1])
                # 用 sess.run 输出这些变量
                t_seq_val, TM0_tiled_val, TH0_tiled_val, u_extended_val = self.sess.run(
                    [self.t_seq, TM0_tiled, TH0_tiled, u_extended],
                    feed_dict=feed_dict
                )              
                # 利用 net_output 计算 x_pred_tensor (只取第一個輸出)
                TM_pred_tensor, _ = self.model.net_output(t_seq_val, TM0_tiled_val, TH0_tiled_val, u_extended_val)
                t_seq_val, TM0_tiled_val, TH0_tiled_val, u_extended_val,TM_pred_val = self.sess.run(
                    [self.t_seq, TM0_tiled, TH0_tiled, u_extended,TM_pred_tensor],
                    feed_dict=feed_dict
                ) 
                
                state_cost_val, control_cost_val, total_cost_val, TM_pred_val, TM_ref_val, t_seq_val = self.sess.run(
                    [self.state_cost, self.control_cost, self.loss, TM_pred_tensor, self.TM_ref_tf, self.t_seq],
                    feed_dict=feed_dict)
                print(f"Iteration {it}:")
                print(f"  state_cost = {state_cost_val:.4e}, control_cost = {control_cost_val:.4e}, total_cost = {total_cost_val:.4e}")
                print(f"  t_seq = {t_seq_val.flatten()}")
                print(f"  TM_ref = {TM_ref_val.flatten()}")
                print(f"  TM_pred = {TM_pred_val.flatten()}")
    

                # 取出當前的 x0, v0, 以及控制 u（第一步的控制）
                TM0_val, TH0_val, u_val = self.sess.run(
                    [self.TM0_tf, self.TH0_tf, self.u_seq[0]],
                    feed_dict=feed_dict)
                
                # 轉換成標量
                TM0_val = TM0_val[0, 0]
                TH0_val = TH0_val[0, 0]
                u_val = u_val.item()  # 這行修正 TypeError
                
                # 輸出數值
                print(f"  TM0_val = {TM0_val:.8f}")
                print(f"  TH0_val = {TH0_val:.8f}")
                print(f"  u_val  = {u_val:.8f}")  # 這行不會報錯了

                
                # 利用 real_plant_dynamics 更新狀態（模擬一次 dt 時間步長）
                TM_new_sim, TH_new_sim = real_plant_dynamics(self.Ta, TM0_val, TH0_val, u_val, dt_val, coff_u_real_val, self.hM_CM, self.hH_CH)
                # 將結果轉換為標量
                TM_new_sim = float(TM_new_sim)
                TH_new_sim = float(TH_new_sim)
                print(f"  real_plant_dynamics update: x_new = {TM_new_sim:.8f}, TH_new = {TH_new_sim:.8f}")

                    
        return self.sess.run(self.u_seq, feed_dict=feed_dict)


def real_plant_dynamics(Ta, TM0, TH0, u, dt, coff_u_real, hM_CM, hH_CH):
    """
    模擬真實系統的更新，與 PINN 不同的 coff_u。
    使用解析解計算更新後的位置和速度：
      dv/dt = -x - v + coff_u_real * u
      dx/dt = v
    """  
    lambda_val = np.sqrt(hM_CM**2 - hM_CM*hH_CH + hH_CH**2)
    lambda1 = -(hM_CM + hH_CH) + lambda_val
    lambda2 = -(hM_CM + hH_CH) - lambda_val
    TM_eq = Ta + u/(3*hH_CH)
    TH_eq = Ta + 2*u/(3*hH_CH)
    M = TM0 - TM_eq   # shape: (N_trajectories, 1)
    H = TH0 - TH_eq   # shape: (N_trajectories, 1)
    C1 = ((2*hM_CM+lambda2)*M-hM_CM*H)/(lambda2-lambda1)
    C2 = M - C1
    TM_new = TM_eq + C1 * np.exp(lambda1 * dt) + C2 * np.exp(lambda2 * dt)  
    TH_new = TH_eq + (C1 * (2*hH_CH + lambda1)/hH_CH) * np.exp(lambda1 * dt) + (C2 * (2*hH_CH + lambda2)/hH_CH) * np.exp(lambda2 * dt)
    
    return TM_new, TH_new






