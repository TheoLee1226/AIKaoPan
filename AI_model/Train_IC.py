# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:21:59 2025

@author: l2810
"""
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


#======================================================
# 1. 定義 PINN 模型 (保持原樣，新增 net_output 函式)
#======================================================
class PhysicsInformedNN:
    def __init__(self, t_train, TM0_train, TH0_train, u_train, TM_train, TH_train, layers, hH, CH, hM_CM, coff_u,Ta):
        self.t_train = t_train
        self.TM0_train = TM0_train
        self.TH0_train = TH0_train
        self.u_train = u_train
        self.TM_train = TM_train
        self.TH_train = TH_train

        # 物理參數

        self.hH = hH  # 阻尼係數
        self.CH = CH  # 質量
        self.hM_CM = hM_CM  # 彈性係數
        self.coff_u = coff_u  # u 的係數
        self.Ta = Ta  # 阻尼係數

        # 記錄邊界 (用於輸入特徵的歸一化)
        self.lb_t = t_train.min()
        self.ub_t = t_train.max()
        self.lb_TM0 = TM0_train.min()
        self.ub_TM0 = TM0_train.max()
        self.lb_TH0 = TH0_train.min()
        self.ub_TH0 = TH0_train.max()
        self.lb_u = u_train.min()
        self.ub_u = u_train.max()

        # --- TensorFlow placeholder ---
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])  # t
        self.TM0_tf = tf.placeholder(tf.float32, shape=[None, 1]) # 初始條件 x0
        self.TH0_tf = tf.placeholder(tf.float32, shape=[None, 1]) # 初始速度 v0
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])  # 控制輸入 u
        self.TM_tf = tf.placeholder(tf.float32, shape=[None, 1])  # 真實 x
        self.TH_tf = tf.placeholder(tf.float32, shape=[None, 1]) # 真實 v_x
        
        # 定義 TensorFlow Placeholder
        self.t_zero_tf = tf.placeholder(tf.float32, shape=[None, 1])  # 時間點
        self.TM0_zero_tf = tf.placeholder(tf.float32, shape=[None, 1])  # x0=0
        self.TH0_zero_tf = tf.placeholder(tf.float32, shape=[None, 1])  # v0=0
        self.u_zero_tf = tf.placeholder(tf.float32, shape=[None, 1])  # u=0
        
        # 初始化神經網絡
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # 使用 net_PDE() (含硬條件) 建構 PDE 殘差
        self.TM_pred, self.TH_pred, self.f1_pred, self.f2_pred = self.net_PDE(
            self.t_tf, self.TM0_tf, self.TH0_tf, self.u_tf
        )
        self.TM_pred_zero,self.TH_pred_zero,_,_ = self.net_PDE(self.t_zero_tf, self.TM0_zero_tf, self.TH0_zero_tf, self.u_zero_tf)        
        
        
        # 特殊條件：u=0, v0=0, x0=0
        t_zero = np.linspace(self.lb_t, self.ub_t, 50).reshape(-1, 1)  # 取 50 個時間點
        zero_input = np.zeros_like(t_zero)
        self.t_zero = t_zero
        self.zero_input = zero_input
        


        
        
        # --- 分別定義 Data Loss 、Constraint_Loss、 PDE Loss ---
        # Data Loss
        self.data_loss = ( 1*tf.reduce_mean(tf.square(self.TM_tf - self.TM_pred)) +
                           1*tf.reduce_mean(tf.square(self.TH_tf - self.TH_pred)) )
        # 讓 x_pred_zero 收斂到 0
        self.constraint_loss = (100 * tf.reduce_mean(tf.square(self.TM_pred_zero)) +
                                100*tf.reduce_mean(tf.square(self.TH_pred_zero)))
        # PDE Loss
        self.pde_loss = ( 100*tf.reduce_mean(tf.square(self.f1_pred)) +
                          100*tf.reduce_mean(tf.square(self.f2_pred)) )

        # 損失函數 = 數據損失 + 物理損失
        self.loss = self.data_loss + self.pde_loss + self.constraint_loss

        # 優化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op = self.optimizer.minimize(self.loss)

        # TensorFlow 會話
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def initialize_NN(self, layers):
        """ Xavier 初始化權重 """
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            in_dim = layers[l]
            out_dim = layers[l+1]
            W = tf.Variable(tf.random.truncated_normal(
                [in_dim, out_dim],
                stddev=np.sqrt(2.0/(in_dim+out_dim))
            ))
            b = tf.Variable(tf.zeros([1, out_dim], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def neural_net(self, T, TM0, TH0, U, weights, biases):
        """
        构建神经网络 (不含硬条件)，输入 (t, x0, v0, u)
        输出网络的 "基底" (X_pred, Vx_pred)
        """
        # 拼接輸入
        H = tf.concat([T, TM0, TH0, U], axis=1)
        
        # 做線性歸一化到 [-1, 1]
        lb = tf.convert_to_tensor([self.lb_t, self.lb_TM0, self.lb_TH0, self.lb_u], dtype=tf.float32)
        ub = tf.convert_to_tensor([self.ub_t, self.ub_TM0, self.ub_TH0, self.ub_u], dtype=tf.float32)
        lb = tf.reshape(lb, [1, -1])
        ub = tf.reshape(ub, [1, -1])

        H = 2.0 * (H - lb) / (ub - lb) - 1.0
        
        # 前向傳播
        num_layers = len(weights) + 1
        for i in range(num_layers - 2):
            W = weights[i]
            b = biases[i]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        # 輸出層
        W = weights[-1]
        b = biases[-1]
        out = tf.add(tf.matmul(H, W), b)  # shape: [batch, 2]
        TM_out = out[:, 0:1]
        TH_out = out[:, 1:2]
        return TM_out, TH_out

    def net_PDE(self, t, TM0, TH0, u):
        """
        ODE: x'(t) = vx
             vx'(t) + (d/m)*vx + (k/m)*x - coff_u*u = 0
        修改硬條件:
        1. 當 x0 = 0, v0 = 0, u = 0 時，x 和 vx 必須為 0
        2. 重新設計 x, vx 的表達方式
        """
        TM_neural_net, TH_neural_net = self.neural_net(t, TM0, TH0, u, self.weights, self.biases)
        
        # **新硬條件設計**: 讓 x0, v0, u 影響神經網路的預測
        condition = tf.cast(tf.equal(TM0, 0) & tf.equal(TH0, 0) & tf.equal(u, 0), tf.float32)
        
        TM = (1 - condition) * (TM0 + t * TM_neural_net)
        TH = (1 - condition) * (TH0 + t * TH_neural_net)
    
        # 求微分
        dTM_dt = tf.gradients(TM, t)[0]  # dx/dt
        dTH_dt = tf.gradients(TH, t)[0]  # dvx/dt
    
        # PDE 殘差
        f1 = dTM_dt - self.hM_CM*((TH-TM) +(self.Ta-TM)) 
        f2 = dTH_dt - (self.hH*((TM-TH) + (self.Ta-TH)) + self.coff_u*u)/self.CH
    
        return TM, TH, f1, f2
    
    def train(self, nIter):
        """
        訓練模型
        """
        tf_dict = {
            self.t_tf:  self.t_train,
            self.TM0_tf: self.TM0_train,
            self.TH0_tf: self.TH0_train,
            self.u_tf:  self.u_train,
            self.TM_tf:  self.TM_train,
            self.TH_tf: self.TH_train,
            
            # 傳入 u=0, x0=0, v0=0 的條件
            self.t_zero_tf: self.t_zero,
            self.TM0_zero_tf: self.zero_input,
            self.TH0_zero_tf: self.zero_input,
            self.u_zero_tf: self.zero_input
        }
        for it in range(nIter):
            self.sess.run(self.train_op, tf_dict)

            # 每 1000 次輸出一次 loss
            if it % 1000 == 0:
                data_loss_val, constraint_loss_val, pde_loss_val, total_loss_val = self.sess.run(
                    [self.data_loss, self.constraint_loss, self.pde_loss , self.loss],
                    feed_dict=tf_dict
                )
                print(f"Iter {it:6d} | Data Loss = {data_loss_val:.4e} | Constraint Loss = {constraint_loss_val:.4e} | PDE Loss = {pde_loss_val:.4e} | Total Loss = {total_loss_val:.4e}")

    def predict(self, t_star, TM0_star, TH0_star, u_star):
        """
        以 NumPy array 作為輸入，返回 x, vx (硬條件後)
        """
        feed_dict = {
            self.t_tf:  t_star,
            self.TM0_tf: TM0_star,
            self.TH0_tf: TH0_star,
            self.u_tf:  u_star
        }
        TM_out, TH_out = self.sess.run([self.TM_pred, self.TH_pred], feed_dict)
        return TM_out, TH_out

    def save_model(self, checkpoint_dir="PINN_ODE_output", checkpoint_name="pinn_model.ckpt"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = os.path.join(checkpoint_dir, checkpoint_name)
        save_path = self.saver.save(self.sess, path)
        print(f"模型已儲存至: {save_path}")
    
    def load_model(self, checkpoint_dir="PINN_ODE_output", checkpoint_name="pinn_model.ckpt"):
        path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(path + ".meta"):
            self.saver.restore(self.sess, path)
            print(f"成功加載模型: {path}")
        else:
            print(f"找不到指定的模型檔案: {path}")

    #=== 下面是「新增」的函式，供 MPC graph 內呼叫 ===#
    def net_output(self, t, TM0, TH0, u):
        """
        與 net_PDE 類似，但只回傳 x, vx（符號層 Tensor），統一硬條件處理。
        """
        TM_neural_net, TH_neural_net = self.neural_net(t, TM0, TH0, u, self.weights, self.biases)
        # 統一使用與 net_PDE 相同的硬條件處理：
        TM =  (TM0 + t * TM_neural_net)
        TH =  (TH0 + t * TH_neural_net)
        return TM, TH