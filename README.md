# huber-bilstm-gru-DA
Nudging-based Data Assimilation method for Error Correction coupled with Huber Loss and BiLSTM-GRU Hybrids
我们在研究中使用了多份 Python 脚本来分别演示和测试不同阶段的同化流程。关键文件示例如下：

da_bilstm-gru-1_sparse.py
代码示例 1：演示如何生成 Lorenz-96 模拟数据并利用 BiLSTM + GRU 进行训练与同化。该脚本包含自定义的 Huber 损失函数，并提供最基础的同化流程。

da_bilstm-gru-2_sparse.py、da_bilstm-gru-3_sparse.py、da_bilstm-gru-4_sparse.py、da_bilstm-gru-5_sparse.py
代码示例 2–5：从不同角度对同化过程进行扩展或测试，包括不同网络深度、神经元规模、训练轮数、以及观测配置的演示。脚本结构大体相似，但在网络搭建和训练参数上有所不同。

lorenz96_bilstm-gru_sparse.py
用于生成 Lorenz-96 真值轨迹与观测数据的参考脚本，包含 Lorenz-96 方程的 rhs 函数以及 RK4 积分器的实现，并演示如何构造观测噪声和先验误差。

在各脚本中，核心函数包括：

rhs(ne, u, forcing)：计算 Lorenz-96 方程右端项。
rk4(ne, dt, u, forcing)：利用四阶 Runge-Kutta 对 Lorenz-96 系统做一步积分。
create_training_data_lstm(...) 或 create_training_data(...)：将输入特征与标签转换为适合 LSTM/GRU 训练的 3D 结构。
神经网络模块（BiLSTM + GRU）：使用 Keras/TensorFlow 完成网络搭建、训练、推断等流程。
1.2 数据文件
lstm_data_sparse.npz
包含本研究的 Lorenz-96 模拟数据、观测数据以及初步先验轨迹。载入方式例如：
python
复制
编辑
data = np.load('lstm_data_sparse.npz')
utrue = data['utrue']  # 真实场 (ne, nt+1)
uobs  = data['uobs']   # 观测数据 (ne, nb+1)
uwe   = data['uwe']    # 有误差的先验 (ne, npe, nt+1)
具体结构可参考脚本中的调用示例。
1.3 关键依赖与环境说明
Python 版本：建议 3.7+
TensorFlow / Keras：用于搭建与训练 BiLSTM + GRU 网络；大多数脚本使用 tensorflow==2.x 或 tensorflow-gpu==2.x。
NumPy、SciPy：数值计算与积分。
scikit-learn：数据预处理（MinMaxScaler）、训练集划分（train_test_split）与评价指标（MSE、MAE、R²）。
若需 GPU 加速，请保证安装合适的 GPU 版 TensorFlow 并配置 CUDA 和 cuDNN 等。

S2. 代码使用说明
2.1 运行顺序
生成模拟数据（可选）
若需自行生成新的 Lorenz-96 数据，可先运行 lorenz96_bilstm-gru_sparse.py 以生成真值轨迹和观测数据，并保存为 lstm_data_sparse.npz。
训练与同化
运行 da_bilstm-gru-1_sparse.py（或其他同化示例脚本），该脚本会读取 lstm_data_sparse.npz 并搭建 BiLSTM + GRU 模型进行训练和推断。
训练完成后，将自动生成并保存 HDF5 格式的神经网络权重文件（例如 model_bilstm_gru.h5）。
接着脚本会加载训练好的模型，在时间步与观测频率匹配时进行同化校正，输出同化后的状态结果。
评价与可视化
在脚本内部会自动计算 MAE、MSE、RMSE、R² 等评价指标，并可绘制同化后场与真值场的差异等图。
2.2 参数设置
ne：Lorenz-96 状态维度（默认为 40）。
dt：时间步长（默认 0.005）。
tmax：总积分时间长度（默认 10.0）。
nf：观测时间步频率；即每隔 nf 步做一次观测。
me：观测向量数量；与 freq = ne / me 共同决定观测位置。
npe：样本或集合大小，控制训练集中先验轨迹的数量。
神经网络：主要通过脚本内的超参数指定（如 LSTM/GRU 单元数、epochs 训练轮次、batch_size 批大小等）。
2.3 模型结构与输出
神经网络结构：
使用双向 LSTM（Bidirectional(LSTM(...))）与 GRU（GRU(...)）相结合，并添加若干全连接层（Dense(...)）。
输出文件：
HDF5 格式的网络权重（例如 model_bilstm_gru.h5）；
.csv 格式的结果文件（如 t.csv, utrue.csv, uobs.csv, ulstm.csv 等），便于后续分析或可视化；
脚本可视化各指标曲线和同化前后场景图。
