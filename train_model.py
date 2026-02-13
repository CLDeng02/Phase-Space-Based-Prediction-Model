import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")


class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=50, num_hidden=2, activation='tanh'):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        else:
            raise ValueError('Invalid activation')

    def forward(self, x, v):
        input_combined = torch.cat([x, v], dim=0)  #(2,k)
        out = input_combined.T    #输入维度是2，对应(x,v)特征，需要转置成(k,2),注意当3阶张量时不要转置
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)   #(k,2)

        x_pre=out[:,0:-1]   #取x特征(k,1)
        v_pre=out[:,-1:]
        return x_pre, v_pre

    def losses(self, x, v, xx, vv):
        x_pre, v_pre = self.forward(x, v)
        res = (xx.flatten()-x_pre.flatten())**2+(vv.flatten()-v_pre.flatten())**2  #展成1维张量计算误差
        loss = torch.mean(res)

        if not hasattr(self, 'history'):
            self.history = {'loss': [],}

        self.history['loss'].append(loss.item())
        return loss

def train_pinn_adam(pinn, x_train, v_train,xx_train, vv_train, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    print("开始Adam训练")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn.losses(x_train, v_train, xx_train, vv_train)  # 修正为 losses
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 500 == 0:
            print(f"轮次{epoch}:总损失={loss.item():.6e},")
    print("Adam训练完成")

def plot_loss_history(pinn):
    if not hasattr(pinn, 'history'):
        print("NO loss history available")
        return

    epochs = range(len(pinn.history['loss']))
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, pinn.history['loss'])
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(True)
    plt.tight_layout()
    save_folder = "plots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "Loss.png")
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.show()

def get_data(total_time, dt, N, k, dk, beta, w):
    # 中心二阶差分得到的系数
    A1 = 2 * (1 - w ** 2 * dt ** 2 * 0.5) / (1 + beta * dt) + (beta * dt - 1) / (beta * dt + 1)
    A2 = (1 - beta * dt) / (beta * dt + 1) * dt
    A3 = (2 * (1 - w ** 2 * dt ** 2 * 0.5) / (1 + beta * dt) + (beta * dt - 1) / (beta * dt + 1) - 1) / dt
    A4 = (1 - beta * dt) / (beta * dt + 1)
    # 构造相空间旋转矩阵
    rot = np.array([[A1, A2], [A3, A4]])
    # 进行运算
    input_xv = np.random.rand(2, k) - 0.5
    data_xv = []
    for i in range(1, int(N) + 1):
        data_xv.append(input_xv)    #(N,2,k)
        output_xv = rot @ input_xv
        input_xv = output_xv

    x_coords_np = np.array([vec[0, :] for vec in data_xv])  #(N,k)  k个点x随时间的变化序列
    v_coords_np = np.array([vec[1, :] for vec in data_xv])  #(N,k)  k个点v随时间的变化序列
    x_coords_tensor = torch.tensor(x_coords_np, dtype=torch.float32).to(device)
    v_coords_tensor = torch.tensor(v_coords_np, dtype=torch.float32).to(device)
    x_in_train = x_coords_tensor[0:1,:]    #取起始点并且保留维度 (1,k)
    v_in_train = v_coords_tensor[0:1,:]
    x_out_train = x_coords_tensor[dk:dk+1,:]
    v_out_train = v_coords_tensor[dk:dk+1,:]

    return rot,x_coords_np,v_coords_np,x_in_train, v_in_train, x_out_train, v_out_train

def plot_sampling_points(x_coords, v_coords, x_in_train, v_in_train, x_out_train, v_out_train):
    plt.figure(figsize=(8, 8))
    # 相空间流形
    plt.plot(x_coords, v_coords, 'b-', linewidth=1)
    plt.plot(x_in_train[0,:], v_in_train[0,:], 'r.',label='input dot', linewidth=1)
    plt.plot(x_out_train[0,:], v_out_train[0,:], 'g.',label='output dot', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('X ', fontsize=12)
    plt.ylabel('v ', fontsize=12)
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    plt.title('phase space')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    save_folder = "plots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "sampling.png")
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.show()


if __name__ == "__main__":
    # 弹簧小球谐振子系统参数设置
    w = 1  # 角频率
    beta = 0.1 # 阻尼系数，合理区间在0-w，通常设为0.1
    total_time = 10  # 训练时长
    dt = 0.01  # 时间步长
    N = total_time / dt
    k = 2000  # 随机采样点数
    dk = 10  # 采样间隔
    # 生成随机点
    rot, x_coords, v_coords, x_in_train, v_in_train, x_out_train, v_out_train = get_data(total_time, dt, N, k, dk, beta,
                                                                                         w)
    # 相空间
    plot_sampling_points(x_coords, v_coords, x_in_train, v_in_train, x_out_train, v_out_train)

    # 模型参数设置
    model = PINN(
        input_dim=2, output_dim=2, hidden_dim=10, num_hidden=2, activation='tanh'
    ).to(device)

    t1 = default_timer()
    train_pinn_adam(model, x_in_train, v_in_train, x_out_train, v_out_train, epochs=20000, lr=0.001)
    t2 = default_timer()
    print(f"耗时：{t2 - t1}s")

    plot_loss_history(model)

    # 保存路径
    save_dir = "./saved_pinn_models"  # 模型保存文件夹
    model_filename = "oscillation.pth"  # 模型文件名
    save_path = os.path.join(save_dir, model_filename)

    # 创建文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型参数
    torch.save({
        'model_state_dict': model.state_dict(),  # 模型权重
        'model_config': {  # 保存模型配置，方便加载时重建
            'input_dim': 2,
            'output_dim': 2,
            'hidden_dim': 10,
            'num_hidden': 2,
            'activation': 'tanh'
        },
        'training_params': {  # 可选：保存训练参数，方便追溯
            'w': w,
            'beta': beta,
            'epochs': 20000,
            'lr': 0.001
        }
    }, save_path)

    print(f"模型已保存到: {save_path}")