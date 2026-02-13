import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    from train_model import PINN, device

    # 模型保存路径
    save_dir = "./saved_pinn_models"
    model_filename = "oscillation_circle.pth"
    save_path = os.path.join(save_dir, model_filename)


    def load_model():
        checkpoint = torch.load(save_path, map_location=device)

        # 加载模型
        model_config = checkpoint['model_config']
        model = PINN(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hidden=model_config['num_hidden'],
            activation=model_config['activation']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model


    model= load_model()
    # 测试模型预测能力
    w=1
    beta=0
    dk = 10
    dt = 0.01  # 和训练的保持一致
    t_test = 100  # 测试时长
    tt = np.linspace(0, t_test, int(t_test / dt))


    A1 = 2 * (1 - w ** 2 * dt ** 2 * 0.5) / (1 + beta * dt) + (beta * dt - 1) / (beta * dt + 1)
    A2 = (1 - beta * dt) / (beta * dt + 1) * dt
    A3 = (2 * (1 - w ** 2 * dt ** 2 * 0.5) / (1 + beta * dt) + (beta * dt - 1) / (beta * dt + 1) - 1) / dt
    A4 = (1 - beta * dt) / (beta * dt + 1)
    # 构造相空间旋转矩阵
    rot = np.array([[A1, A2], [A3, A4]])

    #x_np = (np.random.rand(1, 1))-0.5#调范围进行测试
    #v_np = (np.random.rand(1, 1))-0.5
    x_np = (np.array([[1]]))  # 固定点进行测试
    v_np = (np.array([[0]]))
    xv_test = np.array([x_np[0], v_np[0]])
    x_test = torch.tensor(x_np, dtype=torch.float32).to(device)
    v_test = torch.tensor(v_np, dtype=torch.float32).to(device)

    x_pre_list = []
    v_pre_list = []

    data_solve = []
    for i in range(1, int(t_test / dt) + 1):
        data_solve.append(xv_test)
        out_xv = rot @ xv_test
        xv_test = out_xv

    tpre = []
    with torch.no_grad():  # 评估模式关闭梯度计算，节省内存
        for i in range(1, int(t_test / dt / dk) + 1):
            x_pre_list.append(x_test[0, 0])
            v_pre_list.append(v_test[0, 0])
            x_pre, v_pre = model(x_test, v_test)
            x_test = x_pre
            v_test = v_pre
            tpre.append(i * dk * dt)

    data_solve = np.array(data_solve)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.array(x_pre_list), np.array(v_pre_list), 'b-', label='model prediction', linewidth=1)
    ax[0].plot(data_solve[:, 0], data_solve[:, 1], 'r--', label='analytical solution', linewidth=1)
    ax[0].plot(x_pre_list[0], v_pre_list[0], 'go', label='test dot', linewidth=1)
    ax[0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax[0].axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax[0].set_xlabel('X ', fontsize=12)
    ax[0].set_ylabel('v ', fontsize=12)
    ax[0].set_title('phase space')
    ax[0].legend()
    rect = patches.Rectangle(
        (-0.5, -0.5),  # 矩形左下角坐标
        1.0,
        1.0,
        linewidth=1,
        edgecolor='k',
        linestyle='--',
        facecolor='none',
        alpha=1
    )

    # 将矩形添加到子图上
    ax[0].add_patch(rect)
    ax[0].set_xlim(-2, 2)
    ax[0].set_ylim(-2, 2)

    ax[1].plot(tpre, np.array(x_pre_list), 'b-', label='model prediction', linewidth=1)
    ax[1].plot(tt, data_solve[:, 0], 'r--', label='analytical solution', linewidth=1)
    ax[1].set_xlabel('t ', fontsize=12)
    ax[1].set_ylabel('x ', fontsize=12)
    ax[1].legend()
    ax[1].set_xlim(0, t_test)
    ax[1].set_ylim(-2, 2)
    ax[1].set_title('time domain')
    plt.tight_layout()

    save_folder = "plots"
    if not os.path.exists(save_folder):
           os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "in_domain_2.png")
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.show()