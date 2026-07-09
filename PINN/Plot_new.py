import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------
# 1. 加载模型和参数
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载新模型（无量纲化版本）
checkpoint = torch.load('pinn_model_nondim.pkl', map_location=device)

# 提取参数
layers = checkpoint['config']
Lx = checkpoint['Lx']
Ly = checkpoint['Ly']
U_ref = checkpoint['U_ref']          # 特征温度 (15 K)
t_ref = checkpoint['t_ref']          # 特征时间 (80 s)
# 也可读取 gamma, Q_nondim 等（绘图用不到）

# 重建网络（与训练时完全一致）
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            self.net.append(nn.SiLU())
        self.net.append(nn.Linear(layers[-2], layers[-1]))
    def forward(self, x):
        return self.net(x)

model = PINN(layers).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型加载成功！")

# ------------------------------
# 2. 生成绘图网格（物理坐标）
# ------------------------------
nx, ny = 100, 50
x_phys = np.linspace(0, Lx, nx)
y_phys = np.linspace(0, Ly, ny)
X_phys, Y_phys = np.meshgrid(x_phys, y_phys)

# 转换为无量纲坐标 (X = x/Lx, Y = y/Ly)
X = X_phys / Lx
Y = Y_phys / Ly
X_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

# 时间帧：模拟物理时间从 0 到 t_ref 秒，也可以改为 0~t_ref
t_phys_frames = np.linspace(0, 50.0, 250)  # 总帧数200

# ------------------------------
# 3. 制作动画（动态色标）
# ------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
cbar = None
def update(frame):
    global cbar
    if cbar is not None:
        cbar.remove()
    ax.clear()
    t_phys = t_phys_frames[frame]
    # 无量纲时间 tau = t_phys / t_ref
    tau = t_phys / t_ref
    tau_input = torch.ones_like(X_flat) * tau

    model.eval()
    with torch.no_grad():
        xyt = torch.cat([X_flat, Y_flat, tau_input], dim=1)
        Theta_pred = model(xyt).cpu().numpy().reshape(X.shape)
    # 转换为物理温度
    u_pred = Theta_pred * U_ref
    #print(f"最高温度: {u_pred.max():.3f}")

    # 调整色标
    # 中间 vmin ~ vmax 的温度，平滑地映射为 蓝-青-绿-黄-橙-红
    im = ax.contourf(X_phys, Y_phys, u_pred, levels=100, cmap='jet', vmin=0.0, vmax=4.0)
    ax.set_title(f'Temperature at t = {t_phys:.1f} s')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # 添加色标（每帧更新，否则会累积）
    # 添加新的colorbar
    cbar = fig.colorbar(im, ax=ax)
    return im,
# interval：帧之间的延迟（毫秒）
ani = animation.FuncAnimation(fig, update, frames=len(t_phys_frames), interval=200, blit=False)

# 保存 GIF
# fps （每秒帧数）越小，动画播放越慢，总时长越长,这种方式文件大小不变（因为帧数没变），只是每帧停留的时间变长了
ani.save('output.gif', writer='pillow', fps=10)
print("动画已保存为 output.gif")