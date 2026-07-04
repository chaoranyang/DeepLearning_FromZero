import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# 0. 设置设备（自动检测 GPU）
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")  #  cuda 或 cpu


# ------------------------------
# 1. 生成数据：y = x^2 + 噪声
# ------------------------------
np.random.seed(66)
M = 200
x = np.linspace(-2, 2, M)
y_true = x**2
noise = np.random.randn(M) * 0.05
y = y_true + noise

# 转换为 PyTorch 张量，形状 (M, 1)，并直接送到 GPU
x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)


# ------------------------------
# 2. 网络结构（1 -> 5 -> 5 -> 1）
# ------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

        # 使用 Xavier 正态初始化（与原代码的 scale = sqrt(2/(nin+nout)) 一致）
        nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc3.weight, gain=1.0)
        # 偏置初始化为 0
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)          # 输出层无激活（恒等）
        return x


# ------------------------------
# 3. 训练设置
# ------------------------------
# 关键：将模型参数也放到 GPU 上
model = Net().to(device)

eta = 0.01
epochs = 3000
print_interval = 500

# 损失函数：0.5 * MSE（与原始代码 total_loss/M 完全等价）
loss_fn = lambda pred, target: 0.5 * torch.mean((pred - target) ** 2)

optimizer = optim.SGD(model.parameters(), lr=eta)

loss_history = []

# 设置随机种子（保证 PyTorch 初始化的可复现性）
torch.manual_seed(66)

# ------------------------------
# 4. 训练（全批量梯度下降，所有运算在 GPU 上）
# ------------------------------
for epoch in range(epochs):
    optimizer.zero_grad()           # 清零梯度
    pred = model(x_t)               # 前向传播（数据已在 GPU）
    loss = loss_fn(pred, y_t)       # 计算损失（平均）
    loss.backward()                 # 反向传播（GPU 执行）
    optimizer.step()                # 更新参数

    loss_history.append(loss.item())

    if (epoch + 1) % print_interval == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


# ------------------------------
# 5. 预测与绘图（预测时需将数据送 GPU，结果取回 CPU 才能绘图!）
# ------------------------------
x_plot = np.linspace(-2, 2, 200)
# 预测数据也要放到 GPU
x_plot_t = torch.tensor(x_plot, dtype=torch.float32).reshape(-1, 1).to(device)

with torch.no_grad():
    y_plot_t = model(x_plot_t)          # GPU 推理
    y_plot = y_plot_t.cpu().numpy().flatten()  # 取回 CPU 并转为 numpy

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, s=5, alpha=0.6, label='Noisy data')
plt.plot(x_plot, x_plot**2, 'g-', label='True $y=x^2$')
plt.plot(x_plot, y_plot, 'r-', label='Prediction')
plt.legend()
plt.title('Curve Fitting (PyTorch GPU)')

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (×0.5)')
plt.title('Training Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('output_torch.png', dpi=300)