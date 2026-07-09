import torch as tc
import torch.nn as nn
import time

#------------------------------------- 1. 超参数与无量纲化 -------------------------------------
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 原始物理参数（可以任意虚构，但量纲需自洽） 
Lx, Ly = 2.0, 1.0            # 物理尺寸 [m]
alpha = 0.05                 # 热扩散系数 [m^2/s]
init_temp = 15.0             # 初始温差（相对环境） [K]
Q_physical = 5.0             # 物理热源强度 [K/s]
# 热源几何参数（物理坐标）
size = 0.2                   # 热源方形尺寸
gapx = 0.4                   # 与左右边界间隙
gapy = 0.2                   # 与上下边界间隙

# 特征量，计算无量纲数 
U_ref = Q_physical * Lx * Ly / alpha #以热源决定的温度作为特征温度 = 5 * 2 * 1 / 0.05 = 200
t_ref = 80.0       # 特征时间 
Q_nondim = Q_physical * t_ref / U_ref  # 无量纲热源 (=1)
init_nondim = init_temp / U_ref   

print(f"（无量纲）：热源强度 Q~ = {Q_nondim:.4f},初始温度Theta_init = {init_nondim:.4f} ")
print(f"特征时间 t_ref = {t_ref:.2f} 秒 (模拟将覆盖 0~{t_ref:.2f} 秒)")

# 神经网络参数
config = [3, 128, 128, 128, 1]   # 输入 (X, Y, tau)，输出 Theta
N_f = 20000                
N_ic = 2000                
N_bc = 2000                

# 训练参数
adam_epochs = 10000        # Adam 预热轮数 
lbfgs_steps = 200         # L-BFGS 精细迭代步数 
eta = 5e-3               # Adam 学习率（L-BFGS 不使用）
tc.manual_seed(26)
lmd_ic = 1000.0         # 在总loss里的权重
lmd_bc = 100.0  



#------------------------------------- 2. 神经网络定义 -------------------------------------
class PINN(nn.Module):
    def __init__(self, config):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(config) - 2):
            self.net.append(nn.Linear(config[i], config[i+1]))
            self.net.append(nn.SiLU())
        #最后一层是纯线性层，无激活函数
        self.net.append(nn.Linear(config[-2], config[-1]))
        
        # 初始化权重
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # 将最后一层偏置设为 1，让初始化预测接近 Theta=1（符合初始条件）
                if m == self.net[-1]:
                    nn.init.constant_(m.bias, init_nondim ) # 直接把NN初始化到初始条件附近
                else:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
# 在科学机器学习的标准实践中，输出层永远保持线性（不加任何激活函数），除非先验地知道解被严格限制在某个区间内
model = PINN(config).to(device)



#------------------------------------- 3. 采样数据 -------------------------------------
def sample_data():
    # PDE 配点：X, Y 全部在 [0, 1] 区间均匀采样  
    X_f = tc.rand(N_f, 1, requires_grad=True)
    Y_f = tc.rand(N_f, 1, requires_grad=True)
    
    tau_f = tc.rand(N_f, 1, requires_grad=True)
    # tau_f = tc.rand(N_f, 1, requires_grad=True)**0.3  
    # 幂变换采样(^0.3)， 可以使得60% 的点会落在 [0, 0.3] 区间，试图让PINN学会早期的剧烈变化
    # 但这会引起loss.backward()的不明原因报错，因为非均匀采样更改了自动微分的计算图，使得后面的简单代码不能处理

    # 初始条件 (tau=0)：
    X_ic = tc.rand(N_ic, 1)
    Y_ic = tc.rand(N_ic, 1)
    tau_ic = tc.zeros(N_ic, 1)
    Theta_ic = tc.ones(N_ic, 1)* init_nondim           # 目标值 = 1.0

    # 边界条件 (四条边 Theta=0, tau 随机)
    tau_bc = tc.rand(N_bc, 1)            # 0~1 之间随机
    n_per_edge = N_bc // 4

    # 四条边的坐标
    X_left = tc.zeros(n_per_edge, 1)     # 左边界 x=0
    Y_left = tc.rand(n_per_edge, 1)      # 左边 y 随机采样以全覆盖
    X_right = tc.ones(n_per_edge, 1)     # 右边界 x=1
    Y_right = tc.rand(n_per_edge, 1)
    X_bottom = tc.rand(n_per_edge, 1)    #下边界
    Y_bottom = tc.zeros(n_per_edge, 1)   
    X_top = tc.rand(n_per_edge, 1)       #上边界
    Y_top = tc.ones(n_per_edge, 1)
    
    X_bc = tc.cat([X_left, X_right, X_bottom, X_top], dim=0)
    Y_bc = tc.cat([Y_left, Y_right, Y_bottom, Y_top], dim=0)
    tau_bc = tc.cat([tau_bc[:n_per_edge], tau_bc[:n_per_edge], tau_bc[:n_per_edge], tau_bc[:n_per_edge]], dim=0)
    Theta_bc = tc.zeros(N_bc, 1)         # 目标值 = 0.0

    return (X_f.to(device), Y_f.to(device), tau_f.to(device),
            X_ic.to(device), Y_ic.to(device), tau_ic.to(device), Theta_ic.to(device),
            X_bc.to(device), Y_bc.to(device), tau_bc.to(device), Theta_bc.to(device))

X_f, Y_f, tau_f, X_ic, Y_ic, tau_ic, Theta_ic, X_bc, Y_bc, tau_bc, Theta_bc = sample_data()



#------------------------------------- 4. 残差计算 -------------------------------------
def compute_pde_residual(model, X, Y, tau):
    # X, Y, tau 均为无量纲坐标 [0,1]
    XYT = tc.cat([X, Y, tau], dim=1)
    Theta = model(XYT)

    # 直接对无量纲坐标求导（结果就是物理无量纲导数）
    Theta_tau = tc.autograd.grad(Theta, tau, grad_outputs=tc.ones_like(Theta),
                                 retain_graph=True, create_graph=True)[0]
    Theta_XX = tc.autograd.grad(Theta, X, grad_outputs=tc.ones_like(Theta),
                                retain_graph=True, create_graph=True)[0]
    Theta_YY = tc.autograd.grad(Theta, Y, grad_outputs=tc.ones_like(Theta),
                                retain_graph=True, create_graph=True)[0]
    Theta_XX = tc.autograd.grad(Theta_XX, X, grad_outputs=tc.ones_like(Theta_XX),
                                retain_graph=True, create_graph=True)[0]
    Theta_YY = tc.autograd.grad(Theta_YY, Y, grad_outputs=tc.ones_like(Theta_YY),
                                retain_graph=True, create_graph=True)[0]

    # 热源项 Q 遮罩（依然在物理坐标下判断，但换算非常简单） -----
    x_phys = X * Lx
    y_phys = Y * Ly
    
    # 四个角内部热源（离边界缩进一点）
    # 左下
    corner1 = (x_phys >= gapx) & (x_phys <= gapx + size) & (y_phys >= gapy) & (y_phys <= gapy + size)
    # 右下
    corner2 = (x_phys >= Lx - gapx - size) & (x_phys <= Lx - gapx) & (y_phys >= gapy) & (y_phys <= gapy + size)
    # 左上
    corner3 = (x_phys >= gapx) & (x_phys <= gapx + size) & (y_phys >= Ly - gapy - size) & (y_phys <= Ly - gapy)
    # 右上
    corner4 = (x_phys >= Lx - gapx - size) & (x_phys <= Lx - gapx) & (y_phys >= Ly - gapy - size) & (y_phys <= Ly - gapy)

    region = (corner1 | corner2 | corner3 | corner4).float()
    
    # 核心：无量纲化的方程 
    # 方程：Theta_tau =  (alpha*t_ref/Lx^2)*Theta_XX + (alpha*t_ref/Ly^2)*Theta_YY + Q_nondim * region

    residual = Theta_tau - ((alpha*t_ref/Lx**2) * Theta_XX + (alpha*t_ref/Ly**2) * Theta_YY) - Q_nondim * region
    return residual



#------------------------------------- 5. 损失函数（IC 和 BC 目标值变为纯 0 和 1） -------------------------------------
def compute_losses(model):
    residual = compute_pde_residual(model, X_f, Y_f, tau_f)  # PDE 损失 

    lmd_Causality = 1.0 + 10.0 * tc.exp(-tau_f / 0.1)   # tau=0 时权重 10，tau=1 时权重 ~1.006
    # “时间因果权重”（Temporal Causality Weighting）。时间在前面的损失大，优先下降，强迫PINN学会早期剧烈变化
    loss_pde = tc.mean(lmd_Causality*(residual**2))

    XYT_ic = tc.cat([X_ic, Y_ic, tau_ic], dim=1)            # 初始条件损失
    Theta_pred_ic = model(XYT_ic)
    loss_ic = tc.mean((Theta_pred_ic - Theta_ic)**2)*lmd_ic   

    XYT_bc = tc.cat([X_bc, Y_bc, tau_bc], dim=1)            # 边界条件损失
    Theta_pred_bc = model(XYT_bc)
    loss_bc = tc.mean((Theta_pred_bc - Theta_bc)**2) *lmd_bc  

    #在多任务学习中，标准做法是让各项loss的数值量级尽量接近，使得BP的梯度来自各项的比例大致相同
    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc



#------------------------------------- 6. 训练（Adam + L-BFGS） -------------------------------------
print("开始训练...")
start = time.time()

# 阶段 1: Adam 热启动 
# 目的：快速让 IC 和 BC 降到合理范围，防止 L-BFGS 初期震荡
optimizer_adam = tc.optim.Adam(model.parameters(), lr=eta)
print(f"阶段 1/2: Adam 热启动 ({adam_epochs} epochs)...")

for epoch in range(adam_epochs + 1):
    optimizer_adam.zero_grad()
    loss, loss_pde, loss_ic, loss_bc = compute_losses(model)
    loss.backward()
    optimizer_adam.step()

    if epoch % 500 == 0:
        print(f'Adam Epoch {epoch:5d} | 总Loss: {loss.item():.4e} | '
              f'PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | BC: {loss_bc.item():.4e}')

# 阶段 2: L-BFGS
print(f"\n阶段 2/2: L-BFGS 高精度收敛 ({lbfgs_steps} steps)...")
optimizer_lbfgs = tc.optim.LBFGS(
    model.parameters(),
    lr=0.8,                      # L-BFGS 的步长因子
    max_iter=20,                 # 每次调用的最大线搜索次数
    max_eval=25,
    tolerance_grad=1e-9,         # 梯度阈值
    tolerance_change=1e-12
)
# 用于存储闭包内计算的损失分量（供打印使用）
loss_components = [0.0, 0.0, 0.0]         # [pde, ic, bc]

for step in range(lbfgs_steps + 1):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, lp, li, lb = compute_losses(model)
        # 存储各分量数值（转化为 Python float）
        loss_components[0] = lp.item()
        loss_components[1] = li.item()
        loss_components[2] = lb.item()
        # 反向传播，计算所有梯度
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    if step % 20 == 0:
        print(f'L-BFGS Step {step:3d} | 总Loss: {loss_components[0] + loss_components[1] + loss_components[2]:.4e} | '
              f'PDE: {loss_components[0]:.4e} | IC: {loss_components[1]:.4e} | BC: {loss_components[2]:.4e}')

waste_time = time.time() - start
minutes = waste_time // 60
seconds = waste_time % 60
print(f"训练完成！总耗时: {int(minutes)}分{seconds:.1f}秒")



#------------------------------------- 7. 保存模型和反无量纲化参数（便于画物理图） -------------------------------------
save_dict = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'U_ref': U_ref,          # 物理温度 = Theta * U_ref
    't_ref': t_ref,          # 物理时间 = tau * t_ref
    'Lx': Lx,
    'Ly': Ly,
    'alpha': alpha,
    'device': device.type,
}
tc.save(save_dict, 'pinn_model_nondim.pkl')
print("模型和参数已保存到 pinn_model_nondim.pkl")
print("\n【画图提示】预测时输入 (X, Y, tau)，输出 Theta。")
print(f"物理温度 = Theta * {U_ref} K")
print(f"物理时间 = tau * {t_ref:.2f} 秒")