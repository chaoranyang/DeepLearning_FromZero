## 1. 项目简介

本代码采用物理信息神经网络（Physics-Informed Neural Networks, PINN）求解二维矩形区域内的瞬态热传导问题。该问题具有四个固定低温的方形热源，区域边界维持恒定低温，初始时刻整个区域的温度均匀高于环境温度。此类问题广泛存在于电子器件散热（如多热源芯片的冷却）、换热器设计、建筑热工分析以及金属热处理等工业场景中。

传统的数值方法（如有限元、有限体积法）需要生成网格并迭代求解，而 PINN 将控制方程、初始条件和边界条件作为物理约束嵌入神经网络的损失函数，通过自动微分计算偏导数，从而在无网格框架下获得连续可微的温度场近似解。本代码实现了PINN完整的无量纲化流程、残差计算、多任务加权损失函数以及“Adam 预热 + L‑BFGS 精细优化”的两阶段训练策略，并引入了时间因果权重以加速早期瞬态过程的学习。

---

## 2. 数学模型

### 2.1 原始定解问题（物理量）

设物理温度场为 $u(x,y,t)$ ，其中 $(x,y) \in [0, L_x] \times [0, L_y]$ ，时间 $t \in [0, t_{\text{ref}}]$ 。控制方程为带热源的非齐次热传导方程：

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + Q \cdot \mathbf{1}_{\Omega_h}(x,y) $$

其中 $\alpha$ 为热扩散系数（常数）， $Q$ 为热源强度（单位：K/s）， $\mathbf{1}_{\Omega_h}$ 为热源区域指示函数。热源区域 $\Omega_h$ 由四个尺寸为 $size \times size$ 的方形区域组成，分别布置在四个角附近，距离左右边界的间隙为 $gapx$ ，距离上下边界的间隙为 $gapy$ 。具体地，四个角区域的物理坐标范围为：

- 左下： $x \in [gapx, gapx+size]$ ， $y \in [gapy, gapy+size]$
- 右下： $x \in [L_x - gapx - size, L_x - gapx]$ ， $y \in [gapy, gapy+size]$
- 左上： $x \in [gapx, gapx+size]$ ， $y \in [L_y - gapy - size, L_y - gapy]$
- 右上： $x \in [L_x - gapx - size, L_x - gapx]$ ， $y \in [L_y - gapy - size, L_y - gapy]$

边界条件（四边恒温，相对于环境温度为 0）：
$$ u(0,y,t) = u(L_x,y,t) = u(x,0,t) = u(x,L_y,t) = 0 $$

初始条件（均匀高温）：
$$ u(x,y,0) = u_0 \quad (u_0 = 15) $$

### 2.2 无量纲化

定义特征温度 $U_{\text{ref}}$ 和特征时间 $t_{\text{ref}}$ 如下：

$$ U_{\text{ref}} = \frac{Q L_x L_y}{\alpha} , \qquad t_{\text{ref}} = 80 \ \text{s} $$

引入无量纲变量：
$$ \Theta = \frac{u}{U_{\text{ref}}} , \quad X = \frac{x}{L_x} , \quad Y = \frac{y}{L_y} , \quad \tau = \frac{t}{t_{\text{ref}}} $$

则无量纲控制方程为：
$$ \frac{\partial \Theta}{\partial \tau} = \frac{\alpha t_{\text{ref}}}{L_x^2} \frac{\partial^2 \Theta}{\partial X^2} + \frac{\alpha t_{\text{ref}}}{L_y^2} \frac{\partial^2 \Theta}{\partial Y^2} + Q_{\text{nondim}} \cdot \mathbf{1}_{\Omega_h}(X,Y) $$

其中无量纲热源强度 $Q_{\text{nondim}} = \frac{Q t_{\text{ref}}}{U_{\text{ref}}}$ 。在本代码的物理参数下（ $L_x=2.0,\ L_y=1.0,\ \alpha=0.05,\ Q=5.0$ ），计算得 $U_{\text{ref}} = \frac{5.0 \times 2.0 \times 1.0}{0.05} = 200$ ， $Q_{\text{nondim}} = \frac{5.0 \times 80}{200} = 2.0$ 。初始无量纲温度 $\Theta_{\text{init}} = \frac{u_0}{U_{\text{ref}}} = \frac{15}{200} = 0.075$ ，代码中 `init_nondim = init_temp / U_ref` 得到相同值。

边界条件在无量纲形式下为：
$$ \Theta(0,Y,\tau) = \Theta(1,Y,\tau) = \Theta(X,0,\tau) = \Theta(X,1,\tau) = 0 $$

初始条件为：
$$ \Theta(X,Y,0) = \Theta_{\text{init}} $$

热源区域在无量纲坐标下的判定方式为：首先将 $(X,Y)$ 映射回物理坐标 $(x = X L_x, \ y = Y L_y)$ ，再与上述四个物理区域比较，生成指示函数值（热源内为 1，否则为 0）。

---

## 3. 代码结构

代码按功能分为以下几个主要模块：

### 3.1 超参数与无量纲化
- 定义物理参数： $L_x, L_y, \alpha, u_0, Q, size, gapx, gapy$ 。
- 计算特征量 $U_{\text{ref}}$ 和 $t_{\text{ref}}$ ，并导出无量纲参数 $Q_{\text{nondim}}$ 和 $\Theta_{\text{init}}$ 。
- 设置神经网络结构（输入维度 3，隐藏层 3 层各 128 神经元，输出维度 1）、采样点数（ $N_f=20000$ ， $N_{ic}=2000$ ， $N_{bc}=2000$ ）、训练轮数（Adam 10000 轮，L‑BFGS 200 步）以及损失权重（ $\lambda_{IC}=1000$ ， $\lambda_{BC}=100$ ）。

### 3.2 神经网络定义（类 `PINN`）
- 继承自 `nn.Module` ，使用 `nn.Sequential` 堆叠全连接层，隐藏层激活函数为 `SiLU` （即 Swish），输出层为线性层（无激活函数）。
- 权重初始化：所有线性层采用 Xavier 正态初始化，偏置初始化为 0 ，但将最后一层（输出层）的偏置显式设为 $\Theta_{\text{init}}$ ，使网络初始预测接近初始温度，有助于加速收敛。

### 3.3 数据采样（函数 `sample_data`）
- **PDE 配点**：在无量纲区域 $[0,1]^2$ 内均匀采样 $(X_f, Y_f)$ ，时间 $\tau_f$ 也在 $[0,1]$ 均匀采样。
- **初始条件点**：固定 $\tau=0$ ， $(X_{ic}, Y_{ic})$ 在 $[0,1]^2$ 均匀采样，目标值 $\Theta_{ic} = \Theta_{\text{init}}$ （常数）。
- **边界条件点**：四条边分别采样 $N_{bc}/4$ 个点，每条边上的坐标沿边均匀随机（如左边 $X=0$ ， $Y$ 随机），时间 $\tau_{bc}$ 在 $[0,1]$ 均匀随机，目标值 $\Theta_{bc} = 0$ 。
- 所有张量移至指定设备（CUDA），并设置 `requires_grad=True` 用于 PDE 配点（便于自动微分）。

### 3.4 残差计算（函数 `compute_pde_residual`）
- 将 $(X,Y,\tau)$ 拼接后输入网络得到 $\Theta$ 。
- 通过 `torch.autograd.grad` 计算一阶时间导数 $\Theta_\tau$ 和二阶空间导数 $\Theta_{XX}, \Theta_{YY}$ （二阶导通过对一阶导再次求导获得）。
- 构造热源指示函数 `region` ：将 $(X,Y)$ 映射回物理坐标，使用逻辑运算判断是否落在四个角区域内，转换为浮点张量（0 或 1）。
- 计算无量纲 PDE 残差：
  $$ R = \Theta_\tau - \left( \frac{\alpha t_{\text{ref}}}{L_x^2} \Theta_{XX} + \frac{\alpha t_{\text{ref}}}{L_y^2} \Theta_{YY} \right) - Q_{\text{nondim}} \cdot \text{region} $$

### 3.5 损失函数（函数 `compute_losses`）
- **PDE 损失**：残差的加权均方误差，权重 $w(\tau) = 1 + 10 \exp(-\tau / 0.1)$ （时间因果权重，使早期时间点贡献更大）：
  $$ L_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} w(\tau_i) \cdot R_i^2 $$
- **初始条件损失**：均方误差乘以权重 $\lambda_{IC}$ ：
  $$ L_{\text{IC}} = \lambda_{IC} \cdot \frac{1}{N_{ic}} \sum_{j=1}^{N_{ic}} \left( \Theta_{\text{pred}}(X_{ic,j}, Y_{ic,j}, 0) - \Theta_{\text{init}} \right)^2 $$
- **边界条件损失**：均方误差乘以权重 $\lambda_{BC}$ ：
  $$ L_{\text{BC}} = \lambda_{BC} \cdot \frac{1}{N_{bc}} \sum_{k=1}^{N_{bc}} \left( \Theta_{\text{pred}}(X_{bc,k}, Y_{bc,k}, \tau_{bc,k}) - 0 \right)^2 $$
- 总损失： $L = L_{\text{PDE}} + L_{\text{IC}} + L_{\text{BC}}$ 。函数同时返回总损失及各分量（用于打印）。

### 3.6 训练流程（Adam + L‑BFGS）
- **阶段 1 – Adam 热启动**：使用 Adam 优化器（学习率 $5\times 10^{-3}$ ）训练 10000 轮，每 500 轮打印损失。此阶段旨在快速降低 IC 和 BC 损失，避免后续 L‑BFGS 因初始梯度过大而震荡。
- **阶段 2 – L‑BFGS 精细收敛**：使用 L‑BFGS 优化器（步长因子 0.8，最大线搜索迭代 20 次）进行 200 步迭代。每 20 步打印总损失及各分量。L‑BFGS 通过闭包函数 `closure` 计算损失并执行反向传播，可自动调整学习率并实现更精确的收敛。

### 3.7 模型保存
- 将训练好的模型状态字典（`state_dict`）以及无量纲参数（ $U_{\text{ref}}, t_{\text{ref}}, L_x, L_y, \alpha$ ）保存为 `pinn_model_nondim.pkl` 文件，供后续推理和反无量纲化（物理温度 $u = \Theta \cdot U_{\text{ref}}$ ，物理时间 $t = \tau \cdot t_{\text{ref}}$ ）使用。

---
## 4. 损失函数与权重说明

本代码的损失函数由三项组成：PDE 残差损失、初始条件损失和边界条件损失。各项均基于均方误差（MSE），并分别赋予不同的权重系数以平衡量级。

### 4.1 PDE 残差损失 $L_{\text{PDE}}$

对于每一个 PDE 配点 $(X_i, Y_i, \tau_i)$ ，神经网络预测 $\Theta_i$ 代入无量纲控制方程后得到残差 $R_i$ ：

$$ R_i = \frac{\partial \Theta}{\partial \tau}\bigg|_{(X_i,Y_i,\tau_i)} - \left[ \frac{\alpha t_{\text{ref}}}{L_x^2} \frac{\partial^2 \Theta}{\partial X^2}\bigg|_{(X_i,Y_i,\tau_i)} + \frac{\alpha t_{\text{ref}}}{L_y^2} \frac{\partial^2 \Theta}{\partial Y^2}\bigg|_{(X_i,Y_i,\tau_i)} \right] - Q_{\text{nondim}} \cdot \mathbf{1}_{\Omega_h}(X_i,Y_i) $$

其中 $\mathbf{1}_{\Omega_h}$ 为热源指示函数（在四个方形热源区域内为 1，否则为 0）。  
为了加速早期瞬态过程的学习，引入时间因果权重 $w(\tau_i)$ ：

$$ w(\tau_i) = 1 + 10 \exp\left( -\frac{\tau_i}{0.1} \right) $$

该权重在 $\tau=0$ 时取值为 11 ，随着 $\tau$ 增大快速衰减至约 1.006（ $\tau=1$ 时），从而迫使网络优先拟合初始时刻附近的温度变化。  
因此，PDE 损失定义为：

$$ L_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} w(\tau_i) \cdot R_i^2 $$

### 4.2 初始条件损失 $L_{\text{IC}}$

在初始时刻 $\tau=0$ 处，神经网络预测值 $\Theta_{\text{pred}}(X_{ic,j}, Y_{ic,j}, 0)$ 应等于给定的初始温度 $\Theta_{\text{init}}$ 。损失为均方误差，并乘以权重 $\lambda_{\text{IC}}$ ：

$$ L_{\text{IC}} = \lambda_{\text{IC}} \cdot \frac{1}{N_{ic}} \sum_{j=1}^{N_{ic}} \left( \Theta_{\text{pred}}(X_{ic,j}, Y_{ic,j}, 0) - \Theta_{\text{init}} \right)^2 $$

其中 $\lambda_{\text{IC}} = 1000$ 。

### 4.3 边界条件损失 $L_{\text{BC}}$

在四条边界（ $X=0$ 、 $X=1$ 、 $Y=0$ 、 $Y=1$ ）上，任意时刻 $\tau$ 的无量纲温度应为 0 。损失为均方误差，并乘以权重 $\lambda_{\text{BC}}$ ：

$$ L_{\text{BC}} = \lambda_{\text{BC}} \cdot \frac{1}{N_{bc}} \sum_{k=1}^{N_{bc}} \left( \Theta_{\text{pred}}(X_{bc,k}, Y_{bc,k}, \tau_{bc,k}) - 0 \right)^2 $$

其中 $\lambda_{\text{BC}} = 100$ 。

### 4.4 总损失

总损失为上述三项之和：

$$ L = L_{\text{PDE}} + L_{\text{IC}} + L_{\text{BC}} $$

在训练过程中，每轮迭代均计算总损失及其各分量，便于监控各约束的收敛情况。

---

## 5. 训练细节

训练分为两个阶段，采用不同的优化器以兼顾全局收敛速度与局部精度。

### 5.1 阶段一：Adam 热启动
- **优化器**： `torch.optim.Adam` ，学习率 $\eta = 5 \times 10^{-3}$ 。
- **迭代轮数**： $10000$ 轮。
- **目的**：快速降低初始条件和边界条件的误差，使网络参数进入一个较优的区域，避免后续 L‑BFGS 因初始梯度过大而震荡。
- **监控**：每 500 轮打印总损失及三项分量（PDE、IC、BC）。

### 5.2 阶段二：L‑BFGS 精细收敛
- **优化器**： `torch.optim.LBFGS` ，步长因子 `lr=0.8` ，最大线搜索迭代 `max_iter=20` ，梯度容差 `tolerance_grad=1e-9` ，变化容差 `tolerance_change=1e-12` 。
- **迭代步数**： $200$ 步。
- **实现方式**：L‑BFGS 需要定义闭包函数 `closure()` ，该函数内部清零梯度、计算总损失、执行反向传播并返回损失值。优化器在每一步中调用该闭包并更新参数。
- **监控**：每 20 步打印总损失及三项分量。

### 5.3 随机性控制
- 使用 `torch.manual_seed(26)` 固定所有随机操作（采样、权重初始化等），确保结果可重复。

---

## 6. 结果可视化与后处理

训练完成后，模型及无量纲参数被保存为 `pinn_model_nondim.pkl` 。用户可通过加载该文件，输入任意无量纲坐标 $(X,Y,\tau)$ 得到预测的无量纲温度 $\Theta$ ，再乘以 $U_{\text{ref}}$ 还原为物理温度 $u$ ，物理时间 $t = \tau \cdot t_{\text{ref}}$ 。以下说明后处理与动画生成的流程及关键实现。

### 6.1 模型加载与参数恢复
首先从保存的 `.pkl` 文件中加载模型状态字典和超参数，包括网络结构配置 `config` 、物理尺寸 $L_x, L_y$ 、特征温度 $U_{\text{ref}}$ 和特征时间 $t_{\text{ref}}$ 。重建与训练时结构完全相同的 PINN 网络（输入 3 维，输出 1 维，隐藏层激活函数为 `SiLU` ），并载入训练好的权重。此步骤确保预测结果与训练环境一致。

### 6.2 预测网格生成
在物理区域 $[0, L_x] \times [0, L_y]$ 内生成均匀网格点（例如 $100 \times 50$ ），得到物理坐标矩阵 $(x_{\text{phys}}, y_{\text{phys}})$ 。随后将其转换为无量纲坐标 $X = x_{\text{phys}} / L_x$ ， $Y = y_{\text{phys}} / L_y$ ，并展平为张量形式，留待批量输入网络。

### 6.3 时间序列预测
定义需要可视化的物理时间序列（例如从 0 到 50.0 秒，共 200 帧）。对于每一帧，将当前物理时间 $t_{\text{phys}}$ 转换为无量纲时间 $\tau = t_{\text{phys}} / t_{\text{ref}}$ ，并与网格的 $(X, Y)$ 拼接成形状为 $(N_{\text{grid}}, 3)$ 的输入张量，一次性送入模型得到所有网格点上的 $\Theta$ 预测值。将 $\Theta$ 乘以 $U_{\text{ref}}$ 即可得到物理温度场 $u$ 。

### 6.4 动画生成
利用 `matplotlib` 的 `contourf` 绘制每一帧的温度云图，并使用 `animation.FuncAnimation` 将这些帧组合成动态动画。色标范围固定（例如 0 至 4.0 K），以便于对比不同时刻的温度分布。最终动画可通过 `ani.save` 保存为 GIF 格式（使用 Pillow 驱动），帧率及间隔可调。该过程完全在 `torch.no_grad()` 上下文中进行，避免梯度计算开销。

### 6.5 后处理要点
- **反无量纲化**：物理温度 $u = \Theta \cdot U_{\text{ref}}$ ，物理时间 $t = \tau \cdot t_{\text{ref}}$ 。此步骤在绘图前完成，使输出结果具有物理意义。
- **色标调整**：根据实际模拟的最高温度，可调整 `vmin` 和 `vmax` 以优化显示效果。
- **性能优化**：将全部网格点一次性输入网络进行预测，比逐点循环更高效；动画生成时，每帧只进行一次前向传播，可大幅减少计算时间。

通过上述后处理流程，用户可以直观地观察温度场从初始均匀高温状态开始，在四个恒定低温热源和边界冷却作用下逐渐演化的全过程，从而验证 PINN 求解瞬态热传导问题的有效性。

## 7. 运行环境与依赖

- **Python 版本**： ≥ 3.8
- **核心库**：
  - PyTorch ≥ 1.8.0（支持自动微分和 GPU 加速）
  - NumPy
  - Matplotlib（用于后处理可视化）
  - Pillow（用于保存 GIF 动画）
- **硬件建议**：CUDA 兼容的 GPU 可显著加速训练，若无则自动回退至 CPU。

可直接运行主训练脚本（所有代码均在同一个文件中），无需额外配置。训练过程中会生成模型权重文件 `pinn_model_nondim.pkl` ，后处理脚本需与该文件位于同一目录。

---

## 8. 注意事项

1. **非均匀时间采样**：代码中注释了幂变换采样（ `tau_f = torch.rand(N_f, 1)**0.3` ），旨在使更多配点集中在早期时间，但该操作与后续的自动微分链式法则冲突，会导致 `loss.backward()` 报错，因此当前采用均匀采样。用户若需自定义时间分布，需谨慎处理计算图。

2. **权重初始化**：输出层偏置被初始化为 $\Theta_{\text{init}}$ （0.075），使初始预测接近初始条件，有助于早期收敛。若修改物理参数或初始温度，需相应调整该值。

3. **损失权重**： $\lambda_{\text{IC}}$ 和 $\lambda_{\text{BC}}$ 分别设为 1000 和 100，这是为了平衡三个损失的数量级，并弥补 IC 和 BC 采样点较少（相对 PDE 配点）的影响。实际应用时可根据各损失的量级动态调整。

4. **色标范围**：后处理动画中固定色标范围为 0~4.0，该值基于预估的最高温度设定。若物理参数改变，最高温度可能变化，需调整 `vmax` 以获得更好的可视化效果。

5. **模型保存内容**：保存文件中包含网络结构配置和无量纲参数，但不包含优化器状态或训练历史。加载时需手动重建网络并载入权重。
