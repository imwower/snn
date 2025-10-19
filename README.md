# SNN‑3Comp‑FPT（事件驱动）

> 三隔室 SNN（soma/顶树突/基底树突）× 固定点并行训练（Fixed‑point Parallel Training, FPT）× 事件驱动（默认使用NATS JetStream 消息总线）的参考实现蓝图与脚手架。

## 0) TL;DR

- 三隔室神经元 SNN（soma/顶树突/基底树突）
- FPT：把传统按时间步 𝑇展开的 SNN 训练从 𝑂(𝑇)降到 𝑂(𝐾)（常取小常数，如 K≈3），做法是把 LIF 等时序神经元写成**固定点方程**，用少量并行迭代逼近整段时间序列的解；保持原模型结构并给出收敛分析与实证。
- 事件驱动：训练拆成“事件”，默认使用 NATS JetStream 做持久化流、拉式消费、至少一次投递、可回放、发布去重（Nats‑Msg‑Id + 去重窗口）。可以通过配置文件修改为其他消息队列中间件。

## 1) 核心理念
### 1.1 FPT：用固定点并行替代时间展开

- 将 LIF/SNN 的时序动力学改写为 𝑢 = Φ(𝑢) 的固定点形式；在每个 batch 上做 K 次并行迭代求解近似稳态，然后基于该迭代图求梯度（自动微分/隐式微分均可），无需改变网络结构。实证表明在长序列任务上显著加速同时保持精度。
- 论文报告了 从 𝑂(𝑇)到 𝑂(𝐾) 的复杂度下降与典型 K=3 的配置，并讨论了与并行脉冲神经元的关系与收敛性。

## 2) 架构设计
```text
    //TODO
```

## 3) 如何启动训练

### 3.1 前置依赖

- Python 3.12+（建议 3.12）；所有依赖使用标准库，减少第三方依赖包。
- Docker / Docker Compose（用于 事件消息的默认消息队列中间件：NATS + JetStream + NATS-UI）
- pip 依赖
  ```shell
  # 后端通用依赖
  pip install -r requirements.txt

  ```
- 验证入口：

  - 客户端：`nats://127.0.0.1:4222`
  - JetStream 监控：`http://127.0.0.1:8222`
  - NATS UI：`http://127.0.0.1:31311`
  
  
# 4) UI 展示

前端 `ui-vue` 会通过 `/ws` 实时接收 `metrics`、`spikes` 与 `log` 事件：


# 8) 参考文献与资料
- https://arxiv.org/abs/2506.12087
- https://openreview.net/forum?id=HZKCXym5cS&referrer=%5Bthe+profile+of+Wanjin+Feng%5D%28%2Fprofile%3Fid%3D~Wanjin_Feng1%29
- https://icml.cc/virtual/2025/poster/45776?utm_source=chatgpt.com https://openreview.net/forum?id=HZKCXym5cS&noteId=WKlg9fTYXC