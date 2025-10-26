# SNN‑3Comp‑FPT（事件驱动）

> 三隔室 SNN（soma/顶树突/基底树突）× 固定点并行训练（Fixed‑point Parallel Training, FPT）× 事件驱动（默认使用 NATS JetStream 消息总线）的参考实现蓝图与脚手架。

## 0) TL;DR

> - 三隔室神经元 SNN（soma/顶树突/基底树突）
>- FPT：把传统按时间步 𝑇 展开的 SNN 训练从 𝑂(𝑇) 降到 𝑂(𝐾)（常取小常数，如 K≈3），做法是把 LIF 等时序神经元写成固定点方程，用少量并行迭代逼近整段时间序列的解；保持原模型结构并给出收敛分析与实证。
>- 事件驱动：训练拆成“事件”，默认使用 NATS JetStream 做持久化流、拉式消费、至少一次投递、可回放、发布去重（`Nats-Msg-Id` + 去重窗口）；可通过配置文件切换为其他消息队列。

## 1) 核心理念
- **用固定点并行替代时间展开** 将 LIF/SNN 的时序动力学改写为 𝑢 = Φ(𝑢) 的固定点形式；在每个 batch 上做 K 次并行迭代求解近似稳态，然后基于该迭代图求梯度（自动微分/隐式微分均可），无需改变网络结构。实证表明在长序列任务上显著加速同时保持精度。论文报告了 从 𝑂(𝑇)到 𝑂(𝐾) 的复杂度下降与典型 K=3 的配置，并讨论了与并行脉冲神经元的关系与收敛性。
- **事件驱动训练编排**：训练服务负责与数据集/算力交互，产生的指标、脉冲、日志等通过消息总线汇总，UI 仅消费事件。
- **三隔室可视化**：`ui-vue` 使用 Three.js 将网络节点、突触映射为三维布局，并根据脉冲事件实时点亮神经元与边。
- **运行态监控**：状态栏包含 Sparkline、批次/epoch 指标、吞吐量、残差、学习率等信息；Toast 与日志面板协助排错。
- **后端交互蓝图**：前端约定了 REST API（初始化/启动/停止训练、数据集下载）与 SSE 事件（metrics/log/spike 等），便于快速对接自定义后端。
- **NATS JetStream 默认集成**：`docker-compose.yml` 与 `nats.conf` 预置消息总线，提供持久化与可回放能力。

## 2) 代码结构

- `ui-vue/`：基于 Vue 3 + Vite + Pinia + Three.js 的实时训练监控界面。
- `docker-compose.yml`：启动 NATS 服务与 Web UI（NATS-NUI）监控面板。
- `nats.conf`：JetStream 存储、域名与端口设置（数据目录默认写入 `./.data/nats`）。

## 3) 系统架构与数据流

```text
┌────────────────┐     REST /api/*            ┌────────────────────────┐
│    ui-vue      │ ────────────────────────▶ │   训练/编排服务（自实现）│
│ Vue + Three.js │                           │ - 数据集管理            │
│ SSE EventSource│ ◀───── SSE /events ───────│ - FPT 迭代与指标计算     │
└─────┬──────────┘                           └───────────┬────────────┘
      │                                                       │
      │ NATS JetStream publish / subscribe                     │
      ▼                                                       ▼
┌────────────────┐      持久化/回放/监控           ┌──────────────────┐
│   NATS Server  │ <────────────────────────────── │ NATS-NUI (可选)  │
└────────────────┘                                  └──────────────────┘
```

典型流程：训练服务消费数据集并执行 FPT → 通过 JetStream 推送指标/日志 → 后端聚合成 SSE 事件供前端消费 → 前端实时渲染三维网络与状态信息。

## 4) 环境准备

- Docker / Docker Compose（NATS + JetStream + NATS UI）
- Node.js 18+ 与 npm（前端界面）
- Python 3.12+ 或其他你选择的运行时（实现训练/编排服务；本仓库未附带后端实现，仅提供接口约定）

## 5) 快速开始

### 5.1 启动消息总线

```bash
docker compose up -d
```

- NATS 客户端：`nats://127.0.0.1:4222`
- JetStream 监控：`http://127.0.0.1:8222`
- NATS UI（NATS-NUI）：`http://127.0.0.1:31311`

停止服务：`docker compose down`

### 5.2 对接训练服务（需自行实现）

实现一个后端进程，负责：

1. 提供 REST API（见第 7 节）管理数据集、训练生命周期。
2. 消费 NATS/内部队列，生成训练指标、脉冲与日志，封装为 SSE 事件（见第 6 节）。
3. 可选：直接使用 JetStream 作为事件缓冲，或将 JetStream 作为训练服务与 UI 之间的中间层。

参考类型定义：`ui-vue/src/types.ts`。

默认训练配置会根据当前数据集样本数自动设置 `steps_per_epoch = ceil(N_{\text{train}} / batch_size)`，不再需要手动硬编码 128；同时在前向最后一步按 `logit_scale=1.25` 放大 logits。训练前端到端会以 NumPy 计算每个特征的 mean/std（存放在 `.data/datasets/<name>/mean.npy|std.npy`），并在训练阶段启用轻量增广（平移±2 像素、随机左右翻转、随机裁剪+Pad）以避免过拟合，可通过 `training_service.augment` 开关。

读出层默认使用 NumPy 实现的两层 MLP（`in_dim → head_hidden → classes`，激活函数为 `tanh`），并携带可学习温度 `logit_scale`。相关超参可在 `training_service` 段配置：`head_hidden` / `head_momentum` 控制 MLP 隐层与动量，`logit_scale_init/min/max`（默认 `1.60/0.50/3.00`）约束温度范围，`unfreeze_at_conf` 则决定读出头单独预热多久（当批次置信度超过该阈值后再解冻 FPT 主体权重）。

### 5.3 启动前端 UI

```bash
cd ui-vue
npm install
npm run dev
```

- 默认开发端口：`http://127.0.0.1:5173`
- 开发模式下 SSE 默认连接 `http://127.0.0.1:8000/events`（参见 `ui-vue/src/ws.ts`），生产环境改为相对路径 `/events`。
- 如需调整 API/SSE 地址，可在 `ui-vue/src/ws.ts` 与组件中修改 axios/fetch 基础路径。

### 5.4 常见排查

- UI 无法连接事件源：检查 NATS 是否启动、后端是否暴露 `/events` SSE；浏览器开发者工具应能看到 `metrics_batch` 等事件。
- 数据集下拉为空：确认后端实现了 `GET /api/datasets` 并返回符合约定的 JSON。
- 训练状态未更新：`POST /api/train/start` 应返回 2xx，同时 SSE 中应发送 `train_status` 或 `metrics_batch` 事件。
- 3D 网络只在第一次脉冲时点亮：确保后端持续推送 `spike` 事件；前端通过 `spikeSequence` 计数（`ui-vue/src/store/ui.ts`）对每次 `pushSpike` 自增，`Network3D` 监听该计数来渲染最新脉冲，即使 `spikes` 数组因裁剪保持恒长也能继续触发高亮。

## 6) 事件流与数据约定（SSE `/events`）

后端需为每个连接维护独立的推送队列。所有事件采用 `event: <name>` / `data: <json>\n\n` 形式发送，并使用毫秒级 Unix 时间戳（`time_unix` 字段）保持客户端时间轴一致。事件负载与 `ui-vue/src/types.ts` 中的接口保持一对一映射：

- `train_init` → `TrainInitEvent`：初始化训练参数，字段包括 `dataset`、`epochs`、`fixed_point_K`、`fixed_point_tol`、`timesteps`、`hidden`、`layers`、`lr`。
- `train_status` → `{ status: TrainingStatus }`：广播 `Idle` / `Initializing` / `Training` / `Stopped` / `Error`。
- `train_iter` → `TrainIterEvent`：固定点残差进度，携带 `epoch`、`step`、`k`、`max_k`、`layer`、`residual`、`solver` 等（便于 UI 展示 Anderson 混合步数和迭代上限）。
- `metrics_batch` / `metrics_epoch` → `MetricPayload`：分别代表批次指标与 epoch 汇总，字段覆盖 `loss/nll`、`conf`、`entropy`、`acc`、`top5`、`throughput`、`step_ms`、`ema_loss`、`ema_acc`、`lr`、`temperature`、`s_rate`、`logit_scale/logit_mean/logit_std`、`residual`、`k`/`k_bin`、`examples`、`best_acc`、`best_loss`、`avg_throughput`、`epoch_sec`。
- `spike` → `SpikePayload`：实时脉冲，可选 `edges` 与功率 `power`。
- `log` → `UISysLogEvent` 或 `LogPayload`：用于 UI 控制台和 Toast，需提供 `level`、`msg` / `message`。
- `dataset_download` → `DatasetDownloadEvent`：三态状态机：`start`、`progress`、`complete`（如失败可回传 `error` 并附带 `message`）。
- `config`（可选）→ `{ training: TrainInitEvent }`：在连接建立时推送最新配置快照。

UI 默认缓存 500 条指标与日志（参见 `ui-vue/src/store/ui.ts`），若后端事件过于频繁建议在服务侧做节流。

## 7) REST 接口约定（由后端提供）

与 `Sidebar.vue`、`ws.ts` 所使用的 axios/fetch 调用保持一致，所有响应以 JSON 返回，错误情况建议包含 `{ "message": "...", "detail": ... }` 便于前端展示。

| 方法 | 路径 | 请求体 | 成功响应（示例） | 说明 |
|------|------|--------|-----------------|------|
| GET | `/api/config` | — | `{ "training": TrainInitEvent, "status": "Idle" }` | 返回最近一次初始化参数与当前训练状态 |
| GET | `/api/datasets` | — | `{ "datasets": [{ "name": "MNIST", "installed": true, "progress": 100 }] }` | 供侧边栏渲染数据集下拉及安装状态 |
| POST | `/api/datasets/download` | `{ "name": DatasetName }` | `{ "ok": true, "message": "download scheduled" }` | 触发数据集下载，并通过 `dataset_download` 事件汇报进度 |
| POST | `/api/train/init` | `TrainingConfig` | `{ "ok": true }` | 写入训练配置，同时发布 `train_init` 事件 |
| POST | `/api/train/start` | `{}` | `{ "ok": true }`（202 Accepted） | 启动训练流水线，后续事件见第 6 节 |
| POST | `/api/train/stop` | `{}` | `{ "ok": true }` | 请求停止当前训练，完成后应广播 `train_status: Stopped` |
| GET | `/events` | — | `text/event-stream` | Server-Sent Events 长连接 |

### 5.5 启动后端并触发训练

```bash
pip install -r requirements.txt   # 建议在虚拟环境中
uvicorn snn.server:create_app --host 0.0.0.0 --port 8000
```

后端启动后，在浏览器打开 `http://127.0.0.1:5173`（Vite 默认地址），依次点击“下载到本地”“初始化”“启动”即可开始训练。队列默认指向 `config.yaml` 中的 NATS 配置，可通过修改该文件或向 `create_app()` 传入参数进行覆盖。

## 8) 自定义与二次开发建议

- **可视化参数**：`ui-vue/src/components/Sidebar.vue` 中的默认数据集与提示可按需扩展；`buildLayout` 可调整三维布局密度。
- **指标与日志上限**：通过 `ui-vue/src/store/ui.ts` 顶部常量（如 `MAX_METRICS`、`MAX_LOGS`）调节前端缓存量。
- **SSE 重连策略**：目前使用浏览器原生 `EventSource`，如需断线重连或认证，可在 `ui-vue/src/ws.ts` 中补充逻辑。
- **消息总线替换**：若不使用 NATS，可保持 SSE/REST 接口不变，将训练事件直接写入自定义队列或内存总线。

## 9) 参考文献与资料

- [Fixed-Point Parallel Training for Spiking Neural Networks（arXiv 2506.12087）](https://arxiv.org/abs/2506.12087)
- [Fixed-Point Parallel Training for Spiking Neural Networks · OpenReview](https://openreview.net/forum?id=HZKCXym5cS)
- [Fixed-Point Parallel Training for Spiking Neural Networks · ICML 2025 Poster](https://icml.cc/virtual/2025/poster/45776)
