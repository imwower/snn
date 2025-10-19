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

## 6) 事件流与数据约定（SSE `/events`）

| 事件名            | 说明                               | 核心字段（取自 `ui-vue/src/types.ts`） |
|-------------------|------------------------------------|----------------------------------------|
| `config`          | 初始配置快照（可选）               | `training`（同 `TrainInitEvent`）      |
| `train_init`      | 训练参数初始化                     | `dataset`, `epochs`, `fixed_point_K`, `timesteps`, `lr`, `hidden`, `layers` |
| `train_iter`      | 固定点迭代残差/进度                | `epoch`, `step`, `k`, `residual`, `layer` |
| `metrics_batch`   | 批次指标                           | `loss`, `acc`, `top5`, `throughput`, `step_ms`, `ema_loss`, `ema_acc`, `lr`, `examples` |
| `metrics_epoch`   | Epoch 汇总指标                     | `loss`, `acc`, `best_acc`, `best_loss`, `avg_throughput`, `epoch_sec` |
| `spike`           | 脉冲可视化事件                     | `layer`, `t`, `neurons`, `edges`, `power` |
| `log`             | 系统/训练日志行                    | `level`, `msg`, `time_unix`            |
| `dataset_download`| 数据集下载状态                     | `name`, `state`, `progress`, `message` |
| `train_status`    | 训练状态广播（Idle/Training 等）   | `status`                               |

- 时间戳字段统一使用毫秒 Unix 时间（`time_unix`）。
- UI 会维护最多 500 条指标与日志，可根据需要调整 `ui-vue/src/store/ui.ts` 中的常量。

## 7) REST 接口约定（由后端提供）

| 方法 | 路径                     | 说明                     | 请求体要点 |
|------|-------------------------|--------------------------|------------|
| GET  | `/api/config`           | 返回当前训练配置         | `{ "training": TrainInitEvent }`（可选） |
| GET  | `/api/datasets`         | 数据集列表               | 任意组合 `{datasets, available, installed}`，元素可为字符串或对象 |
| POST | `/api/datasets/download`| 触发数据集下载           | `{ name: DatasetName }` |
| POST | `/api/train/init`       | 写入训练配置但不启动     | `TrainingConfig` 或兼容字段 |
| POST | `/api/train/start`      | 开始训练                 | `{}`（如需参数可扩展） |
| POST | `/api/train/stop`       | 停止训练                 | `{}` |
| GET  | `/events`               | SSE 通道                 | 按第 6 节推送事件 |

所有接口返回 JSON；失败状态建议返回 `{ "message": "...", "detail": ... }` 便于 UI 在 Toast 中展示。

## 8) 自定义与二次开发建议

- **可视化参数**：`ui-vue/src/components/Sidebar.vue` 中的默认数据集与提示可按需扩展；`buildLayout` 可调整三维布局密度。
- **指标与日志上限**：通过 `ui-vue/src/store/ui.ts` 顶部常量（如 `MAX_METRICS`、`MAX_LOGS`）调节前端缓存量。
- **SSE 重连策略**：目前使用浏览器原生 `EventSource`，如需断线重连或认证，可在 `ui-vue/src/ws.ts` 中补充逻辑。
- **消息总线替换**：若不使用 NATS，可保持 SSE/REST 接口不变，将训练事件直接写入自定义队列或内存总线。

## 9) 参考文献与资料

- [Fixed-Point Parallel Training for Spiking Neural Networks（arXiv 2506.12087）](https://arxiv.org/abs/2506.12087)
- [Fixed-Point Parallel Training for Spiking Neural Networks · OpenReview](https://openreview.net/forum?id=HZKCXym5cS)
- [Fixed-Point Parallel Training for Spiking Neural Networks · ICML 2025 Poster](https://icml.cc/virtual/2025/poster/45776)
