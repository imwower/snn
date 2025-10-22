import type { Pinia } from 'pinia';
import { useUiStore } from './store/ui';
import type {
  MetricPayload,
  TrainInitEvent,
  TrainIterEvent,
  UISysLogEvent,
  LogPayload,
  DatasetDownloadEvent,
  TrainingStatus,
  SpikePayload
} from './types';

let source: EventSource | null = null;
let started = false;
let storeInstance: ReturnType<typeof useUiStore> | null = null;

const toLogLevel = (level?: string): LogPayload['level'] => {
  const normalized = (level || 'INFO').toUpperCase();
  if (normalized === 'DEBUG' || normalized === 'INFO' || normalized === 'WARNING' || normalized === 'ERROR') {
    return normalized;
  }
  return 'INFO';
};

const parseJSON = <T>(raw: string): T | null => {
  try {
    return JSON.parse(raw) as T;
  } catch (err) {
    console.warn('无法解析 SSE 负载', err);
    return null;
  }
};

const pushTextLog = (message: string, ts?: number, level: LogPayload['level'] = 'INFO') => {
  storeInstance?.pushPlainLog(message, level, ts);
  storeInstance?.pushMessage('log-text', { message, ts, level }, 'log');
};

const formatFixed = (value: number | null | undefined, digits: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';

const handleMetricsBatch = (payload: MetricPayload) => {
  const store = storeInstance;
  store?.pushMetric(payload);
  if (store && store.status === 'Idle') {
    store.setStatus('Training');
  }
  const loss = formatFixed(payload.loss, 4);
  const acc = formatFixed(payload.acc, 4);
  const top5 = formatFixed(payload.top5, 4);
  const throughput = formatFixed(payload.throughput, 1);
  const stepMs = formatFixed(payload.step_ms, 0);
  const emaLoss = formatFixed(payload.ema_loss, 4);
  const emaAcc = formatFixed(payload.ema_acc, 4);
  const lrValue = typeof payload.lr === 'number' && Number.isFinite(payload.lr)
    ? payload.lr
    : typeof store?.lastLearningRate === 'number'
      ? store.lastLearningRate
      : null;
  const lrText = lrValue !== null ? String(lrValue) : 'n/a';
  const examplesValue = typeof payload.examples === 'number'
    ? payload.examples
    : typeof store?.examples === 'number'
      ? store.examples
      : null;
  const examplesText = examplesValue !== null ? String(examplesValue) : 'n/a';
  pushTextLog(
    `[BATCH] ep=${payload.epoch ?? '-'} st=${payload.step ?? '-'} loss=${loss} acc=${acc} top5=${top5} tps=${throughput} step_ms=${stepMs} ema_loss=${emaLoss} ema_acc=${emaAcc} lr=${lrText} examples=${examplesText}`,
    payload.time_unix
  );
  store?.pushMessage('metrics_batch', payload, 'metrics_batch');
  store?.maybeTriggerPulseFromMetric(payload);
};

const handleMetricsEpoch = (payload: MetricPayload) => {
  const store = storeInstance;
  store?.applyEpochMetric(payload);
  const loss = formatFixed(payload.loss, 4);
  const acc = formatFixed(payload.acc, 4);
  const bestAccValue = typeof payload.best_acc === 'number' ? payload.best_acc : store?.bestAcc;
  const bestLossValue = typeof payload.best_loss === 'number' ? payload.best_loss : store?.bestLoss;
  const avgTpsValue = typeof payload.avg_throughput === 'number' ? payload.avg_throughput : store?.avgThroughput;
  const epochSecValue = typeof payload.epoch_sec === 'number' ? payload.epoch_sec : store?.epochDuration;
  const bestAcc = formatFixed(bestAccValue, 4);
  const bestLoss = formatFixed(bestLossValue, 4);
  const avgTps = formatFixed(avgTpsValue, 1);
  const epochSec = formatFixed(epochSecValue, 1);
  pushTextLog(
    `[EPOCH] epoch=${payload.epoch ?? '-'} loss=${loss} acc=${acc} best_acc=${bestAcc} best_loss=${bestLoss} avg_tps=${avgTps} time=${epochSec}s`,
    payload.time_unix
  );
  store?.pushMessage('metrics_epoch', payload, 'metrics_epoch');
};

const handleSysLog = (payload: UISysLogEvent) => {
  const level = toLogLevel(payload.level);
  const message = payload.msg ?? '';
  console.log('[UI] 日志', message);
  storeInstance?.pushPlainLog(message, level, payload.time_unix);
  storeInstance?.pushMessage('log', payload, 'log');
  if (message.includes('训练完成')) {
    storeInstance?.markTrainingDone();
  }
};

const handleTrainInit = (payload: TrainInitEvent) => {
  console.log('[UI] train_init', payload);
  const text = `[INIT] dataset=${payload.dataset ?? '-'} epochs=${payload.epochs ?? '-'} batch=${payload.batch_size ?? '-'} layers=${payload.layers ?? '-'} T=${payload.timesteps ?? '-'} K=${payload.fixed_point_K ?? '-'} lr=${payload.lr ?? '-'}`;
  storeInstance?.setStatus('Idle');
  if (storeInstance) {
    const current = storeInstance.cfg;
    storeInstance.setCfg({
      dataset: payload.dataset ?? current.dataset,
      lr: typeof payload.lr === 'number' ? payload.lr : current.lr,
      K: typeof payload.fixed_point_K === 'number' ? payload.fixed_point_K : current.K,
      T: typeof payload.timesteps === 'number' ? payload.timesteps : current.T,
      network_size: typeof payload.hidden === 'number' ? payload.hidden : current.network_size,
      layers: typeof payload.layers === 'number' ? payload.layers : current.layers,
      tol: typeof payload.fixed_point_tol === 'number' ? payload.fixed_point_tol : current.tol,
      epochs: typeof payload.epochs === 'number' ? payload.epochs : current.epochs
    });
    storeInstance.prepareRun(payload);
  }
  storeInstance?.pushPlainLog(text, 'INFO', payload.time_unix);
  storeInstance?.pushMessage('train_init', payload, 'train_init');
};

const handleTrainIter = (payload: TrainIterEvent) => {
  if (typeof payload.residual === 'number') {
    console.log('[UI] train_iter residual', payload.residual);
  }
  console.debug('[SSE] train_iter payload', payload);
  const residualText =
    typeof payload.residual === 'number' ? payload.residual.toFixed(6) : String(payload.residual ?? 'n/a');
  pushTextLog(
    `[FPT] epoch=${payload.epoch ?? '-'} step=${payload.step ?? '-'} residual=${residualText}`,
    payload.time_unix
  );
  if (storeInstance) {
    storeInstance.triggerPulseFromIter(payload);
  }
  storeInstance?.pushMessage('train_iter', payload, 'train_iter');
};

const handleDatasetDownload = (payload: DatasetDownloadEvent) => {
  storeInstance?.applyDatasetEvent(payload);
};

const handleSpike = (payload: SpikePayload) => {
  storeInstance?.pushSpike(payload);
  storeInstance?.pushMessage('spike', payload, 'spike');
};

const setupListeners = (evSource: EventSource) => {
  evSource.addEventListener('metrics_batch', (event: MessageEvent<string>) => {
    const payload = parseJSON<MetricPayload>(event.data);
    if (payload) {
      handleMetricsBatch(payload);
    }
  });

  evSource.addEventListener('metrics_epoch', (event: MessageEvent<string>) => {
    const payload = parseJSON<MetricPayload>(event.data);
    if (payload) {
      handleMetricsEpoch(payload);
    }
  });

  evSource.addEventListener('log', (event: MessageEvent<string>) => {
    const payload = parseJSON<UISysLogEvent>(event.data);
    if (payload) {
      handleSysLog(payload);
    }
  });

  evSource.addEventListener('train_init', (event: MessageEvent<string>) => {
    const payload = parseJSON<TrainInitEvent>(event.data);
    if (payload) {
      handleTrainInit(payload);
    }
  });

  evSource.addEventListener('train_iter', (event: MessageEvent<string>) => {
    const payload = parseJSON<TrainIterEvent>(event.data);
    if (payload) {
      handleTrainIter(payload);
    }
  });
  evSource.addEventListener('spike', (event: MessageEvent<string>) => {
    const payload = parseJSON<SpikePayload>(event.data);
    if (payload) {
      handleSpike(payload);
    }
  });
  evSource.addEventListener('train_status', (event: MessageEvent<string>) => {
    const payload = parseJSON<{ status?: string }>(event.data);
    if (payload?.status && storeInstance) {
      const nextStatus = payload.status as TrainingStatus;
      const prevStatus = storeInstance.status;
      storeInstance.setStatus(nextStatus);
      if (prevStatus === 'Training' && nextStatus === 'Idle') {
        storeInstance.markTrainingDone();
      }
    }
  });
  evSource.addEventListener('dataset_download', (event: MessageEvent<string>) => {
    const payload = parseJSON<DatasetDownloadEvent>(event.data);
    if (payload) {
      handleDatasetDownload(payload);
    }
  });

  evSource.addEventListener('open', () => {
    storeInstance?.pushMessage('events', { state: 'open' }, 'sse');
  });

  evSource.onerror = (err) => {
    console.warn('SSE 错误', err);
    storeInstance?.pushMessage('events', { state: 'error', error: err }, 'sse');
  };
};

const fetchConfig = async () => {
  const store = storeInstance;
  if (!store) {
    return;
  }
  try {
    const response = await fetch('/api/config', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`status=${response.status}`);
    }
    const cfg = await response.json();
    const training = cfg?.training as TrainInitEvent | undefined;
    if (training) {
      const text = `[CONFIG] dataset=${training.dataset ?? '-'} epochs=${training.epochs ?? '-'} batch=${training.batch_size ?? '-'} T=${training.timesteps ?? '-'} K=${training.fixed_point_K ?? '-'} lr=${training.lr ?? '-'}`;
      pushTextLog(text);
      const current = store.cfg;
      store.setCfg({
        dataset: training.dataset ?? current.dataset,
        lr: typeof training.lr === 'number' ? training.lr : current.lr,
        K: typeof training.fixed_point_K === 'number' ? training.fixed_point_K : current.K,
        T: typeof training.timesteps === 'number' ? training.timesteps : current.T,
        network_size: typeof training.hidden === 'number' ? training.hidden : current.network_size,
        tol: typeof training.fixed_point_tol === 'number' ? training.fixed_point_tol : current.tol,
        epochs: typeof training.epochs === 'number' ? training.epochs : current.epochs
      });
    }
    store.pushMessage('config', cfg, 'config');
  } catch (err) {
    console.warn('获取配置失败', err);
    pushTextLog('[WARN] 获取配置失败', undefined, 'WARNING');
  }
};

const fetchRecentLogs = async () => {
  const store = storeInstance;
  if (!store) {
    return;
  }
  try {
    const response = await fetch('/api/logs/recent?limit=200', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`status=${response.status}`);
    }
    const payload = await response.json();
    const logs = Array.isArray(payload?.logs) ? (payload.logs as LogPayload[]) : [];
    store.replaceLogs(logs);
  } catch (err) {
    console.warn('获取历史日志失败', err);
  }
};

const connect = () => {
  if (!storeInstance) {
    return;
  }
  const url =
    typeof window !== 'undefined' && import.meta.env.DEV
      ? 'http://127.0.0.1:8000/events'
      : '/events';
  source = new EventSource(url, { withCredentials: false });
  setupListeners(source);
};

export const startSocket = (pinia: Pinia) => {
  if (started) {
    return;
  }
  storeInstance = useUiStore(pinia);
  started = true;
  void fetchRecentLogs();
  void fetchConfig();
  connect();
};

export const getSocket = () => source;
