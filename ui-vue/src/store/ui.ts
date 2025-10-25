import { defineStore } from 'pinia';
import type {
  DatasetName,
  LayerLayout,
  MetricEntry,
  MetricPayload,
  MessageEntry,
  SpikeEntry,
  SpikePayload,
  TrainingConfig,
  TrainingStatus,
  LogPayload,
  LogEntry,
  DatasetDownloadEvent,
  TrainInitEvent,
  TrainIterEvent
} from '../types';

const MAX_METRICS = 500;
const MAX_SPIKES = 60;
const MAX_MESSAGES = 100;
const MAX_LOGS = 500;
const DEFAULT_TOAST_DURATION = 3000;
const PULSE_COOLDOWN_MS = 60;
const SPIKE_STREAM_TIMEOUT_MS = 1500;

const defaultConfig = (): TrainingConfig => ({
  dataset: 'MNIST',
  mode: 'fpt',
  network_size: 128,
  layers: 2,
  lr: 1e-3,
  K: 4,
  tol: 1e-5,
  T: 12,
  epochs: 20
});

const defaultBatchSnapshot = () => ({
  epoch: null as number | null,
  step: null as number | null,
  loss: null as number | null,
  acc: null as number | null,
  top5: null as number | null,
  throughput: null as number | null,
  stepMs: null as number | null,
  residual: null as number | null,
  emaLoss: null as number | null,
  emaAcc: null as number | null,
  lr: null as number | null,
  examples: null as number | null
});

type FinalSummary = {
  epoch: number | null;
  loss: number | null;
  acc: number | null;
  bestAcc: number | null;
  bestLoss: number | null;
  avgThroughput: number | null;
  epochSec: number | null;
};

const buildLayout = (cfg: TrainingConfig): LayerLayout[] => {
  const layers: LayerLayout[] = [];
  const perLayer = Math.max(1, Math.ceil(cfg.network_size / Math.max(1, cfg.layers)));
  const layerSpacing = 6;
  const gridSpacing = 1.4;

  for (let layerIndex = 0; layerIndex < cfg.layers; layerIndex += 1) {
    const count = perLayer;
    const positions = new Float32Array(count * 3);
    const columns = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / columns);
    const x = (layerIndex - (cfg.layers - 1) / 2) * layerSpacing;

    for (let i = 0; i < count; i += 1) {
      const row = Math.floor(i / columns);
      const col = i % columns;
      const y = ((rows - 1) / 2 - row) * gridSpacing;
      const z = (col - (columns - 1) / 2) * gridSpacing;
      const offset = i * 3;
      positions[offset] = x;
      positions[offset + 1] = y;
      positions[offset + 2] = z;
    }

    layers.push({
      layer: layerIndex,
      count,
      startIndex: layerIndex * perLayer,
      positions
    });
  }

  return layers;
};

export const useUiStore = defineStore('ui', {
  state: () => {
    const cfg = defaultConfig();
    return {
      cfg,
      status: 'Idle' as TrainingStatus,
      metrics: [] as MetricEntry[],
      lastMetric: null as MetricEntry | null,
      lastBatch: defaultBatchSnapshot(),
      bestAcc: 0,
      bestLoss: Number.POSITIVE_INFINITY,
      avgThroughput: null as number | null,
      epochDuration: null as number | null,
      totalEpochs: cfg.epochs ?? null,
      lastEpochIndex: null as number | null,
      examples: 0,
      lastLearningRate: cfg.lr,
      showDoneModal: false,
      finalSummary: null as FinalSummary | null,
      spikes: [] as SpikeEntry[],
      layersLayout: buildLayout(cfg),
      messages: [] as MessageEntry[],
      logs: [] as LogEntry[],
      toast: null as { message: string; type: 'info' | 'error'; at: number; duration?: number } | null,
      download: {
        active: false,
        name: '',
        progress: 0,
        startedAt: 0
      },
      pulseLayerCursor: 0,
      lastPulseAt: null as number | null,
      spikeStreamActive: false,
      lastSpikeEventAt: null as number | null,
      spikeSequence: 0
    };
  },
  getters: {
    totalNodes(state) {
      return state.layersLayout.reduce((acc, layer) => acc + layer.count, 0);
    },
    isDownloadActive(state) {
      return state.download.active;
    },
    downloadPercent(state) {
      return Math.round(state.download.progress * 100);
    },
    isControlLocked(state) {
      return state.download.active || state.status === 'Initializing';
    }
  },
  actions: {
    setCfg(partial: Partial<TrainingConfig>) {
      const current = this.cfg;
      const next: TrainingConfig = { ...current, ...partial };
      next.layers = Math.max(1, Math.round(Number.isFinite(next.layers) ? next.layers : current.layers));
      next.network_size = Math.max(1, Math.round(Number.isFinite(next.network_size) ? next.network_size : current.network_size));
      const lrValue = Number(next.lr);
      next.lr = Number.isFinite(lrValue) ? lrValue : current.lr;
      next.K = Math.max(1, Math.round(Number.isFinite(next.K) ? next.K : current.K));
      const tolValue = Number(next.tol);
      next.tol = Number.isFinite(tolValue) && tolValue > 0 ? tolValue : current.tol;
      if (typeof next.T === 'number' && Number.isFinite(next.T)) {
        next.T = Math.max(1, Math.round(next.T));
      } else {
        next.T = current.T;
      }
      if (typeof next.epochs === 'number' && Number.isFinite(next.epochs)) {
        next.epochs = Math.max(1, Math.round(next.epochs));
      } else {
        next.epochs = current.epochs;
      }
      this.cfg = next;
      this.layersLayout = buildLayout(next);
      this.totalEpochs = next.epochs ?? null;
      this.lastLearningRate = next.lr;
    },
    setDataset(name: DatasetName) {
      this.setCfg({ dataset: name });
    },
    setStatus(status: TrainingStatus) {
      this.status = status;
    },
    updateBatchSnapshot(payload: MetricPayload) {
      const snapshot = this.lastBatch ?? defaultBatchSnapshot();
      if (!this.lastBatch) {
        this.lastBatch = snapshot;
      }
      if (typeof payload.epoch === 'number') {
        snapshot.epoch = payload.epoch;
        this.lastEpochIndex = payload.epoch;
      }
      if (typeof payload.step === 'number') {
        snapshot.step = payload.step;
      }
      if (typeof payload.loss === 'number') {
        snapshot.loss = payload.loss;
      }
      if (typeof payload.acc === 'number') {
        snapshot.acc = payload.acc;
      }
      if (typeof payload.top5 === 'number') {
        snapshot.top5 = payload.top5;
      }
      if (typeof payload.throughput === 'number') {
        snapshot.throughput = payload.throughput;
      }
      if (typeof payload.step_ms === 'number') {
        snapshot.stepMs = payload.step_ms;
      }
      if (typeof payload.residual === 'number') {
        snapshot.residual = payload.residual;
      }
      if (typeof payload.ema_loss === 'number') {
        snapshot.emaLoss = payload.ema_loss;
      }
      if (typeof payload.ema_acc === 'number') {
        snapshot.emaAcc = payload.ema_acc;
      }
      if (typeof payload.lr === 'number' && Number.isFinite(payload.lr)) {
        snapshot.lr = payload.lr;
        this.lastLearningRate = payload.lr;
      }
      if (typeof payload.examples === 'number') {
        snapshot.examples = payload.examples;
        this.examples = payload.examples;
      }
    },
    prepareRun(init?: TrainInitEvent) {
      this.metrics = [];
      this.lastMetric = null;
      this.lastBatch = defaultBatchSnapshot();
      this.bestAcc = 0;
      this.bestLoss = Number.POSITIVE_INFINITY;
      this.avgThroughput = null;
      this.epochDuration = null;
      this.examples = 0;
      this.lastEpochIndex = null;
      this.showDoneModal = false;
      this.finalSummary = null;
      this.pulseLayerCursor = 0;
      this.clearSpikes();
      if (typeof init?.epochs === 'number') {
        this.totalEpochs = init.epochs;
      }
      if (typeof init?.lr === 'number' && Number.isFinite(init.lr)) {
        this.lastLearningRate = init.lr;
      } else {
        this.lastLearningRate = this.cfg.lr;
      }
      if (this.lastBatch) {
        this.lastBatch.epoch = 0;
        this.lastBatch.step = 0;
        this.lastBatch.loss = null;
        this.lastBatch.acc = null;
        this.lastBatch.top5 = null;
        this.lastBatch.throughput = null;
        this.lastBatch.stepMs = null;
        this.lastBatch.residual = null;
        this.lastBatch.emaLoss = null;
        this.lastBatch.emaAcc = null;
        this.lastBatch.lr = this.lastLearningRate;
        this.lastBatch.examples = 0;
      }
    },
    applyEpochMetric(payload: MetricPayload) {
      if (typeof payload.epoch === 'number') {
        this.lastEpochIndex = payload.epoch;
      }
      if (typeof payload.best_acc === 'number') {
        this.bestAcc = payload.best_acc;
      } else if (typeof payload.acc === 'number') {
        this.bestAcc = Math.max(this.bestAcc, payload.acc);
      }
      if (typeof payload.best_loss === 'number') {
        this.bestLoss = payload.best_loss;
      } else if (typeof payload.loss === 'number') {
        this.bestLoss = Number.isFinite(this.bestLoss) ? Math.min(this.bestLoss, payload.loss) : payload.loss;
      }
      if (typeof payload.avg_throughput === 'number') {
        this.avgThroughput = payload.avg_throughput;
      }
      if (typeof payload.epoch_sec === 'number') {
        this.epochDuration = payload.epoch_sec;
      }
      const isFinal = typeof this.totalEpochs === 'number' && typeof payload.epoch === 'number' && payload.epoch === this.totalEpochs;
      if (isFinal) {
        const fallbackAcc = typeof payload.acc === 'number' ? payload.acc : null;
        const fallbackLoss = typeof payload.loss === 'number' ? payload.loss : null;
        this.finalSummary = {
          epoch: payload.epoch ?? null,
          loss: fallbackLoss,
          acc: fallbackAcc,
          bestAcc: Number.isFinite(this.bestAcc) ? this.bestAcc : fallbackAcc,
          bestLoss: Number.isFinite(this.bestLoss) ? this.bestLoss : fallbackLoss,
          avgThroughput: typeof payload.avg_throughput === 'number' ? payload.avg_throughput : this.avgThroughput,
          epochSec: typeof payload.epoch_sec === 'number' ? payload.epoch_sec : this.epochDuration
        };
        this.showDoneModal = true;
      }
    },
    setTotalEpochs(value?: number | null) {
      this.totalEpochs = typeof value === 'number' ? value : null;
    },
    setLearningRate(value?: number) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        this.lastLearningRate = value;
      }
    },
    dismissDoneModal() {
      this.showDoneModal = false;
      this.finalSummary = null;
    },
    markTrainingDone() {
      if (!this.finalSummary) {
        const epoch = this.lastEpochIndex ?? this.lastBatch.epoch ?? null;
        const loss = typeof this.lastBatch.loss === 'number' ? this.lastBatch.loss : null;
        const acc = typeof this.lastBatch.acc === 'number' ? this.lastBatch.acc : null;
        this.finalSummary = {
          epoch,
          loss,
          acc,
          bestAcc: Number.isFinite(this.bestAcc) ? this.bestAcc : acc,
          bestLoss: Number.isFinite(this.bestLoss) ? this.bestLoss : loss,
          avgThroughput: this.avgThroughput,
          epochSec: this.epochDuration
        };
      }
      this.showDoneModal = true;
    },
    pushMetric(payload: MetricPayload) {
      const entry: MetricEntry = { ...payload, at: Date.now() };
      this.metrics.push(entry);
      if (this.metrics.length > MAX_METRICS) {
        this.metrics.splice(0, this.metrics.length - MAX_METRICS);
      }
      this.lastMetric = entry;
      this.updateBatchSnapshot(payload);
    },
    pushSpike(payload: SpikePayload, isReal = true) {
      const now = Date.now();
      const entry: SpikeEntry = { ...payload, at: now };
      this.spikes.push(entry);
      console.debug('[UI] pushSpike', {
        layer: entry.layer,
        neurons: entry.neurons,
        edges: entry.edges,
        totalSpikes: this.spikes.length
      });
      if (this.spikes.length > MAX_SPIKES) {
        this.spikes.splice(0, this.spikes.length - MAX_SPIKES);
      }
      this.lastPulseAt = now;
      if (isReal) {
        this.spikeStreamActive = true;
        this.lastSpikeEventAt = now;
      }
      this.spikeSequence += 1;
    },
    triggerPulseFromIter(payload: TrainIterEvent) {
      const now = Date.now();
      if (this.spikeStreamActive) {
        const lastEvent = typeof this.lastSpikeEventAt === 'number' ? this.lastSpikeEventAt : 0;
        const idle = now - lastEvent;
        if (idle > SPIKE_STREAM_TIMEOUT_MS) {
          console.debug('[UI] spike stream idle, reverting to synthetic pulses', { idleMs: idle });
          this.spikeStreamActive = false;
        } else {
          return;
        }
      }
      if (!this.layersLayout.length) {
        return;
      }
      if (typeof this.lastPulseAt === 'number' && now - this.lastPulseAt < PULSE_COOLDOWN_MS) {
        return;
      }
      const layouts = this.layersLayout;
      let layerIndex = 0;
      if (typeof payload.layer === 'number' && payload.layer >= 0 && payload.layer < layouts.length) {
        layerIndex = Math.floor(payload.layer);
      } else {
        layerIndex = this.pulseLayerCursor % layouts.length;
        this.pulseLayerCursor = (this.pulseLayerCursor + 1) % layouts.length;
      }
      const layout = layouts[layerIndex];
      if (!layout || layout.count <= 0) {
        return;
      }
      const baseCount = Math.max(1, Math.round(layout.count * 0.05));
      const residualMagnitude = typeof payload.residual === 'number' ? Math.min(1, Math.abs(payload.residual) * 0.8) : 0;
      const extra = Math.round(baseCount * residualMagnitude * 3);
      const total = Math.min(layout.count, baseCount + extra);

      const neurons: number[] = [];
      const used = new Set<number>();
      while (neurons.length < total) {
        const idx = Math.floor(Math.random() * layout.count);
        if (used.has(idx)) {
          continue;
        }
        used.add(idx);
        neurons.push(idx);
      }

      this.pushSpike({
        layer: layerIndex,
        t: typeof payload.time_unix === 'number' ? payload.time_unix : Math.floor(now / 1000),
        neurons
      }, false);
    },
    maybeTriggerPulseFromMetric(payload: MetricPayload) {
      const now = Date.now();
      if (this.spikeStreamActive) {
        const lastEvent = typeof this.lastSpikeEventAt === 'number' ? this.lastSpikeEventAt : 0;
        const idle = now - lastEvent;
        if (idle > SPIKE_STREAM_TIMEOUT_MS) {
          console.debug('[UI] spike stream idle during metrics, re-enabling fallback', { idleMs: idle });
          this.spikeStreamActive = false;
        } else {
          return;
        }
      }
      if (typeof this.lastPulseAt === 'number' && now - this.lastPulseAt < PULSE_COOLDOWN_MS) {
        return;
      }
      console.debug('[UI] fallback pulse from metrics', {
        epoch: payload.epoch,
        step: payload.step,
        residual: payload.residual
      });
      this.triggerPulseFromIter({
        epoch: payload.epoch,
        step: payload.step,
        residual: payload.residual,
        time_unix: payload.time_unix
      });
    },
    pushMessage(subject: string, payload?: unknown, type?: string) {
      const entry: MessageEntry = {
        subject,
        type,
        payload,
        at: Date.now()
      };
      this.messages.push(entry);
      if (this.messages.length > MAX_MESSAGES) {
        this.messages.splice(0, this.messages.length - MAX_MESSAGES);
      }
    },
    pushLog(payload: LogPayload) {
      const timestamp = typeof payload.ts === 'number' ? payload.ts * 1000 : Date.now();
      const entry: LogEntry = {
        ...payload,
        at: timestamp
      };
      this.logs.push(entry);
      if (this.logs.length > MAX_LOGS) {
        this.logs.splice(0, this.logs.length - MAX_LOGS);
      }
    },
    pushPlainLog(message: string, level: LogPayload['level'] = 'INFO', ts?: number) {
      const unixSeconds = typeof ts === 'number' ? ts : Math.floor(Date.now() / 1000);
      const last = this.logs[this.logs.length - 1];
      if (last && last.message === message && last.level === level) {
        const diff = typeof last.ts === 'number' ? Math.abs(last.ts - unixSeconds) : Number.POSITIVE_INFINITY;
        if (diff <= 1) {
          return;
        }
      }
      this.pushLog({
        ts: unixSeconds,
        level,
        message
      });
    },
    replaceMetrics(payloads: MetricPayload[]) {
      const entries = payloads.slice(-MAX_METRICS).map((payload) => {
        const historic = payload as MetricPayload & { at?: number };
        const timestamp = typeof historic.at === 'number' ? historic.at : Date.now();
        return { ...payload, at: timestamp };
      });
      this.metrics = entries;
      this.lastMetric = entries.length > 0 ? entries[entries.length - 1] : null;
      if (this.lastMetric) {
        this.updateBatchSnapshot(this.lastMetric);
      } else {
        this.lastBatch = defaultBatchSnapshot();
      }
    },
    replaceSpikes(payloads: SpikePayload[]) {
      const entries = payloads.slice(-MAX_SPIKES).map((payload) => {
        const historic = payload as SpikePayload & { at?: number };
        const timestamp = typeof historic.at === 'number' ? historic.at : Date.now();
        return { ...payload, at: timestamp };
      });
      this.spikes = entries;
    },
    replaceLogs(payloads: LogPayload[]) {
      const entries = payloads.slice(-MAX_LOGS).map((payload) => {
        const ts = typeof payload.ts === 'number' ? payload.ts * 1000 : Date.now();
        return {
          ...payload,
          at: ts
        } as LogEntry;
      });
      this.logs = entries;
    },
    clearSpikes() {
      this.spikes = [];
      this.lastPulseAt = null;
      this.spikeStreamActive = false;
      this.lastSpikeEventAt = null;
    },
    showToast(message: string, type: 'info' | 'error' = 'info', duration = DEFAULT_TOAST_DURATION) {
      this.toast = {
        message,
        type,
        at: Date.now(),
        duration
      };
    },
    clearToast(at?: number) {
      if (!this.toast) {
        return;
      }
      if (typeof at === 'number' && this.toast.at !== at) {
        return;
      }
      this.toast = null;
    },
    startDownload(name: string) {
      if (this.download.active && this.download.name === name) {
        return;
      }
      this.download = {
        active: true,
        name,
        progress: 0,
        startedAt: Date.now()
      };
      this.pushMessage('dataset', { name, state: 'start' }, 'dataset_download');
      this.pushPlainLog(`开始下载数据集 ${name}`, 'INFO');
    },
    updateDownloadProgress(progress: number) {
      if (!this.download.active) {
        return;
      }
      const clamped = Number.isFinite(progress) ? Math.min(Math.max(progress, 0), 1) : this.download.progress;
      this.download.progress = clamped;
    },
    finishDownload(success: boolean, message?: string, name?: string) {
      if (!this.download.active || (name && this.download.name !== name)) {
        return;
      }
      const datasetName = this.download.name;
      if (success && this.download.progress < 1) {
        this.download.progress = 1;
      }
      this.download = {
        active: false,
        name: '',
        progress: 0,
        startedAt: 0
      };
      if (success) {
        this.pushPlainLog(`数据集 ${datasetName} 下载完成`, 'INFO');
      } else {
        this.pushPlainLog(`数据集 ${datasetName} 下载失败${message ? `：${message}` : ''}`, 'ERROR');
      }
      this.pushMessage('dataset', { name: datasetName, success, message }, 'dataset_download');
    },
    applyDatasetEvent(event: DatasetDownloadEvent) {
      if (!event || !event.name) {
        return;
      }
      const state = event.state ?? '';
      if (state === 'start') {
        this.startDownload(event.name);
        if (typeof event.progress === 'number') {
          this.updateDownloadProgress(event.progress);
        }
        return;
      }
      if (state === 'progress') {
        this.startDownload(event.name);
        if (typeof event.progress === 'number') {
          this.updateDownloadProgress(event.progress);
        }
        return;
      }
      if (state === 'complete') {
        const progress = typeof event.progress === 'number' ? event.progress : 1;
        this.updateDownloadProgress(progress);
        this.finishDownload(true, undefined, event.name);
        return;
      }
      if (state === 'error') {
        this.finishDownload(false, event.message, event.name);
      }
    },
    reset() {
      const cfg = defaultConfig();
      this.cfg = cfg;
      this.status = 'Idle';
      this.metrics = [];
      this.lastMetric = null;
      this.lastBatch = defaultBatchSnapshot();
      this.bestAcc = 0;
      this.bestLoss = Number.POSITIVE_INFINITY;
      this.avgThroughput = null;
      this.epochDuration = null;
      this.examples = 0;
      this.totalEpochs = cfg.epochs ?? null;
      this.lastEpochIndex = null;
      this.lastLearningRate = cfg.lr;
      this.showDoneModal = false;
      this.finalSummary = null;
      this.spikes = [];
      this.layersLayout = buildLayout(cfg);
      this.messages = [];
      this.logs = [];
      this.toast = null;
      this.download = {
        active: false,
        name: '',
        progress: 0,
        startedAt: 0
      };
    }
  }
});
