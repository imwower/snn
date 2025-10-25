export type DatasetName = string;

export interface DatasetSummary {
  slug?: string;
  name?: string;
  status?: string;
  message?: string | null;
  installed?: boolean;
  path?: string;
  progress?: number;
  downloaded?: number;
  total?: number;
  [key: string]: unknown;
}

export interface DatasetListPayload {
  datasets?: Array<DatasetSummary | DatasetName>;
  available?: DatasetName[];
  installed?: DatasetName[];
}

export type TrainingStatus = 'Idle' | 'Initializing' | 'Training' | 'Stopped' | 'Error';

export interface TrainingConfig {
  dataset: DatasetName;
  mode: 'tstep' | 'fpt';
  network_size: number;
  layers: number;
  lr: number;
  K: number;
  tol: number;
  T?: number;
  epochs: number;
  solver?: 'plain' | 'anderson';
  anderson_m?: number;
  anderson_beta?: number;
  K_schedule?: string | null;
}

export interface MetricPayload {
  epoch: number;
  step: number;
  loss: number;
  nll?: number;
  acc?: number;
  top5?: number;
  conf?: number;
  entropy?: number;
  ema_loss?: number;
  ema_acc?: number;
  throughput?: number;
  step_ms?: number;
  lr?: number;
  logit_scale?: number;
  logit_mean?: number;
  logit_std?: number;
  s_rate?: number;
  residual?: number;
  k?: number;
  k_bin?: number;
  temperature?: number;
  examples?: number;
  best_acc?: number;
  best_loss?: number;
  avg_throughput?: number;
  epoch_sec?: number;
  time_unix?: number;
}

export interface MetricEntry extends MetricPayload {
  at: number;
}

export interface SpikePayload {
  layer: number;
  t: number;
  neurons: number[];
  edges?: [number, number][];
  power?: number;
  apical_trace?: number[];
  basal_trace?: number[];
}

export interface SpikeEntry extends SpikePayload {
  at: number;
}

export interface MessageEntry {
  subject: string;
  type?: string;
  at: number;
  payload?: unknown;
}

export interface LayerLayout {
  layer: number;
  count: number;
  startIndex: number;
  positions: Float32Array;
}

export interface LogPayload {
  ts: number;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  metric?: Record<string, unknown>;
}

export interface LogEntry extends LogPayload {
  at: number;
}

export interface TrainInitEvent {
  dataset?: string;
  epochs?: number;
  batch_size?: number;
  timesteps?: number;
  fixed_point_K?: number;
  fixed_point_tol?: number;
  hidden?: number;
  layers?: number;
  lr?: number;
  solver?: string;
  anderson_m?: number;
  anderson_beta?: number;
  K_schedule?: string | null;
  temperature?: number;
  logit_scale?: number;
  logit_scale_learnable?: boolean;
  steps_per_epoch?: number;
  grad_clip?: number;
  weight_decay?: number;
  rate_target?: number;
  time_unix?: number;
}

export interface TrainIterEvent {
  epoch?: number;
  step?: number;
  k?: number;
  layer?: number;
  residual?: number;
  time_unix?: number;
  max_k?: number;
  solver?: string;
  lr?: number;
  k_bin?: number;
}

export interface UISysLogEvent {
  level?: string;
  msg?: string;
  time_unix?: number;
}

export interface DatasetDownloadEvent {
  name?: string;
  state?: 'start' | 'progress' | 'complete' | 'error';
  progress?: number;
  message?: string;
  installed?: boolean;
  time_unix?: number;
}
