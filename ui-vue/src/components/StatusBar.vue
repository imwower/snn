<template>
  <div class="status-bar-wrapper" :class="{ 'logs-open': showLogs }">
    <transition name="logs-expand">
      <div v-if="showLogs" class="logs-container">
        <LogsPanel />
      </div>
    </transition>
    <footer class="status-core">
      <span class="status-badge" :class="statusClass">{{ status }}</span>
      <div class="metrics">
        <div class="sparklines">
          <div class="sparkline-block">
            <svg class="sparkline" viewBox="0 0 100 30" preserveAspectRatio="none">
              <polyline
                v-if="lossSparkPoints"
                :points="lossSparkPoints"
                fill="none"
                stroke="var(--accent)"
                stroke-width="1.5"
                stroke-linejoin="round"
                stroke-linecap="round"
              />
              <line x1="0" y1="28" x2="100" y2="28" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
            </svg>
            <span class="spark-label">loss</span>
          </div>
          <div class="sparkline-block">
            <svg class="sparkline temperature" viewBox="0 0 100 30" preserveAspectRatio="none">
              <polyline
                v-if="temperatureSparkPoints"
                :points="temperatureSparkPoints"
                fill="none"
                stroke="#f6c343"
                stroke-width="1.5"
                stroke-linejoin="round"
                stroke-linecap="round"
              />
              <line x1="0" y1="28" x2="100" y2="28" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
            </svg>
            <span class="spark-label">logit_scale</span>
          </div>
        </div>
        <div class="metric-badges">
          <span class="badge">epoch: {{ metric.epoch }}</span>
          <span class="badge">step: {{ metric.step }}</span>
          <span class="badge">loss: {{ metric.loss }}</span>
          <span class="badge">acc: {{ metric.acc }}</span>
          <span class="badge">top5: {{ metric.top5 }}</span>
          <span class="badge">tps: {{ metric.throughput }}</span>
          <span class="badge">step_ms: {{ metric.stepMs }}</span>
          <span class="badge">residual: {{ metric.residual }}</span>
          <span class="badge">lr: {{ metric.lr }}</span>
          <span class="badge">examples: {{ metric.examples }}</span>
          <span class="badge">logit_scale: {{ metric.logitScale }}</span>
          <span class="badge">低置信 streak: {{ metric.lowConfStreak }}</span>
          <span class="badge">best_acc: {{ summary.bestAcc }}</span>
          <span class="badge">best_loss: {{ summary.bestLoss }}</span>
          <span class="badge">epoch_sec: {{ summary.epochSec }}</span>
          <span class="badge">avg_tps: {{ summary.avgThroughput }}</span>
        </div>
      </div>
      <div class="message-area">
        <div class="message-pill">
          <strong>{{ message.subject }}</strong>
          <span>{{ message.summary }}</span>
        </div>
        <button class="logs-toggle" type="button" @click="handleToggleLogs">
          {{ logsButtonLabel }}
        </button>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useUiStore } from '../store/ui';
import type { MetricEntry } from '../types';
import LogsPanel from './LogsPanel.vue';

const props = defineProps<{
  showLogs: boolean;
}>();

const emit = defineEmits<{
  (event: 'toggle-logs'): void;
}>();

const store = useUiStore();

const status = computed(() => store.status);
const statusClass = computed(() => `status-${status.value.toLowerCase()}`);

const formatNumber = (value: number | null | undefined, digits: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '--';

const formatResidual = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(6) : '--';

const formatInt = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? String(value) : '--';

const formatLR = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toString() : '--';

const formatExamples = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toString() : '--';

const metric = computed(() => {
  const batch = store.lastBatch;
  if (!batch) {
    return {
      epoch: '--',
      step: '--',
      loss: '--',
      acc: '--',
      top5: '--',
      throughput: '--',
      stepMs: '--',
      residual: '--',
      lr: '--',
      examples: '--',
      logitScale: '--',
      lowConfStreak: '--'
    };
  }
  const residualSource = batch.residual ?? store.lastMetric?.residual ?? null;
  return {
    epoch: formatInt(batch.epoch),
    step: formatInt(batch.step),
    loss: formatNumber(batch.loss, 4),
    acc: formatNumber(batch.acc, 4),
    top5: formatNumber(batch.top5, 4),
    throughput: formatNumber(batch.throughput, 1),
    stepMs: formatNumber(batch.stepMs, 0),
    residual: formatResidual(residualSource),
    lr: formatLR(batch.lr ?? store.lastLearningRate ?? null),
    examples: formatExamples(batch.examples ?? store.examples),
    logitScale: formatNumber(batch.logitScale, 2),
    lowConfStreak: formatInt(batch.lowConfStreak)
  };
});

const summary = computed(() => ({
  bestAcc: formatNumber(store.bestAcc, 4),
  bestLoss: formatNumber(store.bestLoss, 4),
  avgThroughput: formatNumber(store.avgThroughput, 1),
  epochSec: formatNumber(store.epochDuration, 1)
}));

const computeSparkPoints = (selector: (entry: MetricEntry) => number | null | undefined) => {
  const entries = store.metrics.slice(-60);
  if (entries.length < 2) {
    return '';
  }
  const series = entries.map((entry) => {
    const value = selector(entry);
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  });
  const valid = series.filter((value): value is number => value !== null);
  if (valid.length < 2) {
    return '';
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  return series
    .map((value, index) => {
      const x = (index / (series.length - 1)) * 100;
      const normalized = value === null ? 0.5 : range === 0 ? 0.5 : (value - min) / range;
      const y = 28 - normalized * 24;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
};

const lossSparkPoints = computed(() => computeSparkPoints((entry) => entry.loss));
const temperatureSparkPoints = computed(() => computeSparkPoints((entry) => entry.logit_scale));

const message = computed(() => {
  if (store.isDownloadActive) {
    const time = new Date(store.download.startedAt || Date.now()).toLocaleTimeString('zh-CN', { hour12: false });
    return {
      subject: '下载中',
      summary: `${store.download.name} · ${store.downloadPercent}% · ${time}`
    };
  }
  const last = store.logs[store.logs.length - 1];
  if (!last) {
    return { subject: '日志', summary: '暂无日志' };
  }
  const time = new Date(last.at).toLocaleTimeString('zh-CN', { hour12: false });
  const level = last.level ?? 'INFO';
  return {
    subject: `[${level}]`,
    summary: `${last.message} · ${time}`
  };
});

const logsButtonLabel = computed(() => (props.showLogs ? '隐藏日志' : '显示日志'));
const handleToggleLogs = () => emit('toggle-logs');
</script>

<style scoped>
.metrics {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.sparklines {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.sparkline-block {
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.sparkline {
  width: 120px;
  height: 30px;
}

.spark-label {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-top: 0.2rem;
}

.metric-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
}
</style>
