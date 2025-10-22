<template>
  <div class="logs-panel" ref="container">
    <div
      v-for="(entry, index) in logs"
      :key="entry.at + '-' + index"
      class="log-line"
      :data-level="entry.level"
    >
      <span class="log-time">[{{ formatTime(entry.at) }}]</span>
      <span class="log-level">{{ entry.level }}</span>
      <span class="log-message">{{ entry.message }}</span>
      <span v-if="entry.metric" class="log-metric">{{ formatMetric(entry.metric) }}</span>
    </div>
    <div v-if="logs.length === 0" class="log-empty">暂无日志</div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useUiStore } from '../store/ui';

const LOG_LIMIT = 20;
const store = useUiStore();
const logs = computed(() => store.logs.slice(-LOG_LIMIT));
const container = ref<HTMLDivElement | null>(null);

const pad = (value: number) => value.toString().padStart(2, '0');

const formatTime = (timestamp: number) => {
  const date = new Date(timestamp);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
};

const formatMetric = (metric: Record<string, unknown>) => JSON.stringify(metric);

watch(
  () => logs.value.length,
  async () => {
    await nextTick();
    const el = container.value;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }
);
</script>

<style scoped>
.logs-panel {
  max-height: 220px;
  overflow-y: auto;
  font-family: 'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 12px;
  padding: 8px 10px;
  background-color: rgba(18, 18, 18, 0.85);
  color: #f5f5f5;
  border-radius: 6px;
}

.log-line {
  display: flex;
  align-items: center;
  gap: 6px;
  line-height: 1.5;
  word-break: break-word;
}

.log-line + .log-line {
  margin-top: 4px;
}

.log-time {
  color: #9ca3af;
}

.log-level {
  padding: 2px 6px;
  border-radius: 4px;
  background-color: rgba(255, 255, 255, 0.08);
  text-transform: uppercase;
  font-weight: 600;
}

.log-message {
  flex: 1;
}

.log-metric {
  color: #6ee7b7;
  font-family: inherit;
}

.log-line[data-level='ERROR'] .log-level {
  background-color: rgba(248, 113, 113, 0.3);
  color: #fca5a5;
}

.log-line[data-level='WARNING'] .log-level {
  background-color: rgba(251, 191, 36, 0.3);
  color: #facc15;
}

.log-line[data-level='DEBUG'] .log-level {
  background-color: rgba(96, 165, 250, 0.3);
  color: #93c5fd;
}

.log-empty {
  color: #9ca3af;
  text-align: center;
  padding: 12px 0;
}
</style>
