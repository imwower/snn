<template>
  <div class="logs-panel" ref="container">
    <div v-for="(entry, index) in logs" :key="entry.at + '-' + index" class="log-line">
      <span class="log-time">[{{ formatTime(entry.at) }}]</span>
      <span class="log-message">{{ entry.message }}</span>
    </div>
    <div v-if="logs.length === 0" class="log-empty">暂无日志</div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useUiStore } from '../store/ui';

const LOG_LIMIT = 6;
const store = useUiStore();
const logs = computed(() => store.logs.slice(-LOG_LIMIT));
const container = ref<HTMLDivElement | null>(null);

const pad = (value: number) => value.toString().padStart(2, '0');

const formatTime = (timestamp: number) => {
  const date = new Date(timestamp);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
};

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
