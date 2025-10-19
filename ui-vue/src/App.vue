<template>
  <div class="app-shell">
    <div class="content-row">
      <Sidebar class="sidebar" />
      <div class="main-area">
        <Network3D class="network-view" />
        <StatusBar
          :show-logs="showLogs"
          @toggle-logs="toggleLogs"
        />
      </div>
    </div>
    <Toast />
    <div
      v-if="showDone"
      style="position:fixed;inset:0;background:rgba(0,0,0,0.35);display:flex;align-items:center;justify-content:center;z-index:50;"
    >
      <div
        style="background:#ffffff;color:#0f172a;border-radius:8px;min-width:320px;max-width:90vw;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,0.2);"
      >
        <h3 style="margin-top:0;">训练完成</h3>
        <p>最终结果（epoch {{ finalSummary?.epoch ?? '--' }}）</p>
        <ul style="line-height:1.8;margin:0 0 12px 0;padding-left:18px;">
          <li>Loss: {{ finalSummary?.loss ?? '--' }}</li>
          <li>Acc: {{ finalSummary?.acc ?? '--' }}</li>
          <li>Best Acc: {{ finalSummary?.bestAcc ?? '--' }}</li>
          <li>Best Loss: {{ finalSummary?.bestLoss ?? '--' }}</li>
          <li>平均吞吐: {{ finalSummary?.avgTps ?? '--' }} samples/s</li>
          <li>本轮耗时: {{ finalSummary?.epochSec ?? '--' }} s</li>
        </ul>
        <div style="text-align:right;">
          <button
            type="button"
            @click="closeDone"
            style="padding:6px 12px;border:1px solid #d1d5db;background:#f9fafb;border-radius:6px;cursor:pointer;"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import Sidebar from './components/Sidebar.vue';
import Network3D from './components/Network3D.vue';
import StatusBar from './components/StatusBar.vue';
import Toast from './components/Toast.vue';
import { useUiStore } from './store/ui';

const store = useUiStore();

const showLogs = ref(false);
const toggleLogs = () => {
  showLogs.value = !showLogs.value;
};

const showDone = computed(() => store.showDoneModal);

const formatSummary = (value: number | null | undefined, digits: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '--';

const finalSummary = computed(() => {
  const summary = store.finalSummary;
  if (!summary) {
    return null;
  }
  return {
    epoch: summary.epoch ?? '--',
    loss: formatSummary(summary.loss, 4),
    acc: formatSummary(summary.acc, 4),
    bestAcc: formatSummary(summary.bestAcc, 4),
    bestLoss: formatSummary(summary.bestLoss, 4),
    avgTps: formatSummary(summary.avgThroughput, 1),
    epochSec: formatSummary(summary.epochSec, 1)
  };
});

const closeDone = () => {
  store.dismissDoneModal();
};
</script>
