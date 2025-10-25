<template>
  <aside>
    <div class="scrollable">
      <section>
        <label>
          <span>数据集</span>
          <select v-model="dataset">
            <option v-for="option in datasetOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </label>
        <button @click="downloadDataset" :disabled="downloadLocked">
          {{ downloadButtonText }}
        </button>
      </section>
      <fieldset :disabled="parametersLocked">
        <legend>训练参数</legend>
        <label>
          <span>network_size</span>
          <input type="number" min="1" v-model.number="networkSize" />
          <span class="hint">整体神经元数量，建议 512 ~ 4096，越大越细腻但渲染压力更高。</span>
        </label>
        <label>
          <span>layers</span>
          <input type="number" min="1" v-model.number="layers" />
          <span class="hint">网络层数，2 ~ 6 层较易观察；层数越多可视化深度越大。</span>
        </label>
        <label>
          <span>lr</span>
          <input type="number" step="0.0001" v-model.number="lr" />
          <span class="hint">学习率，默认 1e-3；尝试 5e-4 ~ 2e-3 平衡收敛速度与稳定性。</span>
        </label>
        <label>
          <span>K</span>
          <input type="number" min="1" v-model.number="K" />
          <span class="hint">K 表示突触邻域大小，10~30 可形成适中稀疏连接。</span>
        </label>
        <label>
          <span>tol</span>
          <input type="number" step="0.000001" v-model.number="tol" />
          <span class="hint">容差阈值控制停止条件，越小越精准但耗时更久。</span>
        </label>
        <label>
          <span>epochs</span>
          <input type="number" min="1" v-model.number="epochs" />
          <span class="hint">训练轮次数量，通常 3~10 可观察收敛趋势。</span>
        </label>
      </fieldset>
    </div>
    <div class="buttons">
      <button @click="handlePrimaryAction" :disabled="primaryButtonDisabled">
        <span class="train-icon">{{ primaryButtonIcon }}</span>
        {{ primaryButtonLabel }}
      </button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import axios from 'axios';
import { useUiStore } from '../store/ui';
import type { DatasetListPayload, DatasetName } from '../types';

const store = useUiStore();

interface DatasetOption {
  value: DatasetName;
  label: string;
  status?: string;
  message?: string | null;
  installed?: boolean;
}

const DEFAULT_DATASETS: DatasetOption[] = [
  { value: 'MNIST', label: 'MNIST 手写数字' },
  { value: 'FASHION', label: 'Fashion-MNIST 服饰' }
];

const datasetOptions = ref<DatasetOption[]>(ensureDefaultDatasets([{ value: store.cfg.dataset, label: store.cfg.dataset }]));
const isBusy = ref(false);
const hasInitialized = ref(false);

const isTraining = computed(() => store.status === 'Training');
const dataset = computed({
  get: () => store.cfg.dataset,
  set: (value: DatasetName) => store.setDataset(value)
});
const networkSize = computed({
  get: () => store.cfg.network_size,
  set: (value: number) => store.setCfg({ network_size: value })
});
const layers = computed({
  get: () => store.cfg.layers,
  set: (value: number) => store.setCfg({ layers: value })
});
const lr = computed({
  get: () => store.cfg.lr,
  set: (value: number) => store.setCfg({ lr: value })
});
const K = computed({
  get: () => store.cfg.K,
  set: (value: number) => store.setCfg({ K: value })
});
const tol = computed({
  get: () => store.cfg.tol,
  set: (value: number) => store.setCfg({ tol: value })
});
const epochs = computed({
  get: () => store.cfg.epochs,
  set: (value: number) => store.setCfg({ epochs: value })
});

const downloadLocked = computed(() => store.isDownloadActive || isBusy.value);
const parametersLocked = computed(() => isBusy.value || isTraining.value);
const downloadPercentText = computed(() => `${store.downloadPercent}%`);
const downloadButtonText = computed(() => {
  if (store.isDownloadActive) {
    const percent = downloadPercentText.value;
    const name = store.download.name || dataset.value;
    return `下载中 ${name} · ${percent}`;
  }
  return '下载到本地';
});
const primaryButtonState = computed<'init' | 'initializing' | 'start' | 'stop'>(() => {
  if (store.status === 'Initializing') {
    return 'initializing';
  }
  if (isTraining.value) {
    return 'stop';
  }
  if (hasInitialized.value) {
    return 'start';
  }
  return 'init';
});
const primaryButtonDisabled = computed(() => isBusy.value || store.status === 'Initializing');
const primaryButtonIcon = computed(() => {
  switch (primaryButtonState.value) {
    case 'init':
      return '⚙';
    case 'initializing':
      return '…';
    case 'stop':
      return '■';
    case 'start':
    default:
      return '▶';
  }
});
const primaryButtonLabel = computed(() => {
  switch (primaryButtonState.value) {
    case 'init':
      return '初始化';
    case 'initializing':
      return '初始化中…';
    case 'stop':
      return '停止';
    case 'start':
    default:
      return '训练';
  }
});

watch(
  () => [
    store.cfg.dataset,
    store.cfg.mode,
    store.cfg.network_size,
    store.cfg.layers,
    store.cfg.lr,
    store.cfg.K,
    store.cfg.tol,
    store.cfg.T,
    store.cfg.epochs
  ],
  () => {
    hasInitialized.value = false;
  }
);

const withBusy = async (task: () => Promise<void>, opts: { allowWhenDownloading?: boolean } = {}) => {
  if (isBusy.value) {
    return;
  }
  if (store.isDownloadActive && !opts.allowWhenDownloading) {
    store.showToast('数据集下载进行中，请稍候完成后再试', 'error');
    return;
  }
  isBusy.value = true;
  try {
    await task();
  } finally {
    isBusy.value = false;
  }
};

const loadDatasets = async () => {
  try {
    const { data } = await axios.get<DatasetListPayload | DatasetName[]>('/api/datasets');
    const options = ensureDefaultDatasets(normalizeDatasetList(data));
    if (options.length > 0) {
      datasetOptions.value = options;
      if (!options.some((option) => option.value === store.cfg.dataset)) {
        store.setDataset(options[0]?.value ?? store.cfg.dataset);
      }
      return;
    }
    console.warn('Dataset API returned empty payload', data);
    store.showToast('数据集列表为空', 'error');
  } catch (err) {
    console.warn('Failed to load datasets', err);
    store.showToast('无法加载数据集列表', 'error');
  }
};

onMounted(() => {
  void loadDatasets();
});

const downloadDataset = () =>
  withBusy(async () => {
    try {
      await axios.post(
        '/api/datasets/download',
        { name: dataset.value },
        {}
      );
    } catch (err) {
      console.warn('Download failed', err);
      store.showToast('数据集下载失败', 'error');
    }
  }, { allowWhenDownloading: true });

const initTraining = () =>
  withBusy(async () => {
    try {
      store.pushPlainLog(`发送训练初始化请求：${store.cfg.dataset}`, 'INFO');
      console.info('[UI] init training payload', store.cfg);
      store.setStatus('Initializing');
      await axios.post('/api/train/init', {
        dataset: store.cfg.dataset,
        mode: store.cfg.mode,
        network_size: store.cfg.network_size,
        layers: store.cfg.layers,
        lr: store.cfg.lr,
        K: store.cfg.K,
        tol: store.cfg.tol,
        T: store.cfg.T,
        epochs: store.cfg.epochs
      });
      store.setStatus('Idle');
      store.pushPlainLog(`训练初始化完成：${store.cfg.dataset}`, 'INFO');
      hasInitialized.value = true;
    } catch (err) {
      console.warn('Init failed', err);
      store.showToast('初始化训练失败', 'error');
      store.setStatus('Idle');
      store.pushPlainLog('训练初始化失败', 'ERROR');
    }
  });

const startTraining = () =>
  withBusy(async () => {
    try {
      await axios.post('/api/train/start', {});
      store.setStatus('Training');
    } catch (err) {
      console.warn('Start failed', err);
      store.showToast('启动训练失败', 'error');
    }
  });

const stopTraining = () =>
  withBusy(async () => {
    try {
      await axios.post('/api/train/stop', {});
      store.setStatus('Stopped');
    } catch (err) {
      console.warn('Stop failed', err);
      store.showToast('停止训练失败', 'error');
    }
  });

const handlePrimaryAction = () => {
  const state = primaryButtonState.value;
  if (state === 'init') {
    void initTraining();
    return;
  }
  if (state === 'start') {
    void startTraining();
    return;
  }
  if (state === 'stop') {
    void stopTraining();
    return;
  }
};

function normalizeDatasetList(payload: unknown): DatasetOption[] {
  const result: DatasetOption[] = [];
  const seen = new Set<string>();
  const pushOption = (option: DatasetOption | null) => {
    if (!option || seen.has(option.value)) {
      return;
    }
    seen.add(option.value);
    result.push(option);
  };

  if (Array.isArray(payload)) {
    payload.forEach((entry) => pushOption(toDatasetOption(entry)));
    return result;
  }
  if (payload && typeof payload === 'object') {
    const obj = payload as DatasetListPayload;
    if (Array.isArray(obj.datasets)) {
      obj.datasets.forEach((entry) => pushOption(toDatasetOption(entry)));
    }
    if (Array.isArray(obj.available)) {
      obj.available.forEach((entry) => pushOption(toDatasetOption(entry)));
    }
    if (Array.isArray(obj.installed)) {
      obj.installed.forEach((entry) => pushOption(toDatasetOption(entry, 'installed')));
    }
  }
  return result;
}

function toDatasetOption(entry: unknown, statusHint?: string): DatasetOption | null {
  if (typeof entry === 'string') {
    const value = entry.trim();
    if (!value) {
      return null;
    }
    return { value: value as DatasetName, label: value, status: statusHint };
  }
  if (entry && typeof entry === 'object') {
    const typed = entry as {
      value?: unknown;
      name?: unknown;
      label?: unknown;
      status?: unknown;
      message?: unknown;
      installed?: unknown;
    };
    const value =
      typeof typed.value === 'string'
        ? typed.value
        : typeof typed.name === 'string'
          ? typed.name
          : typeof typed.label === 'string'
            ? typed.label
            : null;
    if (!value) {
      return null;
    }
    const installedFlag = typeof typed.installed === 'boolean' ? typed.installed : undefined;
    let statusValue: string | undefined;
    if (typeof typed.status === 'string') {
      statusValue = typed.status;
    } else if (installedFlag === true) {
      statusValue = "installed";
    } else if (installedFlag === false) {
      statusValue = "missing";
    } else {
      statusValue = statusHint;
    }
    const resolvedLabel =
      typeof typed.label === 'string'
        ? typed.label
        : typeof typed.name === 'string'
          ? typed.name
          : value;
    return {
      value: value as DatasetName,
      label: resolvedLabel,
      status: statusValue,
      message: typeof typed.message === 'string' ? typed.message : null,
      installed: installedFlag
    };
  }
  return null;
}

function ensureDefaultDatasets(options: DatasetOption[]): DatasetOption[] {
  const seen = new Set(options.map((option) => option.value));
  const withDefaults = [...options];
  DEFAULT_DATASETS.forEach((preset) => {
    if (!seen.has(preset.value)) {
      withDefaults.push({ ...preset, status: 'missing' });
    }
  });
  return withDefaults.map((option) => {
    const preset = DEFAULT_DATASETS.find((item) => item.value === option.value);
    const baseLabel = preset ? preset.label : option.label;
    const label =
      option.status === 'missing' || option.installed === false ? `${baseLabel}（未安装）` : baseLabel;
    return {
      ...option,
      label
    };
  });
}

watch(
  () => store.download.active,
  (active, prevActive) => {
    if (!active && prevActive) {
      void loadDatasets();
    }
  }
);
</script>
