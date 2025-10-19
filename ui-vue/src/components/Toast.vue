<template>
  <transition name="toast-fade">
    <div v-if="toast" class="toast" :class="`toast-${toast.type}`">
      {{ toast.message }}
    </div>
  </transition>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, watch } from 'vue';
import { useUiStore } from '../store/ui';

const store = useUiStore();
const toast = computed(() => store.toast);

let hideHandle: number | null = null;

const scheduleHide = (at: number, duration = 3000) => {
  if (hideHandle !== null) {
    window.clearTimeout(hideHandle);
    hideHandle = null;
  }
  hideHandle = window.setTimeout(() => {
    store.clearToast(at);
  }, duration);
};

watch(
  toast,
  (value) => {
    if (!value) {
      if (hideHandle !== null) {
        window.clearTimeout(hideHandle);
        hideHandle = null;
      }
      return;
    }
    scheduleHide(value.at, value.duration);
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  if (hideHandle !== null) {
    window.clearTimeout(hideHandle);
    hideHandle = null;
  }
});
</script>

<style scoped>
.toast {
  position: fixed;
  top: 16px;
  right: 16px;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.4;
  color: #fff;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
  max-width: 320px;
  z-index: 1000;
  backdrop-filter: blur(6px);
}

.toast-info {
  background: rgba(24, 144, 255, 0.92);
}

.toast-error {
  background: rgba(245, 63, 63, 0.92);
}

.toast-fade-enter-active,
.toast-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.toast-fade-enter-from,
.toast-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}
</style>
