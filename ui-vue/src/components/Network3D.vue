<template>
  <div ref="container" class="canvas-wrapper"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useUiStore } from '../store/ui';
import type { LayerLayout, SpikeEntry } from '../types';

const MAX_EDGES = 4000;
const store = useUiStore();
const container = ref<HTMLDivElement | null>(null);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let animationHandle = 0;
let nodeMesh: THREE.InstancedMesh<THREE.SphereGeometry, THREE.MeshPhongMaterial> | null = null;
let nodePositions = new Float32Array(0);
let nodeColors = new Float32Array(0);
let glow = new Float32Array(0);
let nodeColorAttr: THREE.InstancedBufferAttribute | null = null;
let nodeGeometry: THREE.SphereGeometry | null = null;
let nodeMaterial: THREE.MeshPhongMaterial | null = null;

let edgeGeometry: THREE.BufferGeometry | null = null;
let edgeMaterial: THREE.LineBasicMaterial | null = null;
let edgeSegments: THREE.LineSegments | null = null;
let edgeColors = new Float32Array(0);
let edgeColorAttr: THREE.BufferAttribute | null = null;
let edgeIntensity = new Float32Array(0);
let edgeIndexMap = new Map<string, number>();
let nodeToEdges = new Map<number, number[]>();
let edgesList: Array<{ src: number; dst: number }> = [];

const baseNodeColor = new THREE.Color('#1d4ed8');
const highlightNodeColor = new THREE.Color('#fde047');
const baseEdgeColor = new THREE.Color('#1f2937');
const highlightEdgeColor = new THREE.Color('#fde047');
const baseEdgeHSL = { h: 0, s: 0, l: 0 };
const highlightEdgeHSL = { h: 0, s: 0, l: 0 };
baseEdgeColor.getHSL(baseEdgeHSL);
highlightEdgeColor.getHSL(highlightEdgeHSL);
const edgeTempColor = new THREE.Color();
const boundsBox = new THREE.Box3();
const boundsCenter = new THREE.Vector3();
const boundsSize = new THREE.Vector3();
const tempVector = new THREE.Vector3();

const clock = new THREE.Clock();
const backdropColorHex = '#e5e7eb';
const backdropOpacity = 0.7;

const disposeNodes = () => {
  if (nodeMesh && scene) {
    scene.remove(nodeMesh);
    nodeMesh.geometry.dispose();
    nodeMesh.material.dispose();
  }
  nodeMesh = null;
  nodeGeometry = null;
  nodeMaterial = null;
  nodePositions = new Float32Array(0);
  nodeColors = new Float32Array(0);
  glow = new Float32Array(0);
  nodeColorAttr = null;
};

const disposeEdges = () => {
  if (edgeSegments && scene) {
    scene.remove(edgeSegments);
  }
  edgeSegments = null;
  if (edgeGeometry) {
    edgeGeometry.dispose();
  }
  if (edgeMaterial) {
    edgeMaterial.dispose();
  }
  edgeGeometry = null;
  edgeMaterial = null;
  edgeColors = new Float32Array(0);
  edgeColorAttr = null;
  edgeIntensity = new Float32Array(0);
  edgeIndexMap = new Map();
  nodeToEdges = new Map();
  edgesList = [];
};

const buildEdges = (layouts: LayerLayout[]) => {
  disposeEdges();
  if (!scene) {
    return;
  }

  edgesList = [];
  edgeIndexMap = new Map();
  nodeToEdges = new Map();

  let edgeCounter = 0;
  const layerPairs = Math.max(1, layouts.length - 1);
  const budgetPerPair = Math.max(1, Math.floor(MAX_EDGES / layerPairs));

  outer: for (let layer = 0; layer < layouts.length - 1; layer += 1) {
    const from = layouts[layer];
    const to = layouts[layer + 1];
    const totalCombos = from.count * to.count;
    const allow = Math.min(totalCombos, budgetPerPair);
    const stride = Math.max(1, Math.floor(totalCombos / allow));

    for (let i = 0; i < from.count; i += 1) {
      for (let j = 0; j < to.count; j += 1) {
        const comboIndex = i * to.count + j;
        if (comboIndex % stride !== 0) {
          continue;
        }
        const src = from.startIndex + i;
        const dst = to.startIndex + j;
        edgesList.push({ src, dst });
        edgeIndexMap.set(`${src}:${dst}`, edgeCounter);
        if (!nodeToEdges.has(src)) {
          nodeToEdges.set(src, []);
        }
        if (!nodeToEdges.has(dst)) {
          nodeToEdges.set(dst, []);
        }
        nodeToEdges.get(src)!.push(edgeCounter);
        nodeToEdges.get(dst)!.push(edgeCounter);
        edgeCounter += 1;
        if (edgeCounter >= MAX_EDGES) {
          break outer;
        }
      }
    }
  }

  if (edgesList.length === 0) {
    return;
  }

  edgeGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(edgesList.length * 6);
  edgeColors = new Float32Array(edgesList.length * 6);
  edgeIntensity = new Float32Array(edgesList.length);

  edgesList.forEach((edge, index) => {
    const srcOffset = edge.src * 3;
    const dstOffset = edge.dst * 3;
    positions.set(
      [
        nodePositions[srcOffset],
        nodePositions[srcOffset + 1],
        nodePositions[srcOffset + 2],
        nodePositions[dstOffset],
        nodePositions[dstOffset + 1],
        nodePositions[dstOffset + 2]
      ],
      index * 6
    );
    for (let v = 0; v < 2; v += 1) {
      edgeColors[index * 6 + v * 3] = baseEdgeColor.r;
      edgeColors[index * 6 + v * 3 + 1] = baseEdgeColor.g;
      edgeColors[index * 6 + v * 3 + 2] = baseEdgeColor.b;
    }
  });

  edgeGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  edgeColorAttr = new THREE.BufferAttribute(edgeColors, 3);
  edgeGeometry.setAttribute('color', edgeColorAttr);

  edgeMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.5,
    linewidth: 2,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  edgeSegments = new THREE.LineSegments(edgeGeometry, edgeMaterial);
  scene.add(edgeSegments);
};

const buildNodes = (layouts: LayerLayout[]) => {
  disposeNodes();
  if (!scene) {
    return;
  }

  const totalInstances = layouts.reduce((sum, layer) => sum + layer.count, 0);
  if (totalInstances === 0) {
    return;
  }

  nodePositions = new Float32Array(totalInstances * 3);
  nodeColors = new Float32Array(totalInstances * 3);
  glow = new Float32Array(totalInstances);
  nodeGeometry = new THREE.SphereGeometry(0.25, 20, 20);
  nodeMaterial = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    emissive: 0x0e223a,
    emissiveIntensity: 0.4,
    shininess: 50,
    vertexColors: true
  });
  nodeMesh = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, totalInstances);
  nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const matrix = new THREE.Matrix4();

  layouts.forEach((layer) => {
    nodePositions.set(layer.positions, layer.startIndex * 3);
    for (let i = 0; i < layer.count; i += 1) {
      const globalIndex = layer.startIndex + i;
      const offset = i * 3;
      const px = layer.positions[offset];
      const py = layer.positions[offset + 1];
      const pz = layer.positions[offset + 2];
      matrix.setPosition(px, py, pz);
      nodeMesh!.setMatrixAt(globalIndex, matrix);
      nodeColors[globalIndex * 3] = baseNodeColor.r;
      nodeColors[globalIndex * 3 + 1] = baseNodeColor.g;
      nodeColors[globalIndex * 3 + 2] = baseNodeColor.b;
      glow[globalIndex] = 0;
    }
  });

  nodeColorAttr = new THREE.InstancedBufferAttribute(nodeColors, 3);
  nodeColorAttr.setUsage(THREE.DynamicDrawUsage);
  nodeMesh.instanceColor = nodeColorAttr;
  nodeMesh.instanceMatrix.needsUpdate = true;

  scene.add(nodeMesh);
};

const updateCameraForLayout = () => {
  if (!camera || !controls) {
    return;
  }
  if (!nodePositions.length) {
    camera.position.set(-12, 10, 40);
    controls.target.set(0, 0, 0);
    controls.update();
    return;
  }
  boundsBox.makeEmpty();
  for (let i = 0; i < nodePositions.length; i += 3) {
    tempVector.set(nodePositions[i], nodePositions[i + 1], nodePositions[i + 2]);
    boundsBox.expandByPoint(tempVector);
  }
  boundsBox.getCenter(boundsCenter);
  boundsBox.getSize(boundsSize);
  const maxExtent = Math.max(boundsSize.x, boundsSize.y, boundsSize.z);
  const distance = Math.max(12, maxExtent * 1.6 + store.cfg.layers * 0.8);
  camera.position.set(boundsCenter.x - distance * 0.4, boundsCenter.y + distance * 0.35, boundsCenter.z + distance);
  controls.target.copy(boundsCenter);
  controls.update();
};

const rebuildScene = (layouts: LayerLayout[]) => {
  if (!layouts.length) {
    disposeNodes();
    disposeEdges();
    updateCameraForLayout();
    return;
  }

  buildNodes(layouts);
  buildEdges(layouts);
  updateCameraForLayout();
};

const highlightNeurons = (globalIndices: number[]) => {
  globalIndices.forEach((index) => {
    if (index >= 0 && index < glow.length) {
      glow[index] = Math.max(glow[index], 2.2);
    }
  });
};

const highlightEdges = (edgePairs: Array<[number, number]> | undefined, neurons: number[]) => {
  const touched = new Set<number>();
  if (edgePairs && edgePairs.length > 0) {
    edgePairs.forEach(([src, dst]) => {
      const key = `${src}:${dst}`;
      const edgeIdx = edgeIndexMap.get(key);
      if (edgeIdx !== undefined) {
        edgeIntensity[edgeIdx] = Math.max(edgeIntensity[edgeIdx] ?? 0, 1.8);
        touched.add(edgeIdx);
      }
    });
  } else {
    neurons.forEach((neuron) => {
      const edgesForNeuron = nodeToEdges.get(neuron);
      if (!edgesForNeuron) {
        return;
      }
      edgesForNeuron.forEach((edgeIdx) => {
        edgeIntensity[edgeIdx] = Math.max(edgeIntensity[edgeIdx] ?? 0, 1.8);
        touched.add(edgeIdx);
      });
    });
  }
  return touched;
};

const processSpike = (spike: SpikeEntry) => {
  if (glow.length === 0) {
    return;
  }
  const layer = store.layersLayout[spike.layer];
  if (!layer) {
    return;
  }
  const neurons: number[] = [];
  spike.neurons.forEach((localIndex) => {
    const clamped = Math.min(Math.max(0, Math.floor(localIndex)), layer.count - 1);
    neurons.push(layer.startIndex + clamped);
  });
  const total = glow.length;
  const edges =
    spike.edges?.map(([src, dst]) => {
      let srcGlobal = Math.floor(src);
      let dstGlobal = Math.floor(dst);
      if (srcGlobal >= total || dstGlobal >= total) {
        const current = store.layersLayout[spike.layer];
        const next = store.layersLayout[Math.min(spike.layer + 1, store.layersLayout.length - 1)];
        const prev = store.layersLayout[Math.max(0, spike.layer - 1)];
        if (current) {
          srcGlobal = current.startIndex + Math.min(current.count - 1, Math.max(0, srcGlobal));
        } else if (prev) {
          srcGlobal = prev.startIndex + Math.min(prev.count - 1, Math.max(0, srcGlobal));
        }
        if (next) {
          dstGlobal = next.startIndex + Math.min(next.count - 1, Math.max(0, dstGlobal));
        } else if (current) {
          dstGlobal = current.startIndex + Math.min(current.count - 1, Math.max(0, dstGlobal));
        }
      }
      return [srcGlobal, dstGlobal] as [number, number];
    }) ?? undefined;

  console.debug('[Network3D] processSpike', {
    layer: spike.layer,
    neurons,
    highlightEdgesCount: edges?.length ?? 0
  });
  highlightNeurons(neurons);
  highlightEdges(edges, neurons);
};

const animate = () => {
  animationHandle = requestAnimationFrame(animate);
  if (!renderer || !scene || !camera) {
    return;
  }
  const delta = clock.getDelta();
  const glowDecay = delta * 0.8;
  const edgeDecay = delta * 0.7;
  let colorDirty = false;

  if (glow.length && nodeMesh && nodeColorAttr) {
    for (let i = 0; i < glow.length; i += 1) {
      if (glow[i] > 0) {
        glow[i] = Math.max(0, glow[i] - glowDecay);
        colorDirty = true;
      }
      const t = Math.min(1, glow[i]);
      nodeColors[i * 3] = THREE.MathUtils.lerp(baseNodeColor.r, highlightNodeColor.r, t);
      nodeColors[i * 3 + 1] = THREE.MathUtils.lerp(baseNodeColor.g, highlightNodeColor.g, t);
      nodeColors[i * 3 + 2] = THREE.MathUtils.lerp(baseNodeColor.b, highlightNodeColor.b, t);
    }
    if (colorDirty) {
      nodeColorAttr.needsUpdate = true;
    }
  }

  let maxEdge = 0;
  for (let i = 0; i < edgeIntensity.length; i += 1) {
    if (edgeIntensity[i] > 0) {
      edgeIntensity[i] = Math.max(0, edgeIntensity[i] - edgeDecay);
    }
    if (edgeIntensity[i] > maxEdge) {
      maxEdge = edgeIntensity[i];
    }
    const weight = Math.min(1, edgeIntensity[i]);
    const h = THREE.MathUtils.lerp(baseEdgeHSL.h, highlightEdgeHSL.h, weight);
    const s = THREE.MathUtils.lerp(baseEdgeHSL.s, highlightEdgeHSL.s, weight);
    const l = THREE.MathUtils.lerp(baseEdgeHSL.l, highlightEdgeHSL.l, weight + weight * 0.2);
    edgeTempColor.setHSL(h, s, Math.min(1, l));
    const colorOffset = i * 6;
    edgeColors[colorOffset] = edgeTempColor.r;
    edgeColors[colorOffset + 1] = edgeTempColor.g;
    edgeColors[colorOffset + 2] = edgeTempColor.b;
    edgeColors[colorOffset + 3] = edgeTempColor.r;
    edgeColors[colorOffset + 4] = edgeTempColor.g;
    edgeColors[colorOffset + 5] = edgeTempColor.b;
  }
  if (edgeColorAttr) {
    edgeColorAttr.needsUpdate = true;
  }
  if (edgeMaterial) {
    edgeMaterial.opacity = 0.25 + Math.min(0.6, maxEdge * 0.85);
  }

  controls?.update();
  renderer.render(scene, camera);
};

const resize = () => {
  if (!container.value || !renderer || !camera) {
    return;
  }
  const { clientWidth, clientHeight } = container.value;
  if (clientWidth === 0 || clientHeight === 0) {
    return;
  }
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
};

onMounted(() => {
  if (!container.value) {
    return;
  }
  scene = new THREE.Scene();
  scene.background = null;
  const { clientWidth, clientHeight } = container.value;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(clientWidth || 800, clientHeight || 600);
  renderer.setClearColor(backdropColorHex, backdropOpacity);
  container.value.appendChild(renderer.domElement);

  camera = new THREE.PerspectiveCamera(50, (clientWidth || 800) / (clientHeight || 600), 0.1, 1000);
  camera.position.set(-12, 10, 40);
  camera.lookAt(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x9fb6d9, 0.55);
  const keyLight = new THREE.PointLight(0xffffff, 1.2);
  keyLight.position.set(18, 14, 36);
  scene.add(ambient);
  scene.add(hemi);
  scene.add(keyLight);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 6;
  controls.maxDistance = 120;
  controls.target.set(0, 0, 0);
  controls.update();

  resize();
  window.addEventListener('resize', resize);
  clock.start();
  rebuildScene(store.layersLayout);
  animate();
});

onBeforeUnmount(() => {
  cancelAnimationFrame(animationHandle);
  window.removeEventListener('resize', resize);
  controls?.dispose();
  if (renderer) {
    renderer.dispose();
    const canvas = renderer.domElement;
    canvas.parentElement?.removeChild(canvas);
  }
  disposeNodes();
  disposeEdges();
  scene = null;
  camera = null;
  renderer = null;
});

const refreshLayout = (layouts: LayerLayout[]) => {
  rebuildScene(layouts);
  resize();
};

watch(
  [() => store.layersLayout, () => store.cfg.layers, () => store.cfg.network_size],
  ([layouts]) => {
    refreshLayout(layouts as LayerLayout[]);
  },
  { immediate: true, deep: true }
);

watch(
  () => store.spikes.length,
  (length) => {
    if (!length) {
      return;
    }
    const spike = store.spikes[length - 1];
    processSpike(spike);
  }
);
</script>

<style scoped>
.canvas-wrapper {
  width: 100%;
  height: calc(100vh - 40px);
  position: relative;
}
</style>
