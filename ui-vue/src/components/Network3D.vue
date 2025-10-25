<template>
  <div ref="container" class="canvas-wrapper"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry';
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

let edgeGeometry: LineSegmentsGeometry | null = null;
let edgeMaterial: LineMaterial | null = null;
let edgeSegments: LineSegments2 | null = null;
let edgeColors = new Float32Array(0);
let edgeColorAttrStart: THREE.InterleavedBufferAttribute | null = null;
let edgeColorAttrEnd: THREE.InterleavedBufferAttribute | null = null;
let edgeIntensity = new Float32Array(0);
let edgeIndexMap = new Map<string, number>();
let nodeToEdges = new Map<number, number[]>();
let edgesList: Array<{ src: number; dst: number }> = [];
let layerLabelGroup: THREE.Group | null = null;

const baseNodeColor = new THREE.Color('#1d4ed8');
const highlightNodeColor = new THREE.Color('#fde047');
const baseEdgeColor = new THREE.Color('#1f2937');
const highlightEdgeColor = new THREE.Color('#f97316');
const baseEdgeOpacity = 0.06;
const maxEdgeOpacity = 0.35;
const edgeOpacityRamp = 0.45;
const layerLabelOffset = 1.8;
const layerLabelLift = 0.4;
const layerLabelScale = 0.015;
const baseEdgeHSL = { h: 0, s: 0, l: 0 };
const highlightEdgeHSL = { h: 0, s: 0, l: 0 };
baseEdgeColor.getHSL(baseEdgeHSL);
highlightEdgeColor.getHSL(highlightEdgeHSL);
const edgeTempColor = new THREE.Color();
const boundsBox = new THREE.Box3();
const boundsCenter = new THREE.Vector3();
const boundsSize = new THREE.Vector3();
const tempVector = new THREE.Vector3();
const layerCenterTemp = new THREE.Vector3();

const clock = new THREE.Clock();
const backdropColorHex = '#e5e7eb';
const backdropOpacity = 0.7;
const suppressContextMenu = (event: MouseEvent) => {
  event.preventDefault();
};

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
  edgeColorAttrStart = null;
  edgeColorAttrEnd = null;
  edgeIntensity = new Float32Array(0);
  edgeIndexMap = new Map();
  nodeToEdges = new Map();
  edgesList = [];
};

const disposeLayerLabels = () => {
  if (layerLabelGroup && scene) {
    layerLabelGroup.children.forEach((child) => {
      if (child instanceof THREE.Sprite) {
        const material = child.material as THREE.SpriteMaterial | THREE.SpriteMaterial[] | undefined;
        if (material && !Array.isArray(material)) {
          material.map?.dispose();
          material.dispose();
        }
      }
    });
    scene.remove(layerLabelGroup);
  }
  layerLabelGroup = null;
};

const drawRoundedRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) => {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
};

const createLayerLabelSprite = (text: string) => {
  if (typeof document === 'undefined') {
    return null;
  }
  const fontSize = 48;
  const paddingX = 30;
  const paddingY = 18;
  const deviceRatio = typeof window !== 'undefined' ? Math.max(1, window.devicePixelRatio || 1) : 1;
  const canvas = document.createElement('canvas');
  const measureCtx = canvas.getContext('2d');
  if (!measureCtx) {
    return null;
  }
  measureCtx.font = `${fontSize}px "Inter", "SF Pro Display", "PingFang SC", "Helvetica Neue", sans-serif`;
  const textWidth = measureCtx.measureText(text).width;
  const width = textWidth + paddingX * 2;
  const height = fontSize + paddingY * 2;
  canvas.width = Math.ceil(width * deviceRatio);
  canvas.height = Math.ceil(height * deviceRatio);
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }
  ctx.scale(deviceRatio, deviceRatio);
  ctx.font = measureCtx.font;
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = 'rgba(59, 130, 246, 0.35)';
  ctx.lineWidth = 2;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
  drawRoundedRect(ctx, 0, 0, width, height, 14);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#f8fafc';
  ctx.fillText(text, paddingX, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width * layerLabelScale, height * layerLabelScale, 1);
  sprite.renderOrder = 10;
  return sprite;
};

const buildLayerLabels = (layouts: LayerLayout[]) => {
  disposeLayerLabels();
  if (!scene || !layouts.length) {
    return;
  }
  layerLabelGroup = new THREE.Group();
  layouts.forEach((layout, index) => {
    if (layout.count === 0) {
      return;
    }
    const sprite = createLayerLabelSprite(`层 ${index + 1}`);
    if (!sprite) {
      return;
    }
    layerCenterTemp.set(0, 0, 0);
    for (let i = 0; i < layout.count; i += 1) {
      const offset = i * 3;
      layerCenterTemp.x += layout.positions[offset];
      layerCenterTemp.y += layout.positions[offset + 1];
      layerCenterTemp.z += layout.positions[offset + 2];
    }
    layerCenterTemp.multiplyScalar(1 / layout.count);
    const direction = layerCenterTemp.x >= 0 ? 1 : -1;
    const horizontalPadding = layerLabelOffset + sprite.scale.x / 2;
    const labelX = layerCenterTemp.x + direction * horizontalPadding;
    const labelY = layerCenterTemp.y + layerLabelLift;
    sprite.position.set(labelX, labelY, layerCenterTemp.z);
    layerLabelGroup!.add(sprite);
  });
  scene.add(layerLabelGroup);
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

  edgeGeometry = new LineSegmentsGeometry();
  edgeGeometry.setPositions(positions);
  edgeGeometry.setColors(edgeColors);
  edgeColorAttrStart = edgeGeometry.getAttribute('instanceColorStart') as THREE.InterleavedBufferAttribute;
  edgeColorAttrEnd = edgeGeometry.getAttribute('instanceColorEnd') as THREE.InterleavedBufferAttribute;

  edgeMaterial = new LineMaterial({
    vertexColors: true,
    transparent: true,
    opacity: baseEdgeOpacity,
    linewidth: 3.2,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });
  const width = container.value?.clientWidth ?? window.innerWidth;
  const height = container.value?.clientHeight ?? window.innerHeight;
  edgeMaterial.resolution.set(Math.max(1, width), Math.max(1, height));

  edgeSegments = new LineSegments2(edgeGeometry, edgeMaterial);
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
  nodeGeometry = new THREE.SphereGeometry(0.18, 18, 18);
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
    disposeLayerLabels();
    updateCameraForLayout();
    return;
  }

  buildNodes(layouts);
  buildEdges(layouts);
  buildLayerLabels(layouts);
  updateCameraForLayout();
};

const highlightNeurons = (globalIndices: number[]) => {
  let touched = false;
  globalIndices.forEach((index) => {
    if (index >= 0 && index < glow.length) {
      glow[index] = 1.2;
      const offset = index * 3;
      nodeColors[offset] = highlightNodeColor.r;
      nodeColors[offset + 1] = highlightNodeColor.g;
      nodeColors[offset + 2] = highlightNodeColor.b;
      touched = true;
    }
  });
  if (touched && nodeColorAttr) {
    nodeColorAttr.needsUpdate = true;
  }
};

const highlightEdges = (edgePairs: Array<[number, number]> | undefined, neurons: number[]) => {
  const touched = new Set<number>();
  let matched = false;
  if (edgePairs && edgePairs.length > 0) {
    edgePairs.forEach(([src, dst]) => {
      const key = `${src}:${dst}`;
      const edgeIdx = edgeIndexMap.get(key);
      if (edgeIdx !== undefined) {
        edgeIntensity[edgeIdx] = Math.max(edgeIntensity[edgeIdx] ?? 0, 2.2);
        const base = edgeIdx * 6;
        edgeColors[base] = highlightEdgeColor.r;
        edgeColors[base + 1] = highlightEdgeColor.g;
        edgeColors[base + 2] = highlightEdgeColor.b;
        edgeColors[base + 3] = highlightEdgeColor.r;
        edgeColors[base + 4] = highlightEdgeColor.g;
        edgeColors[base + 5] = highlightEdgeColor.b;
        touched.add(edgeIdx);
        matched = true;
      }
    });
  }
  if (!matched) {
    neurons.forEach((neuron) => {
      const edgesForNeuron = nodeToEdges.get(neuron);
      if (!edgesForNeuron) {
        return;
      }
      edgesForNeuron.forEach((edgeIdx) => {
        edgeIntensity[edgeIdx] = Math.max(edgeIntensity[edgeIdx] ?? 0, 2.0);
        const base = edgeIdx * 6;
        edgeColors[base] = highlightEdgeColor.r;
        edgeColors[base + 1] = highlightEdgeColor.g;
        edgeColors[base + 2] = highlightEdgeColor.b;
        edgeColors[base + 3] = highlightEdgeColor.r;
        edgeColors[base + 4] = highlightEdgeColor.g;
        edgeColors[base + 5] = highlightEdgeColor.b;
        touched.add(edgeIdx);
      });
    });
  }
  if (touched.size > 0) {
    if (edgeColorAttrStart?.data) {
      edgeColorAttrStart.data.needsUpdate = true;
    }
    if (edgeColorAttrEnd?.data) {
      edgeColorAttrEnd.data.needsUpdate = true;
    }
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
  const glowDecay = delta * 0.9;
  const edgeDecay = delta * 0.75;
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
  if (edgeColorAttrStart?.data) {
    edgeColorAttrStart.data.needsUpdate = true;
  }
  if (edgeColorAttrEnd?.data) {
    edgeColorAttrEnd.data.needsUpdate = true;
  }
  if (edgeMaterial) {
    const extraOpacity = Math.min(maxEdgeOpacity - baseEdgeOpacity, maxEdge * edgeOpacityRamp);
    edgeMaterial.opacity = baseEdgeOpacity + extraOpacity;
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
  if (edgeMaterial) {
    edgeMaterial.resolution.set(Math.max(1, clientWidth), Math.max(1, clientHeight));
  }
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
  renderer.domElement.addEventListener('contextmenu', suppressContextMenu);

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
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN
  };
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
    renderer.domElement.removeEventListener('contextmenu', suppressContextMenu);
    renderer.dispose();
    const canvas = renderer.domElement;
    canvas.parentElement?.removeChild(canvas);
  }
  disposeNodes();
  disposeEdges();
  disposeLayerLabels();
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
  () => store.spikeSequence,
  () => {
    const spike = store.spikes[store.spikes.length - 1];
    if (spike) {
      processSpike(spike);
    }
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
