"use strict";

const embedded = new URLSearchParams(window.location.search).get("embed") === "1";
if (embedded) document.body.classList.add("embedded");

function notifyParent(type, payload = {}) {
  if (embedded && window.parent !== window) {
    window.parent.postMessage({ type, ...payload }, window.location.origin);
  }
}

import * as THREE from "./vendor/three.module.min.js";
import {createFigureView} from "./figure.js";
import {filterTrajectories} from "./figure-utils.mjs";

const canvas = document.querySelector("#viewport");
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10_000);
camera.position.set(0, 0, 20);
const structureGroup = new THREE.Group();
scene.add(structureGroup);
scene.add(new THREE.HemisphereLight(0xffffff, 0x263630, 2.25));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.1);
keyLight.position.set(-4, 6, 9);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x9bc9ff, 1.35);
fillLight.position.set(7, -2, 5);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffd6a1, 1.15);
rimLight.position.set(-6, -5, -3);
scene.add(rimLight);
const raycaster = new THREE.Raycaster();
const pointerNDC = new THREE.Vector2();
const sphereGeometry = new THREE.SphereGeometry(1, 28, 20);
const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 20, 1, false);
const materialCache = new Map();
const initialOrientation = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(0.38, -0.55, 0, "YXZ"),
);
const rotationDelta = new THREE.Quaternion();
const horizontalRotation = new THREE.Quaternion();
const verticalRotation = new THREE.Quaternion();
const cameraUp = new THREE.Vector3();
const cameraRight = new THREE.Vector3();
const localUp = new THREE.Vector3(0, 1, 0);
const localRight = new THREE.Vector3(1, 0, 0);
const elements = {
  status: document.querySelector("#loadStatus"),
  play: document.querySelector("#playButton"),
  playIcon: document.querySelector("#playIcon"),
  timeline: document.querySelector("#timeline"),
  speed: document.querySelector("#speed"),
  speedOutput: document.querySelector("#speedOutput"),
  pointSize: document.querySelector("#pointSize"),
  sizeOutput: document.querySelector("#sizeOutput"),
  bondSize: document.querySelector("#bondSize"),
  bondSizeOutput: document.querySelector("#bondSizeOutput"),
  showBonds: document.querySelector("#showBonds"),
  qualityFilter: document.querySelector("#qualityFilter"),
  qualityCriterion: document.querySelector("#qualityCriterion"),
  qualityStatus: document.querySelector("#qualityStatus"),
  autoRotate: document.querySelector("#autoRotate"),
  autoFit: document.querySelector("#autoFit"),
  sourceName: document.querySelector("#sourceName"),
  atomCount: document.querySelector("#atomCount"),
  segmentCount: document.querySelector("#segmentCount"),
  frameValue: document.querySelector("#frameValue"),
  bondCount: document.querySelector("#bondCount"),
  bondNote: document.querySelector("#bondNote"),
  timeValue: document.querySelector("#timeValue"),
  checkpoint: document.querySelector("#checkpointName"),
  seed: document.querySelector("#seedValue"),
  hover: document.querySelector("#hoverCard"),
  modeLabel: document.querySelector("#modeLabel"),
  trajectoryTitle: document.querySelector("#trajectoryTitle"),
  sampleLabel: document.querySelector("#sampleLabel"),
  processSection: document.querySelector("#processSection"),
  sourceSigma: document.querySelector("#sourceSigma"),
  centerVariance: document.querySelector("#centerVariance"),
  clockPowers: document.querySelector("#clockPowers"),
  internalVariance: document.querySelector("#internalVariance"),
  rotationSigma: document.querySelector("#rotationSigma"),
  internalVarianceLabel: document.querySelector("#internalVarianceLabel"),
  rotationSigmaLabel: document.querySelector("#rotationSigmaLabel"),
  trajectoryPicker: document.querySelector("#trajectoryPicker"),
  trajectorySelect: document.querySelector("#trajectorySelect"),
  pairedSample: document.querySelector("#pairedSample"),
  pairedSampleNote: document.querySelector("#pairedSampleNote"),
  entityCountLabel: document.querySelector("#entityCountLabel"),
  colorLabel: document.querySelector("#colorLabel"),
  segmentKey: document.querySelector("#segmentKey"),
  colorNote: document.querySelector("#colorNote"),
  processLabel: document.querySelector("#processLabel"),
  atomColorControl: document.querySelector("#atomColorControl"),
  atomColorMode: document.querySelector("#atomColorMode"),
  fragmentPalette: document.querySelector("#fragmentPalette"),
  fragmentColor: document.querySelector("#fragmentColor"),
  fragmentColorControl: document.querySelector("#fragmentColorControl"),
  centerControl: document.querySelector("#centerControl"),
  centerMode: document.querySelector("#centerMode"),
  fragmentKey: document.querySelector("#fragmentKey"),
  atomKey: document.querySelector("#atomKey"),
  stage: document.querySelector(".stage"),
  backgroundPreset: document.querySelector("#backgroundPreset"),
  backgroundColor: document.querySelector("#backgroundColor"),
  saveVideo: document.querySelector("#saveVideoButton"),
  saveVideoLabel: document.querySelector("#saveVideoLabel"),
  videoStatus: document.querySelector("#videoStatus"),
  endHold: document.querySelector("#endHold"),
  endHoldOutput: document.querySelector("#endHoldOutput"),
  videoBitrate: document.querySelector("#videoBitrate"),
};

const state = {
  metadata: null,
  positions: null,
  atomLabels: null,
  catalog: null,
  catalogUrl: null,
  quality: null,
  visibleTrajectories: [],
  loadSerial: 0,
  frame: 0,
  playing: true,
  orientation: initialOrientation.clone(),
  zoom: 1,
  dragging: false,
  pointerX: -1,
  pointerY: -1,
  lastX: 0,
  lastY: 0,
  lastTick: performance.now(),
  frameAccumulator: 0,
  atomMeshes: [],
  bondMeshes: [],
  finalBondPairs: [],
  withinFragmentBondPairs: [],
  centroidMeshes: [],
  meshMetadata: null,
  sceneDirty: true,
  displayRadius: 1,
  recording: null,
};

// Classes follow the GEOM writer: H, B, C, N, O, F, Si, P, S, Cl, Br, I,
// Bi, mask. Colors are CPK-like but brightened for the dark canvas.
const atomTypeRGB = [
  [238, 244, 242], [255, 181, 181], [132, 145, 151], [82, 116, 255],
  [255, 77, 77], [83, 214, 115], [222, 162, 222], [255, 151, 64],
  [255, 211, 58], [83, 214, 115], [176, 91, 53], [145, 92, 183],
  [155, 100, 190], [112, 128, 122],
];
const covalentRadii = [
  0.31, 0.85, 0.76, 0.71, 0.66, 0.57, 1.11, 1.07,
  1.05, 1.02, 1.20, 1.39, 1.48, 0.76,
];
const maximumValence = [1, 3, 4, 4, 2, 1, 4, 5, 6, 1, 1, 1, 5, 0];
const moleculeScale = [
  0.72, 1.05, 1.0, 0.96, 0.92, 0.86, 1.12,
  1.08, 1.08, 1.04, 1.14, 1.2, 1.22, 0.68,
];
const geomCentroidRadiusScale = 1.5;
const geomCentroidOpacity = 0.5;
const geomCentroidFadeStart = 0.95;
const backgroundStorageKey = "lpgm-viewer-background";
const backgroundPresets = {
  "deep-green": {
    center: "#10201d",
    middle: "#091411",
    edge: "#050b0a",
    grid: "rgba(148, 208, 185, 0.035)",
  },
  graphite: {
    center: "#30363a",
    middle: "#1b2023",
    edge: "#101315",
    grid: "rgba(220, 235, 230, 0.035)",
  },
  black: {
    center: "#101312",
    middle: "#050606",
    edge: "#000000",
    grid: "rgba(220, 235, 230, 0.025)",
  },
  white: {
    center: "#ffffff",
    middle: "#f3f6f4",
    edge: "#dfe6e2",
    grid: "rgba(28, 73, 57, 0.055)",
  },
};

const fragmentPalettes = {
  accessible: ['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00','#56b4e9','#f0e442','#333333'],
  muted: ['#4e79a7','#e15759','#59a14f','#b07aa1','#f28e2b','#76b7b2','#af9b42','#9c755f'],
  'blue-red': ['#2166ac','#b2182b','#67a9cf','#ef8a62','#053061','#67001f'],
};

function segmentColor(index) {
  const scheme = elements.fragmentPalette.value;
  if (scheme === 'single') return elements.fragmentColor.value;
  const palette = fragmentPalettes[scheme];
  if (palette) return palette[index % palette.length];
  return `hsl(${(index * 137.507764 + 28) % 360}, 76%, 58%)`;
}

function atomTypeColor(type, alpha = 1) {
  const [red, green, blue] = atomTypeRGB[type] || atomTypeRGB[13];
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function validHexColor(value) {
  return /^#[0-9a-f]{6}$/i.test(value || "");
}

function hexRGB(value) {
  const color = validHexColor(value) ? value.slice(1) : "172b28";
  return [
    Number.parseInt(color.slice(0, 2), 16),
    Number.parseInt(color.slice(2, 4), 16),
    Number.parseInt(color.slice(4, 6), 16),
  ];
}

function colorLuminance(value) {
  const channels = hexRGB(value).map((channel) => {
    const scaled = channel / 255;
    return scaled <= 0.04045
      ? scaled / 12.92
      : ((scaled + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] +
    0.7152 * channels[1] +
    0.0722 * channels[2];
}

function activeBackground() {
  const preset = elements.backgroundPreset.value;
  if (preset === "custom") {
    const color = validHexColor(elements.backgroundColor.value)
      ? elements.backgroundColor.value
      : "#172b28";
    const light = colorLuminance(color) > 0.48;
    return {
      center: color,
      middle: color,
      edge: color,
      grid: light
        ? "rgba(20, 45, 37, 0.06)"
        : "rgba(220, 235, 230, 0.035)",
    };
  }
  return backgroundPresets[preset] || backgroundPresets["deep-green"];
}

function lightBackground() {
  return colorLuminance(activeBackground().center) > 0.48;
}

function saveBackgroundPreference() {
  const preference = {
    preset: elements.backgroundPreset.value,
    color: elements.backgroundColor.value,
  };
  try {
    window.localStorage.setItem(
      backgroundStorageKey,
      JSON.stringify(preference),
    );
  } catch (error) {
    console.warn("could not persist background preference", error);
  }
  const url = new URL(window.location.href);
  url.searchParams.set("background", preference.preset);
  if (preference.preset === "custom") {
    url.searchParams.set("backgroundColor", preference.color);
  } else {
    url.searchParams.delete("backgroundColor");
  }
  window.history.replaceState({}, "", url);
  elements.stage.classList.toggle("light-background", lightBackground());
}

function loadBackgroundPreference() {
  const query = new URLSearchParams(window.location.search);
  let preference = null;
  try {
    preference = JSON.parse(
      window.localStorage.getItem(backgroundStorageKey) || "null",
    );
  } catch (error) {
    console.warn("could not read background preference", error);
  }
  const requestedPreset = query.get("background") || preference?.preset;
  if (
    requestedPreset === "custom" ||
    Object.hasOwn(backgroundPresets, requestedPreset)
  ) {
    elements.backgroundPreset.value = requestedPreset;
  }
  const requestedColor = query.get("backgroundColor") || preference?.color;
  if (validHexColor(requestedColor)) {
    elements.backgroundColor.value = requestedColor;
  }
  elements.stage.classList.toggle("light-background", lightBackground());
}

function resizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const targetWidth = Math.round(width * ratio);
  const targetHeight = Math.round(height * ratio);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    renderer.setPixelRatio(ratio);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

function positionOffset(frame, atom) {
  return (frame * state.metadata.atom_count + atom) * 3;
}

function atomTypeAt(frame, atom) {
  if (state.atomLabels) {
    return state.atomLabels[frame * state.metadata.atom_count + atom];
  }
  return state.metadata.atom_type[atom];
}

function currentAtomType(atom) {
  return atomTypeAt(state.frame, atom);
}

function molecularBondPairs(meta, frame) {
  const candidates = [];
  for (let left = 0; left < meta.atom_count - 1; left += 1) {
    const leftType = atomTypeAt(frame, left);
    if (!maximumValence[leftType]) continue;
    const leftOffset = positionOffset(frame, left);
    for (let right = left + 1; right < meta.atom_count; right += 1) {
      const rightType = atomTypeAt(frame, right);
      if (!maximumValence[rightType]) continue;
      const rightOffset = positionOffset(frame, right);
      const dx = state.positions[leftOffset] - state.positions[rightOffset];
      const dy = state.positions[leftOffset + 1] -
        state.positions[rightOffset + 1];
      const dz = state.positions[leftOffset + 2] -
        state.positions[rightOffset + 2];
      const distance = Math.hypot(dx, dy, dz);
      const radiusSum = covalentRadii[leftType] + covalentRadii[rightType];
      if (distance >= 0.4 && distance <= 1.25 * radiusSum) {
        candidates.push({
          left,
          right,
          distance,
          score: distance / radiusSum,
        });
      }
    }
  }
  candidates.sort((a, b) => a.score - b.score || a.distance - b.distance);
  const degree = new Uint8Array(meta.atom_count);
  const bonds = [];
  for (const candidate of candidates) {
    const leftType = atomTypeAt(frame, candidate.left);
    const rightType = atomTypeAt(frame, candidate.right);
    if (
      degree[candidate.left] >= maximumValence[leftType] ||
      degree[candidate.right] >= maximumValence[rightType]
    ) {
      continue;
    }
    degree[candidate.left] += 1;
    degree[candidate.right] += 1;
    bonds.push([candidate.left, candidate.right]);
  }
  return bonds;
}

function threeColorFromRGB(channels) {
  return new THREE.Color().setRGB(
    channels[0] / 255,
    channels[1] / 255,
    channels[2] / 255,
    THREE.SRGBColorSpace,
  );
}

function fragmentThreeColor(fragment, kind = 0) {
  return new THREE.Color().setStyle(segmentColor(fragment));
}

function standardMaterial(key, color, options = {}) {
  if (!materialCache.has(key)) {
    materialCache.set(
      key,
      new THREE.MeshStandardMaterial({
        color,
        roughness: options.roughness ?? 0.3,
        metalness: options.metalness ?? 0.03,
      }),
    );
  }
  return materialCache.get(key);
}

function atomMaterial(type) {
  return standardMaterial(
    `atom-${type}`,
    threeColorFromRGB(atomTypeRGB[type] || atomTypeRGB[13]),
    { roughness: 0.24, metalness: 0.025 },
  );
}

function fragmentMaterial(fragment, kind, use) {
  return standardMaterial(
    `${use}-fragment-${fragment}-${kind}-${segmentColor(fragment)}`,
    fragmentThreeColor(fragment, kind),
    {
      roughness: use === "stick" ? 0.32 : 0.26,
      metalness: use === "stick" ? 0.04 : 0.025,
    },
  );
}

function resetStructureMeshes(metadata) {
  structureGroup.clear();
  state.atomMeshes = [];
  state.bondMeshes = [];
  state.centroidMeshes = [];
  state.meshMetadata = metadata;
}

function ensureAtomMeshes(count) {
  while (state.atomMeshes.length < count) {
    const mesh = new THREE.Mesh(sphereGeometry, atomMaterial(13));
    mesh.userData.entity = "atom";
    structureGroup.add(mesh);
    state.atomMeshes.push(mesh);
  }
}

function ensureBondMesh(index) {
  while (state.bondMeshes.length <= index) {
    const mesh = new THREE.Mesh(
      cylinderGeometry,
      fragmentMaterial(0, 0, "stick"),
    );
    mesh.userData.entity = "bond";
    structureGroup.add(mesh);
    state.bondMeshes.push(mesh);
  }
  return state.bondMeshes[index];
}

function ensureCentroidMeshes(count) {
  while (state.centroidMeshes.length < count) {
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.22,
      metalness: 0.01,
      transmission: 0.08,
      transparent: true,
      opacity: geomCentroidOpacity,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(sphereGeometry, material);
    mesh.userData.entity = "centroid";
    mesh.renderOrder = 1;
    structureGroup.add(mesh);
    state.centroidMeshes.push(mesh);
  }
}

const cylinderAxis = new THREE.Vector3(0, 1, 0);
const cylinderDirection = new THREE.Vector3();
const cylinderMidpoint = new THREE.Vector3();

function placeCylinder(mesh, start, stop, radius, material) {
  cylinderDirection.subVectors(stop, start);
  const length = cylinderDirection.length();
  if (length < 1e-6) {
    mesh.visible = false;
    return;
  }
  cylinderMidpoint.addVectors(start, stop).multiplyScalar(0.5);
  mesh.position.copy(cylinderMidpoint);
  mesh.quaternion.setFromUnitVectors(
    cylinderAxis,
    cylinderDirection.normalize(),
  );
  mesh.scale.set(radius, length, radius);
  mesh.material = material;
  mesh.visible = true;
}

function frameCoordinates(meta) {
  const coordinates = new Array(meta.atom_count);
  const center = new THREE.Vector3();
  for (let atom = 0; atom < meta.atom_count; atom += 1) {
    const offset = positionOffset(state.frame, atom);
    const position = new THREE.Vector3(
      state.positions[offset],
      state.positions[offset + 1],
      state.positions[offset + 2],
    );
    coordinates[atom] = position;
    center.add(position);
  }
  center.multiplyScalar(1 / Math.max(meta.atom_count, 1));
  for (const position of coordinates) position.sub(center);
  return coordinates;
}

function rebuildStructureScene() {
  const meta = state.metadata;
  if (!meta) return;
  if (state.meshMetadata !== meta) resetStructureMeshes(meta);
  const atomTypeMode = elements.atomColorMode.checked;
  const coordinates = frameCoordinates(meta);
  const sizeSetting = Number(elements.pointSize.value);
  const atomRadius = sizeSetting * 0.13;
  const stickRadius = atomRadius * 0.31 * Number(elements.bondSize.value);

  ensureAtomMeshes(meta.atom_count);
  for (let atom = 0; atom < meta.atom_count; atom += 1) {
    const mesh = state.atomMeshes[atom];
    const fragment = meta.segment_id[atom];
    const kind = 0;
    const atomType = currentAtomType(atom);
    const typeScale = moleculeScale[atomType] || 1;
    mesh.position.copy(coordinates[atom]);
    mesh.scale.setScalar(atomRadius * typeScale);
    mesh.material = atomTypeMode
      ? atomMaterial(atomType)
      : fragmentMaterial(fragment, kind, "atom");
    mesh.userData.atom = atom;
    mesh.visible = true;
  }

  const bondPairs = state.frame === meta.frame_count - 1
    ? state.finalBondPairs : state.withinFragmentBondPairs;
  let usedBondMeshes = 0;
  const midpoint = new THREE.Vector3();
  for (const [leftAtom, rightAtom] of (elements.showBonds.checked ? bondPairs : [])) {
    const leftFragment = meta.segment_id[leftAtom];
    const rightFragment = meta.segment_id[rightAtom];
    const leftKind = 0;
    const rightKind = 0;
    if (leftFragment === rightFragment) {
      placeCylinder(
        ensureBondMesh(usedBondMeshes),
        coordinates[leftAtom],
        coordinates[rightAtom],
        stickRadius,
        fragmentMaterial(leftFragment, leftKind, "stick"),
      );
      usedBondMeshes += 1;
    } else {
      midpoint.addVectors(
        coordinates[leftAtom],
        coordinates[rightAtom],
      ).multiplyScalar(0.5);
      placeCylinder(
        ensureBondMesh(usedBondMeshes),
        coordinates[leftAtom],
        midpoint,
        stickRadius,
        fragmentMaterial(leftFragment, leftKind, "stick"),
      );
      usedBondMeshes += 1;
      placeCylinder(
        ensureBondMesh(usedBondMeshes),
        midpoint,
        coordinates[rightAtom],
        stickRadius,
        fragmentMaterial(rightFragment, rightKind, "stick"),
      );
      usedBondMeshes += 1;
    }
  }
  for (
    let index = usedBondMeshes;
    index < state.bondMeshes.length;
    index += 1
  ) {
    state.bondMeshes[index].visible = false;
  }

  ensureCentroidMeshes(meta.segment_count);
  const frameProgress = meta.frame_count > 1
    ? state.frame / (meta.frame_count - 1)
    : 1;
  const centroidFade = Math.max(0, Math.min(
    1,
    (1 - frameProgress) / (1 - geomCentroidFadeStart),
  ));
  const centroidSums = Array.from(
    { length: meta.segment_count },
    () => ({ position: new THREE.Vector3(), count: 0 }),
  );
  for (let atom = 0; atom < meta.atom_count; atom += 1) {
    const fragment = meta.segment_id[atom];
    centroidSums[fragment].position.add(coordinates[atom]);
    centroidSums[fragment].count += 1;
  }
  for (let fragment = 0; fragment < meta.segment_count; fragment += 1) {
    const mesh = state.centroidMeshes[fragment];
    const sum = centroidSums[fragment];
    if (
      !elements.centerMode.checked ||
      centroidFade <= 0 ||
      !sum.count
    ) {
      mesh.visible = false;
      continue;
    }
    mesh.position.copy(sum.position).multiplyScalar(1 / sum.count);
    mesh.scale.setScalar(atomRadius * geomCentroidRadiusScale);
    mesh.material.color.copy(fragmentThreeColor(fragment, 0));
    mesh.material.opacity = geomCentroidOpacity * centroidFade;
    mesh.visible = true;
  }

  state.displayRadius = elements.autoFit.checked &&
    meta.frame_display_radius
    ? meta.frame_display_radius[state.frame]
    : meta.display_radius;
  elements.bondCount.textContent = (elements.showBonds.checked ? bondPairs.length : 0).toLocaleString();
  state.sceneDirty = false;
}

function drawScene() {
  resizeCanvas();
  if (figureView.enabled && state.metadata) {
    figureView.render(canvas.width, canvas.height); return;
  }
  scene.background = new THREE.Color(activeBackground().center);
  if (state.metadata && state.sceneDirty) rebuildStructureScene();
  structureGroup.quaternion.copy(state.orientation);
  if (state.metadata) {
    const halfVerticalFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const verticalDistance =
      state.displayRadius / Math.tan(halfVerticalFov);
    const horizontalDistance = verticalDistance /
      Math.max(camera.aspect, 0.1);
    const distance = Math.max(verticalDistance, horizontalDistance) *
      1.08 / state.zoom;
    camera.position.set(0, 0, Math.max(distance, 0.1));
    camera.near = Math.max(0.005, distance - state.displayRadius * 2.5);
    camera.far = Math.max(
      camera.near + 1,
      distance + state.displayRadius * 3.5,
    );
    camera.updateProjectionMatrix();
  }
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
}

function setFrame(frame) {
  if (!state.metadata) return;
  state.frame = Math.max(0, Math.min(state.metadata.frame_count - 1, frame));
  state.sceneDirty = true;
  elements.timeline.value = String(state.frame);
  elements.frameValue.textContent = `${state.frame} / ${state.metadata.frame_count - 1}`;
  elements.timeValue.textContent = state.metadata.times[state.frame].toFixed(3);
}

function setPlaying(playing) {
  state.playing = playing;
  elements.playIcon.textContent = playing ? "Ⅱ" : "▶";
  elements.play.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function supportedVideoType() {
  if (typeof window.MediaRecorder === "undefined") return "";
  const candidates = [
    "video/mp4;codecs=avc1.42001E",
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4;codecs=avc1.4D401F",
    "video/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function videoFilename() {
  const source = state.metadata?.source_name || "trajectory";
  const slug = source.toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64) || "trajectory";
  return `lpgm-${slug}-viewport.mp4`;
}

function lockRecordingControls(locked) {
  elements.play.disabled = locked;
  elements.timeline.disabled = locked;
  elements.speed.disabled = locked;
  elements.trajectorySelect.disabled = locked;
  if (locked) elements.pairedSample.disabled = true;
  else refreshPairedSample();
  elements.endHold.disabled = locked;
  elements.videoBitrate.disabled = locked;
}

function updateRecordingProgress() {
  if (!state.recording || !state.metadata) return;
  if (state.recording.holding) {
    const tenths = Math.max(
      0,
      Math.ceil(state.recording.holdRemaining * 10),
    );
    if (tenths === state.recording.lastHoldTenths) return;
    state.recording.lastHoldTenths = tenths;
    elements.videoStatus.textContent =
      `Resting at t = 1 · ${(tenths / 10).toFixed(1)} s remaining.`;
    return;
  }
  const denominator = Math.max(state.metadata.frame_count - 1, 1);
  const percent = Math.min(
    100,
    Math.round(100 * state.frame / denominator),
  );
  if (percent === state.recording.lastPercent) return;
  state.recording.lastPercent = percent;
  elements.videoStatus.textContent =
    `Recording viewport · ${percent}% · click to finish early.`;
}

function releaseRecording(recording) {
  for (const track of recording.stream.getTracks()) {
    track.stop();
  }
  lockRecordingControls(false);
  elements.saveVideo.classList.remove("recording");
  elements.saveVideoLabel.textContent = "Save video";
  state.recording = null;
}

function finishVideoRecording() {
  const recording = state.recording;
  if (!recording || recording.finishing) return;
  recording.finishing = true;
  setPlaying(false);
  elements.saveVideoLabel.textContent = "Encoding…";
  elements.videoStatus.textContent = "Finishing viewport video…";
  if (recording.recorder.state !== "inactive") {
    recording.recorder.stop();
  }
}

function startVideoRecording() {
  if (!state.metadata) {
    elements.videoStatus.textContent =
      "Load a trajectory before recording.";
    return;
  }
  if (
    typeof canvas.captureStream !== "function" ||
    typeof window.MediaRecorder === "undefined"
  ) {
    elements.videoStatus.textContent =
      "Viewport recording is not supported by this browser.";
    elements.saveVideo.disabled = true;
    return;
  }
  const mimeType = supportedVideoType();
  if (!mimeType) {
    elements.videoStatus.textContent =
      "This browser cannot encode Canvas recordings as MP4.";
    elements.saveVideo.disabled = true;
    return;
  }
  const stream = canvas.captureStream(60);
  const options = {
    mimeType,
    videoBitsPerSecond: Number(elements.videoBitrate.value) * 1_000_000,
  };
  let recorder;
  try {
    recorder = new MediaRecorder(stream, options);
  } catch (error) {
    for (const track of stream.getTracks()) track.stop();
    console.error(error);
    elements.videoStatus.textContent =
      `Could not start video export: ${error.message}`;
    return;
  }
  const recording = {
    recorder,
    stream,
    chunks: [],
    finishing: false,
    lastPercent: -1,
    holding: false,
    holdRemaining: Number(elements.endHold.value),
    endHoldSeconds: Number(elements.endHold.value),
    lastHoldTenths: -1,
  };
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) recording.chunks.push(event.data);
  });
  recorder.addEventListener("error", (event) => {
    console.error("viewport recording failed", event.error);
    elements.videoStatus.textContent =
      `Video export failed: ${event.error?.message || "unknown error"}`;
    releaseRecording(recording);
  }, { once: true });
  recorder.addEventListener("stop", () => {
    const outputType = recorder.mimeType || mimeType || "video/webm";
    const blob = new Blob(recording.chunks, { type: outputType });
    if (blob.size) {
      const url = URL.createObjectURL(blob);
      const download = document.createElement("a");
      download.href = url;
      download.download = videoFilename();
      document.body.append(download);
      download.click();
      download.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      elements.videoStatus.textContent =
        `Saved viewport-only MP4 at ${elements.videoBitrate.value} Mbps.`;
    } else {
      elements.videoStatus.textContent =
        "The browser returned an empty recording.";
    }
    releaseRecording(recording);
  }, { once: true });

  state.recording = recording;
  setFrame(0);
  state.frameAccumulator = 0;
  state.lastTick = performance.now();
  drawScene();
  lockRecordingControls(true);
  elements.saveVideo.classList.add("recording");
  elements.saveVideoLabel.textContent = "Stop & save";
  recorder.start(250);
  setPlaying(true);
  updateRecordingProgress();
}

function toggleVideoRecording() {
  if (state.recording) {
    finishVideoRecording();
  } else {
    startVideoRecording();
  }
}

function updateHover() {
  if (!state.metadata || state.pointerX < 0) {
    elements.hover.hidden = true;
    return;
  }
  const rect = canvas.getBoundingClientRect();
  pointerNDC.set(
    (state.pointerX / rect.width) * 2 - 1,
    1 - (state.pointerY / rect.height) * 2,
  );
  raycaster.setFromCamera(pointerNDC, camera);
  const hit = raycaster.intersectObjects(
    state.atomMeshes.filter((mesh) => mesh.visible),
    false,
  )[0];
  if (!hit) {
    elements.hover.hidden = true;
    return;
  }
  const meta = state.metadata;
  const atom = hit.object.userData.atom;
  const segment = meta.segment_id[atom];
  const kind = 0;
  const atomType = currentAtomType(atom);
  const atomLabel = meta.atom_label_names?.[atomType] || `class ${atomType}`;
  elements.hover.innerHTML = `<strong>fragment ${segment + 1}</strong><br>` +
    `atom ${meta.atom_index[atom] + 1} · ${atomLabel}`;
  elements.hover.style.left =
    `${Math.min(state.pointerX + 13, canvas.clientWidth - 150)}px`;
  elements.hover.style.top = `${Math.max(state.pointerY - 38, 8)}px`;
  elements.hover.hidden = false;
}

function animate(now) {
  const elapsed = Math.min((now - state.lastTick) / 1000, 0.1);
  state.lastTick = now;
  let completedRecording = false;
  if (state.recording?.holding) {
    state.recording.holdRemaining -= elapsed;
    if (state.recording.holdRemaining <= 0) {
      completedRecording = true;
    }
  } else if (state.metadata && state.playing) {
    state.frameAccumulator += elapsed * Number(elements.speed.value);
    while (state.frameAccumulator >= 1) {
      state.frameAccumulator -= 1;
      const next = state.frame + 1;
      if (state.recording && next >= state.metadata.frame_count) {
        state.frameAccumulator = 0;
        if (state.recording.endHoldSeconds > 0) {
          state.recording.holding = true;
          state.recording.holdRemaining =
            state.recording.endHoldSeconds;
        } else {
          completedRecording = true;
        }
        break;
      }
      setFrame(next >= state.metadata.frame_count ? 0 : next);
    }
  }
  if (state.metadata && elements.autoRotate.checked && !state.dragging) {
    cameraUp.copy(localUp).applyQuaternion(camera.quaternion);
    rotationDelta.setFromAxisAngle(cameraUp, elapsed * 0.12);
    state.orientation.premultiply(rotationDelta).normalize();
  }
  drawScene();
  if (figureView.enabled) elements.hover.hidden = true; else updateHover();
  updateRecordingProgress();
  if (completedRecording) {
    finishVideoRecording();
  }
  requestAnimationFrame(animate);
}

function processLabel(kind, power) {
  if (kind === "smoothstep") return "cubic";
  if (kind === "smoothstep_power") return `cubic×${Number(power).toFixed(0)}`;
  if (kind === "smootherstep") return "quintic";
  if (kind === "smootherstep_power") {
    return `quintic×${Number(power).toFixed(0)}`;
  }
  return `p${Number(power).toFixed(0)}`;
}

function appendColorKey(container, color, label) {
  const item = document.createElement("span");
  const swatch = document.createElement("i");
  swatch.style.background = color;
  item.append(swatch, document.createTextNode(label));
  container.append(item);
}

function updateColorPresentation(metadata) {
  const atomMode = elements.atomColorMode.checked;
  elements.atomColorControl.hidden = false;
  elements.centerControl.hidden = false;
  elements.segmentKey.hidden = true;
  elements.fragmentKey.hidden = false;
  elements.atomKey.hidden = !atomMode;
  elements.fragmentKey.replaceChildren();
  elements.atomKey.replaceChildren();

  const shown = Math.min(metadata.segment_count, 12);
  for (let fragment = 0; fragment < shown; fragment += 1) {
    appendColorKey(
      elements.fragmentKey,
      segmentColor(fragment, 0),
      `F${fragment + 1}`,
    );
  }
  if (atomMode) {
    elements.colorLabel.textContent = "Atom and fragment color";
    const presentTypes = [...new Set(metadata.atom_type)].sort((a, b) => a - b);
    for (const type of presentTypes) {
      const label = metadata.atom_label_names?.[type] || `class ${type}`;
      appendColorKey(elements.atomKey, atomTypeColor(type), label);
    }
    elements.colorNote.textContent =
      "Atoms use CPK-like sampled-label colors. Bonds and translucent centers use the selected fragment palette; centers fade over the final 5% of the path.";
  } else {
    elements.colorLabel.textContent = "Fragment color";
    elements.colorNote.textContent =
      elements.fragmentPalette.value === 'single'
        ? "Atoms, bonds and fragment centers use the selected single color."
        : "The selected palette identifies BRICS fragments across atoms, bonds and centers. Centers fade over the final 5% of the path.";
  }
}

function updatePointSizeOutput() {
  const scale = Number(elements.pointSize.value) / 2.8;
  elements.sizeOutput.textContent = `${scale.toFixed(2)}×`;
}

function configureMetadata(metadata) {
  const process = metadata.process || {};
  const bm = process.coordinate_process === "atom-brownian-kumaraswamy-v1";
  const format = (v, units = "") => Number.isFinite(Number(v)) ? `${Number(v).toFixed(2)}${units}` : "—";
  elements.sourceName.textContent = metadata.source_name;
  elements.atomCount.textContent = metadata.atom_count.toLocaleString();
  elements.segmentCount.textContent = metadata.segment_count.toLocaleString();
  elements.entityCountLabel.textContent = "Fragments";
  elements.checkpoint.textContent = metadata.mode === "conditional_bridge"
    ? "Conditional training process"
    : `${metadata.checkpoint?.split("/").pop() || "Model"} · ${metadata.checkpoint_weights || "raw"}`;
  elements.seed.textContent = `seed ${metadata.seed} · ${metadata.inference_time_schedule} time grid`;
  updateColorPresentation(metadata);
  updatePointSizeOutput();
  elements.bondNote.hidden = false;
  elements.modeLabel.textContent = `${metadata.domain === "qm9" ? "QM9" : "GEOM-Drugs"} · ${bm ? "Brownian Atoms" : "Fragment Noise"}`;
  const conditional = metadata.mode === "conditional_bridge";
  elements.trajectoryTitle.textContent = conditional ? "conditional molecular bridge" : "molecular sampling trajectory";
  elements.sampleLabel.textContent = conditional ? "Data endpoint" : "Generated molecule";
  elements.processLabel.textContent = conditional ? "Conditional process" : "Sampling process";
  elements.sourceSigma.textContent = format(process.source_sigma, " Å");
  elements.centerVariance.textContent = bm ? "—" : format(process.center_variance, " Å²");
  elements.clockPowers.textContent = bm ? `a = ${process.atom_bm_clock_a}, b = ${process.atom_bm_clock_b}` : `quintic / quintic / quintic×${process.internal_clock_power}`;
  elements.internalVarianceLabel.textContent = bm ? "Brownian β" : "Internal variance";
  elements.internalVariance.textContent = format(bm ? process.atom_bm_variance : process.internal_variance, " Å²");
  elements.rotationSigmaLabel.textContent = "Angular variance ρ²";
  elements.rotationSigma.textContent = bm ? "—" : format(process.rotation_variance, " rad²");
  elements.processSection.hidden = !Object.keys(process).length;
}

async function loadCatalog() {
  try {
    const query = new URLSearchParams(window.location.search);
    if (query.has("trajectory") && !query.has("catalog")) return null;
    const catalogUrl = new URL(
      query.get("catalog") || "trajectories.json",
      window.location.href,
    );
    const response = await fetch(catalogUrl, { cache: "no-store" });
    if (!response.ok) return null;
    const catalog = await response.json();
    if (catalog.format_version !== 1 || !Array.isArray(catalog.trajectories)) {
      throw new Error("unsupported trajectory catalog");
    }
    state.catalogUrl = catalogUrl;
    return catalog;
  } catch (error) {
    console.warn("trajectory catalog unavailable", error);
    return null;
  }
}

async function loadQuality() {
  state.quality = null;
  if (state.catalog.quality_file) {
    try {
      const response = await fetch(new URL(state.catalog.quality_file, state.catalogUrl), {cache: 'no-store'});
      if (!response.ok) throw new Error('Quality data could not be loaded');
      const quality = await response.json();
      if (quality.format_version !== 1 || !quality.samples ||
          state.catalog.trajectories.some(item => !quality.samples[item.id])) throw new Error('Quality data is incomplete');
      state.quality = quality;
    } catch (error) {console.warn(error);}
  }
  const available = !!state.quality;
  const requested = new URL(location.href).searchParams.get('quality') ?? state.catalog.default_quality ?? 'valid_connected';
  document.querySelector('#qualityFilterLabel').textContent = state.catalog.quality_scope === 'paired'
    ? 'Only retain pairs where both pass' : 'Only retain passing samples';
  const wantsFilter = ['valid','connected','valid_connected'].includes(requested);
  elements.qualityFilter.disabled = !available && !wantsFilter;
  elements.qualityCriterion.disabled = !available;
  if (['valid','connected','valid_connected'].includes(requested)) {
    elements.qualityFilter.checked = true; elements.qualityCriterion.value = requested;
  }
  if (!available && !wantsFilter) elements.qualityFilter.checked = false;
}

function refreshTrajectoryOptions() {
  state.visibleTrajectories = filterTrajectories(state.catalog.trajectories, state.quality,
    elements.qualityFilter.checked, elements.qualityCriterion.value, state.catalog.quality_scope === 'paired');
  const old = elements.trajectorySelect.value;
  elements.trajectorySelect.replaceChildren();
  for (const item of state.visibleTrajectories) {
    const option = document.createElement('option'); option.value = item.id; option.textContent = item.label;
    elements.trajectorySelect.append(option);
  }
  if (state.visibleTrajectories.some(item => item.id === old)) elements.trajectorySelect.value = old;
  elements.trajectoryPicker.hidden = false;
  elements.qualityStatus.textContent = state.quality
    ? `${state.visibleTrajectories.length.toLocaleString()} / ${state.catalog.trajectories.length.toLocaleString()} samples shown · evaluated final structures`
    : elements.qualityFilter.checked ? 'Quality evaluation unavailable; no samples can pass the filter.'
    : 'Quality evaluation unavailable for this panel; all samples are shown.';
  if (state.quality && elements.qualityFilter.checked && state.catalog.quality_scope === 'paired') {
    elements.qualityStatus.textContent = `${(state.visibleTrajectories.length / 2).toLocaleString()} / ${(state.catalog.trajectories.length / 2).toLocaleString()} matched pairs shown · both models pass`;
  }
  refreshPairedSample();
}

function refreshPairedSample() {
  const current = state.catalog?.trajectories.find(item => item.id === elements.trajectorySelect.value);
  const paired = state.catalog?.trajectories.find(item => item.id === current?.paired_id);
  elements.pairedSample.hidden = !paired;
  elements.pairedSampleNote.hidden = !paired;
  if (!paired) return;
  const visible = state.visibleTrajectories.some(item => item.id === paired.id);
  elements.pairedSample.textContent = `Show ${paired.model_label} · same sample`;
  elements.pairedSample.disabled = !visible;
  elements.pairedSampleNote.textContent = visible
    ? 'Matched conditioning and sample number. Keeps the current time, orientation and figure settings.'
    : 'The other model’s sample does not pass this filter. Disable the filter to compare it.';
}

elements.pairedSample.addEventListener('click', () => {
  const current = state.catalog?.trajectories.find(item => item.id === elements.trajectorySelect.value);
  if (!state.visibleTrajectories.some(item => item.id === current?.paired_id)) return;
  const url = new URL(location.href);
  url.searchParams.set('sample', current.paired_id);
  url.searchParams.set('frame', String(state.frame));
  url.searchParams.set('autoplay', '0');
  history.replaceState({}, '', url);
  loadTrajectory(current.paired_id);
});

function applyQualityFilter() {
  if (!state.catalog) return;
  const current = elements.trajectorySelect.value;
  refreshTrajectoryOptions();
  const url = new URL(location.href);
  if (elements.qualityFilter.checked) url.searchParams.set('quality', elements.qualityCriterion.value);
  else if (state.catalog.default_quality) url.searchParams.set('quality', 'all');
  else url.searchParams.delete('quality');
  const selected = state.visibleTrajectories.some(item => item.id === current) ? current : state.visibleTrajectories[0]?.id;
  if (selected) url.searchParams.set('sample', selected);
  history.replaceState({}, '', url);
  loadTrajectory(selected);
}
elements.qualityFilter.addEventListener('change', applyQualityFilter);
elements.qualityCriterion.addEventListener('change', applyQualityFilter);

async function loadTrajectory(requestedSample = null) {
  const serial = ++state.loadSerial;
  try {
    const query = new URLSearchParams(window.location.search);
    if (!state.catalog) {
      state.catalog = await loadCatalog();
      if (state.catalog) {
        await loadQuality();
        refreshTrajectoryOptions();
      }
    }
    const choices = state.catalog ? state.visibleTrajectories : [];
    if (state.catalog && !choices.length) {
      state.metadata = null; state.positions = null; state.atomLabels = null;
      structureGroup.clear(); state.meshMetadata = null;
      const overlay = document.getElementById('figureLabelCanvas');
      overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
      elements.status.textContent = 'No samples match this filter';
      elements.status.classList.remove('ready');
      document.getElementById('saveFigure').disabled = true;
      setPlaying(false); return;
    }
    const sampleId = requestedSample || query.get("sample") || state.catalog?.default;
    const selected = choices.find(item => item.id === sampleId) || choices[0];
    const metadataUrl = new URL(
      selected?.trajectory || query.get("trajectory") || "trajectory.json",
      selected && state.catalogUrl ? state.catalogUrl : window.location.href,
    );
    if (selected) elements.trajectorySelect.value = selected.id;
    refreshPairedSample();
    elements.status.classList.remove("ready", "error");
    elements.status.innerHTML = "<span></span> loading trajectory";
    const metadataResponse = await fetch(metadataUrl, { cache: "no-store" });
    if (!metadataResponse.ok) throw new Error(`metadata HTTP ${metadataResponse.status}`);
    const metadata = await metadataResponse.json();
    if (metadata.format_version !== 1) throw new Error("unsupported trajectory format");
    const positionResponse = await fetch(
      new URL(metadata.positions_file, metadataUrl),
      { cache: "no-store" },
    );
    if (!positionResponse.ok) throw new Error(`positions HTTP ${positionResponse.status}`);
    const buffer = await positionResponse.arrayBuffer();
    const positions = new Float32Array(buffer);
    const expected = metadata.frame_count * metadata.atom_count * 3;
    if (positions.length !== expected) {
      throw new Error(`positions contain ${positions.length} floats; expected ${expected}`);
    }
    let atomLabels = null;
    if (metadata.atom_labels_file) {
      const labelResponse = await fetch(
        new URL(metadata.atom_labels_file, metadataUrl),
        { cache: "no-store" },
      );
      if (!labelResponse.ok) {
        throw new Error(`atom labels HTTP ${labelResponse.status}`);
      }
      atomLabels = new Uint8Array(await labelResponse.arrayBuffer());
      const expectedLabels = metadata.frame_count * metadata.atom_count;
      if (atomLabels.length !== expectedLabels) {
        throw new Error(
          `atom labels contain ${atomLabels.length} values; expected ${expectedLabels}`
        );
      }
    }
    if (serial !== state.loadSerial) return;
    state.metadata = metadata;
    state.positions = positions;
    state.atomLabels = atomLabels;
    // Infer topology once from the actual X1 coordinates and element labels.
    // Keep intra-fragment bonds throughout; reveal cross-fragment bonds at X1.
    state.finalBondPairs = true
      ? molecularBondPairs(metadata, metadata.frame_count - 1)
      : [];
    state.withinFragmentBondPairs = state.finalBondPairs.filter(
      ([left, right]) => metadata.segment_id[left] === metadata.segment_id[right],
    );
    state.frameAccumulator = 0;
    state.sceneDirty = true;
    elements.timeline.max = String(metadata.frame_count - 1);
    if (query.has("color")) {
      elements.atomColorMode.checked = query.get("color") !== "fragment";
    }
    if (query.has("centers")) {
      elements.centerMode.checked = query.get("centers") !== "0";
    }
    configureMetadata(metadata);
    if (selected?.model_label) {
      elements.modeLabel.textContent = metadata.mode === "conditional_bridge"
        ? `${selected.model_label} · conditional bridge`
        : `${selected.model_label} · ${metadata.checkpoint_step / 1000}K EMA`;
    }
    elements.status.classList.add("ready");
    elements.status.innerHTML = "<span></span> sample ready";
    const requestedFrame = Number.parseInt(query.get("frame") || "0", 10);
    setFrame(Number.isFinite(requestedFrame) ? requestedFrame : 0);
    setPlaying(!figureView.enabled && query.get("autoplay") !== "0");
    if (figureView.enabled) figureView.refresh();
    if (embedded) {
      elements.autoRotate.checked = false;
      setPlaying(false);
      notifyParent("lpgm-ready", { sample: selected?.id, frames: metadata.frame_count });
    }
  } catch (error) {
    if (serial !== state.loadSerial) return;
    console.error(error);
    elements.status.classList.add("error");
    elements.status.innerHTML = `<span></span> ${error.message}`;
    setPlaying(false);
    notifyParent("lpgm-error", { message: error.message });
  }
}

elements.play.addEventListener("click", () => setPlaying(!state.playing));
elements.timeline.addEventListener("input", () => {
  setFrame(Number(elements.timeline.value));
  setPlaying(false);
});
elements.speed.addEventListener("input", () => {
  elements.speedOutput.textContent = `${elements.speed.value} fps`;
});
elements.pointSize.addEventListener("input", () => {
  updatePointSizeOutput();
  state.sceneDirty = true;
});
elements.atomColorMode.addEventListener("change", () => {
  if (state.metadata) {
    const url = new URL(window.location.href);
    url.searchParams.set(
      "color",
      elements.atomColorMode.checked ? "atom" : "fragment",
    );
    window.history.replaceState({}, "", url);
    updateColorPresentation(state.metadata);
    state.sceneDirty = true;
  }
});
function updateFragmentPalette() {
  elements.fragmentColorControl.hidden = elements.fragmentPalette.value !== 'single';
  const url = new URL(location.href);
  url.searchParams.set('palette', elements.fragmentPalette.value);
  url.searchParams.set('fragmentColor', elements.fragmentColor.value);
  history.replaceState({}, '', url);
  if (state.metadata) updateColorPresentation(state.metadata);
  state.sceneDirty = true;
}
elements.fragmentPalette.addEventListener('change', updateFragmentPalette);
elements.fragmentColor.addEventListener('change', updateFragmentPalette);
elements.centerMode.addEventListener("change", () => {
  if (!state.metadata) return;
  const url = new URL(window.location.href);
  url.searchParams.set(
    "centers",
    elements.centerMode.checked ? "1" : "0",
  );
  window.history.replaceState({}, "", url);
  state.sceneDirty = true;
});
elements.bondSize.addEventListener('input', () => {
  elements.bondSizeOutput.value = `${Number(elements.bondSize.value).toFixed(2)}×`;
  state.sceneDirty = true;
});
elements.showBonds.addEventListener('change', () => {state.sceneDirty = true;});
elements.autoFit.addEventListener("change", () => {
  state.sceneDirty = true;
});
elements.backgroundPreset.addEventListener("change", () => {
  saveBackgroundPreference();
});
elements.backgroundColor.addEventListener("input", () => {
  elements.backgroundPreset.value = "custom";
  saveBackgroundPreference();
});
elements.endHold.addEventListener("input", () => {
  elements.endHoldOutput.textContent =
    `${Number(elements.endHold.value).toFixed(1)} s`;
});
elements.saveVideo.addEventListener("click", toggleVideoRecording);
elements.trajectorySelect.addEventListener("change", () => {
  const url = new URL(window.location.href);
  url.searchParams.set("sample", elements.trajectorySelect.value);
  url.searchParams.delete("trajectory");
  url.searchParams.delete("frame");
  window.history.replaceState({}, "", url);
  loadTrajectory(elements.trajectorySelect.value);
});

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  const rect = canvas.getBoundingClientRect();
  state.pointerX = event.clientX - rect.left;
  state.pointerY = event.clientY - rect.top;
  if (!state.dragging) return;
  cameraUp.copy(localUp).applyQuaternion(camera.quaternion);
  cameraRight.copy(localRight).applyQuaternion(camera.quaternion);
  horizontalRotation.setFromAxisAngle(
    cameraUp,
    (event.clientX - state.lastX) * 0.008,
  );
  verticalRotation.setFromAxisAngle(
    cameraRight,
    (event.clientY - state.lastY) * 0.008,
  );
  rotationDelta.multiplyQuaternions(
    verticalRotation,
    horizontalRotation,
  );
  state.orientation.premultiply(rotationDelta).normalize();
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  notifyParent("lpgm-view", { orientation: state.orientation.toArray(), zoom: state.zoom });
});
canvas.addEventListener("pointerup", (event) => {
  state.dragging = false;
  canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointerleave", () => {
  if (!state.dragging) {
    state.pointerX = -1;
    state.pointerY = -1;
  }
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  state.zoom = Math.max(0.35, Math.min(4,
    state.zoom * Math.exp(-event.deltaY * 0.001)));
  notifyParent("lpgm-view", { orientation: state.orientation.toArray(), zoom: state.zoom });
}, { passive: false });

window.addEventListener("message", (event) => {
  if (!embedded || event.origin !== window.location.origin ||
      event.source !== window.parent || event.data?.type !== "lpgm-controls") return;
  const command = event.data;
  if (typeof command.sample === "string") loadTrajectory(command.sample);
  if (Number.isFinite(command.frame)) setFrame(Math.round(command.frame));
  if (typeof command.atomColors === "boolean") {
    elements.atomColorMode.checked = command.atomColors;
    if (state.metadata) updateColorPresentation(state.metadata);
    state.sceneDirty = true;
  }
  if (typeof command.centers === "boolean") {
    elements.centerMode.checked = command.centers;
    state.sceneDirty = true;
  }
  if (Array.isArray(command.orientation) && command.orientation.length === 4 &&
      command.orientation.every(Number.isFinite)) {
    state.orientation.fromArray(command.orientation).normalize();
  }
  if (Number.isFinite(command.zoom)) state.zoom = Math.max(.35, Math.min(4, command.zoom));
});

window.addEventListener("keydown", (event) => {
  if (event.target.closest('input,select,textarea,button')) return;
  if (figureView.enabled) return;
  if (embedded && ["Space", "ArrowRight", "ArrowLeft"].includes(event.code)) {
    event.preventDefault();
    notifyParent("lpgm-key", { code: event.code });
    return;
  }
  if (
    state.recording &&
    ["Space", "ArrowRight", "ArrowLeft"].includes(event.code)
  ) {
    event.preventDefault();
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.code === "ArrowRight") {
    setFrame(state.frame + 1);
    setPlaying(false);
  } else if (event.code === "ArrowLeft") {
    setFrame(state.frame - 1);
    setPlaying(false);
  }
});
window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", () => {
  if (state.recording) {
    for (const track of state.recording.stream.getTracks()) track.stop();
  }
});

loadBackgroundPreference();
const colorQuery = new URLSearchParams(location.search);
if ([...elements.fragmentPalette.options].some(option => option.value === colorQuery.get('palette')))
  elements.fragmentPalette.value = colorQuery.get('palette');
if (validHexColor(colorQuery.get('fragmentColor')))
  elements.fragmentColor.value = colorQuery.get('fragmentColor');
elements.fragmentColorControl.hidden = elements.fragmentPalette.value !== 'single';
const figureView = createFigureView({renderer, sourceScene: scene, structureGroup, state, elements,
  rebuildStructureScene, setPlaying, activeBackground});
await loadTrajectory();
requestAnimationFrame(animate);
