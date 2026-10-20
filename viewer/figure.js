import * as THREE from './vendor/three.module.min.js';
import {selectFigureFrames} from './figure-utils.mjs';

/** A single orthographic scene: equally scaled molecular snapshots, no panel boxes. */
export function createFigureView({renderer, sourceScene, structureGroup, state, elements,
                                  rebuildStructureScene, setPlaying, activeBackground}) {
  const byId = id => document.getElementById(id);
  const ui = Object.fromEntries(['figureMode','figureControls','figureTimes','figureSpacing',
    'figureSpacingOutput','figureLabels','figureWidth','figureHeight','saveFigure','figureStatus',
    'figureFit','figureReset','figureSettings','figureLabelCanvas'].map(id => [id, byId(id)]));
  const scene = new THREE.Scene();
  for (const child of sourceScene.children) if (child.isLight) scene.add(child.clone());
  const snapshots = new THREE.Group(); scene.add(snapshots);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .01, 10000);
  camera.position.set(0, 0, 1000); camera.lookAt(0, 0, 0);
  let signature = '', selected = [], cacheMeta = null, layout = null, valid = false;
  let oldControls = null, lastNotice = '', exportNoticeUntil = 0, exporting = false;
  const query = new URLSearchParams(location.search);
  for (const [param, id] of [['times','figureTimes'], ['spacing','figureSpacing'],
                            ['width','figureWidth'], ['height','figureHeight']]) {
    if (query.has(param)) ui[id].value = query.get(param);
  }
  ui.figureLabels.checked = query.get('labels') === '1';

  function disposeSnapshots() {
    snapshots.traverse(object => { if (object.isMesh) object.material.dispose(); });
    snapshots.clear();
  }
  function settings() {
    const width = Number(ui.figureWidth.value), height = Number(ui.figureHeight.value);
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 256 || height < 128 ||
        width > 12000 || height > 8000 || width * height > 48e6) {
      throw new Error('Use 256–12,000 px width, 128–8,000 px height, up to 48 megapixels.');
    }
    const spacing = Number(ui.figureSpacing.value);
    if (!Number.isFinite(spacing) || spacing < .4 || spacing > 3) throw new Error('Invalid frame spacing.');
    return {width, height, spacing, labels: ui.figureLabels.checked};
  }
  function remember() {
    const url = new URL(location.href);
    url.searchParams.set('figure', ui.figureMode.checked ? '1' : '0');
    for (const [param, id] of [['times','figureTimes'], ['spacing','figureSpacing'],
                              ['width','figureWidth'], ['height','figureHeight']]) {
      url.searchParams.set(param, ui[id].value);
    }
    url.searchParams.set('labels', ui.figureLabels.checked ? '1' : '0');
    history.replaceState({}, '', url);
  }
  function rebuild() {
    const meta = state.metadata;
    if (!meta) return false;
    try {
      const config = settings();
      selected = selectFigureFrames(ui.figureTimes.value, meta.times);
      const next = JSON.stringify([selected, elements.pointSize.value, elements.bondSize.value,
        elements.showBonds.checked, elements.atomColorMode.checked, elements.centerMode.checked,
        elements.fragmentPalette.value, elements.fragmentColor.value]);
      if (signature !== next || cacheMeta !== meta) {
        disposeSnapshots();
        const originalFrame = state.frame;
        try {
          for (const selection of selected) {
            state.frame = selection.frame;
            rebuildStructureScene();
            const group = new THREE.Group();
            for (const child of structureGroup.children) if (child.visible) {
              const copy = child.clone(); copy.material = child.material.clone(); group.add(copy);
            }
            group.userData = {...selection}; snapshots.add(group);
          }
        } finally {
          state.frame = originalFrame; rebuildStructureScene();
        }
        signature = next; cacheMeta = meta;
      }
      const notice = selected.map(s => `t = ${Number(s.time.toFixed(5))}`).join(' · ') +
        (selected.some(s => Math.abs(s.time - s.requested) > 1e-8) ? ' · nearest recorded frames' : '');
      if (notice !== lastNotice || performance.now() >= exportNoticeUntil) {
        ui.figureStatus.textContent = notice; lastNotice = notice;
      }
      ui.figureSpacingOutput.value = `${config.spacing.toFixed(2)}×`;
      elements.stage.style.setProperty('--figure-aspect', `${config.width} / ${config.height}`);
      ui.saveFigure.disabled = exporting; valid = true; return true;
    } catch (error) {
      ui.figureStatus.textContent = error.message; ui.saveFigure.disabled = true;
      valid = false; return false;
    }
  }
  function arrange(width, height) {
    const config = settings();
    const boxes = [];
    for (const group of snapshots.children) {
      group.position.set(0, 0, 0); group.quaternion.copy(state.orientation);
      group.updateMatrixWorld(true); boxes.push(new THREE.Box3().setFromObject(group));
    }
    if (!boxes.length) return;
    // Use one common spacing and physical scale, not an independent fit per frame.
    const half = Math.max(...boxes.map(b => Math.max(Math.abs(b.min.x), Math.abs(b.max.x))), .1);
    const stride = half * 2 * config.spacing;
    const union = new THREE.Box3();
    snapshots.children.forEach((group, i) => {
      group.position.x = (i - (boxes.length - 1)/2) * stride;
      const shifted = boxes[i].clone().translate(group.position); union.union(shifted);
    });
    const center = union.getCenter(new THREE.Vector3());
    const extent = union.getSize(new THREE.Vector3());
    const aspect = width / height;
    const labelFraction = config.labels ? .14 : 0;
    const visibleHeight = Math.max(extent.y / (1 - labelFraction), extent.x / aspect, .1) * 1.10 / state.zoom;
    const visibleWidth = visibleHeight * aspect;
    const cy = center.y - visibleHeight * labelFraction / 2;
    camera.left = center.x - visibleWidth/2; camera.right = center.x + visibleWidth/2;
    camera.top = cy + visibleHeight/2; camera.bottom = cy - visibleHeight/2;
    camera.near = .01; camera.far = Math.max(2000, 1000 - union.min.z + 100);
    camera.updateProjectionMatrix();
    layout = {width, height, x: snapshots.children.map(g =>
      (g.position.x - camera.left) / visibleWidth * width), labels: config.labels};
  }
  function paintLabels(context, width, height) {
    if (!layout?.labels) return;
    const bg = new THREE.Color(activeBackground().center);
    context.fillStyle = .2126*bg.r + .7152*bg.g + .0722*bg.b > .4 ? '#222' : '#fff';
    context.textAlign = 'center'; context.textBaseline = 'middle';
    context.font = `${Math.max(10, Math.min(height * .065, width / selected.length * .12))}px Arial, sans-serif`;
    selected.forEach((s, i) => context.fillText(`t = ${Number(s.time.toFixed(5))}`,
      layout.x[i] / layout.width * width, height * .945));
  }
  function render(width, height, preview = true) {
    if (!rebuild()) return;
    arrange(width, height);
    scene.background = new THREE.Color(activeBackground().center);
    renderer.render(scene, camera);
    if (preview) {
      const overlay = ui.figureLabelCanvas;
      if (overlay.width !== width || overlay.height !== height) {overlay.width = width; overlay.height = height;}
      const ctx = overlay.getContext('2d'); ctx.clearRect(0, 0, width, height);
      paintLabels(ctx, width, height);
    }
  }
  function setMode(enabled) {
    if (enabled && state.recording) {ui.figureMode.checked = false; return;}
    ui.figureMode.checked = enabled;
    document.body.classList.toggle('figure-mode', enabled);
    ui.figureControls.hidden = !enabled; ui.figureLabelCanvas.hidden = !enabled;
    if (enabled) {
      oldControls = {autoRotate: elements.autoRotate.checked, autoFit: elements.autoFit.checked,
        background: elements.backgroundPreset.value, zoom: state.zoom};
      setPlaying(false); elements.autoRotate.checked = false; elements.autoFit.checked = false;
      elements.backgroundPreset.value = 'white'; state.zoom = 1;
      elements.stage.classList.add('light-background'); rebuild();
    } else if (oldControls) {
      elements.autoRotate.checked = oldControls.autoRotate; elements.autoFit.checked = oldControls.autoFit;
      elements.backgroundPreset.value = oldControls.background; state.zoom = oldControls.zoom;
      elements.stage.classList.toggle('light-background', activeBackground().center === '#ffffff');
      state.sceneDirty = true;
    }
    elements.play.disabled = enabled; elements.timeline.disabled = enabled;
    elements.autoFit.disabled = enabled; elements.saveVideo.disabled = enabled;
    elements.hover.hidden = true; remember();
  }
  async function savePNG() {
    if (exporting || !state.metadata || !rebuild() || !valid) return;
    const basename = `${state.metadata.checkpoint_weights || 'conditional'}-${state.metadata.checkpoint_step || 'path'}-${elements.trajectorySelect.value || 'trajectory'}`;
    exporting = true;
    try {
      const {width, height} = settings();
      const limit = renderer.getContext().getParameter(renderer.getContext().MAX_RENDERBUFFER_SIZE);
      if (width > limit || height > limit) throw new Error(`This browser supports up to ${limit} px per side.`);
      ui.saveFigure.disabled = true;
      const output = document.createElement('canvas'); output.width = width; output.height = height;
      renderer.setPixelRatio(1); renderer.setSize(width, height, false); render(width, height, false);
      const ctx = output.getContext('2d'); ctx.drawImage(renderer.domElement, 0, 0); paintLabels(ctx, width, height);
      const blob = await new Promise(resolve => output.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('The browser could not create this image. Try smaller dimensions.');
      const url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `${basename}-frames.png`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      ui.figureStatus.textContent = `Saved ${width.toLocaleString()} × ${height.toLocaleString()} px PNG.`;
      exportNoticeUntil = performance.now() + 8000;
    } catch (error) {ui.figureStatus.textContent = error.message;}
    finally {exporting = false; ui.saveFigure.disabled = !valid;}
  }
  ui.figureMode.addEventListener('change', () => setMode(ui.figureMode.checked));
  for (const id of ['figureTimes','figureSpacing','figureLabels','figureWidth','figureHeight']) {
    ui[id].addEventListener(id === 'figureSpacing' ? 'input' : 'change', () => {remember(); rebuild();});
  }
  ui.figureFit.addEventListener('click', () => {state.zoom = 1;});
  ui.figureReset.addEventListener('click', () => {state.orientation.identity(); state.zoom = 1;});
  ui.saveFigure.addEventListener('click', savePNG);
  ui.figureSettings.addEventListener('click', () => {
    if (!state.metadata || !rebuild()) return;
    const recipe = {format_version: 1, source_url: location.href, sample: elements.trajectorySelect.value,
      checkpoint: state.metadata.checkpoint, checkpoint_weights: state.metadata.checkpoint_weights,
      frames: selected, ...settings(), orientation: state.orientation.toArray(), zoom: state.zoom,
      atom_size: Number(elements.pointSize.value), bond_width: Number(elements.bondSize.value),
      show_bonds: elements.showBonds.checked, atom_colors: elements.atomColorMode.checked,
      fragment_palette: elements.fragmentPalette.value, fragment_color: elements.fragmentColor.value,
      fragment_centers: elements.centerMode.checked, background: activeBackground().center,
      quality_filter: document.getElementById('qualityFilter').checked ? document.getElementById('qualityCriterion').value : 'all',
      quality_scope: state.catalog?.quality_scope || 'sample',
      bond_policy: 'final-X1 within-fragment throughout; between-fragment only at final recorded frame'};
    const url = URL.createObjectURL(new Blob([JSON.stringify(recipe, null, 2)+'\n'], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `${recipe.sample}-figure-settings.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  });
  if (query.get('figure') === '1') setMode(true);
  return {get enabled() {return ui.figureMode.checked;}, render, refresh: rebuild};
}
