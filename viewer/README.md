# Molecular trajectory viewer

This directory is independent of the model and training packages. It requires a browser and a static HTTP server. `publish.py` uses only Python's standard library.

```bash
python viewer/publish.py \
  --panel artifacts/fragment --label 'Fragment Noise' --evaluation artifacts/evaluation/fragment \
  --panel artifacts/bm --label 'Brownian Atoms' --evaluation artifacts/evaluation/bm \
  --output artifacts/site
python -m http.server 8765 --directory artifacts/site
```

Open `http://localhost:8765/index.html?figure=1` for static figures, or omit `figure=1` for playback. Choose physical times, spacing, colors, atom/bond sizes and background. PNG exports contain only the molecular frames. Matched panels offer a button to switch models while retaining the same sample. Validity and connectivity filters apply jointly to both members of each pair and are enabled by default.

Figures default to times 0, 0.2, 0.4, 0.6, 0.8 and 1. Fragment palettes include vivid, colorblind-friendly, muted, blue/red and a custom single color. Keep atom-type colors enabled for element-colored atoms, or disable them to apply the palette to atoms as well. Color choices persist when switching models and are included in saved figure settings and URL parameters (`palette`, `fragmentColor`).

A panel contains `trajectories.json`, listing IDs, labels and relative trajectory URLs. Each trajectory has a `trajectory.json` descriptor, little-endian float32 coordinates of shape `(frames, atoms, 3)` in `positions.f32`, and uint8 atom labels of shape `(frames, atoms)` in `atom_labels.u8`. Descriptors contain physical times, atom indices, fragment IDs, atom-label names, source identity and display radii. Frames are selected from recorded states; coordinates are never interpolated.

Evaluation input is the ADiT evaluator’s `samples.jsonl`, with `file`, `valid` and `connected` for each XYZ. The publisher checks that paired recordings share the same source, atom indices and partition. Third-party Three.js files and their license are in `vendor/`.
