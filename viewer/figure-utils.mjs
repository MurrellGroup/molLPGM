/** Physical-time selection uses recorded states only; never interpolate coordinates. */
export function selectFigureFrames(text, times) {
  const values = text.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!values.length || values.length > 24 || values.some(t => !Number.isFinite(t) || t < 0 || t > 1)) {
    throw new Error('Choose 1–24 physical times between 0 and 1.');
  }
  if (values.some((t, i) => i && t <= values[i - 1])) {
    throw new Error('Enter times in increasing order, without duplicates.');
  }
  const frames = values.map(t => {
    let lo = 0, hi = times.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (times[mid] < t) lo = mid + 1; else hi = mid;
    }
    return lo > 0 && Math.abs(times[lo - 1] - t) <= Math.abs(times[lo] - t) ? lo - 1 : lo;
  });
  if (new Set(frames).size !== frames.length) {
    throw new Error('Two requested times select the same recorded frame. Space them farther apart.');
  }
  return frames.map((frame, i) => ({frame, requested: values[i], time: times[frame]}));
}

export function filterTrajectories(trajectories, quality, enabled, criterion, paired = false) {
  if (!enabled) return trajectories;
  const passes = id => {
    const row = quality?.samples?.[id];
    if (criterion === 'valid_connected') return row?.valid === true && row?.connected === true;
    return ['valid', 'connected'].includes(criterion) && row?.[criterion] === true;
  };
  const byId = new Map(trajectories.map(item => [item.id, item]));
  return trajectories.filter(item => passes(item.id) && (!paired ||
    (item.paired_id !== item.id && byId.get(item.paired_id)?.paired_id === item.id && passes(item.paired_id))));
}
