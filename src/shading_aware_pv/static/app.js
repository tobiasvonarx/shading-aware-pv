import {ImageGallery} from "/emboss-imagery/gallery.js";
const $ = id => document.getElementById(id);
const batchValue = '__selection__';
let current = null, selection = [], activeRun = null, openedJob = null, configuredRun = null;
let runEpoch = 0, resultEpoch = 0, currentView = 'find', submitting = false;
let pollTimer = null, pollRequest = 0;
let selectedDesign = 'clean_slate', countTimer = null, preparingCounts = false;

function notice(id, message = '', error = false) {
  const element = $(id); element.textContent = message;
  element.dataset.tone = error ? 'error' : 'info';
  delete element.dataset.source;
}
function retryNotice(id, message, retry, source = '') {
  notice(id, message, true);
  $(id).dataset.source = source;
  const button = document.createElement('button'); button.type = 'button';
  button.className = 'secondary'; button.textContent = 'Retry'; button.onclick = retry;
  $(id).append(button);
}
async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = response.statusText || 'Request failed';
    try { const body = await response.json(); if (typeof body.detail === 'string') message = body.detail; } catch {}
    throw new Error(message);
  }
  return response.json();
}
function syncAnalysisAction() {
  const batch = $('houses').value === batchValue;
  $('run').hidden = batch; $('run-all').hidden = !batch;
  $('run').disabled = submitting || !$('houses').value;
  $('run-all').disabled = submitting || !selection.length;
  $('run').textContent = submitting ? 'Starting analysis…' : 'Review orthophotos';
  $('run-all').textContent = submitting ? 'Starting analysis…' : `Review ${selection.length} buildings`;
}
function updateSelection(ids) {
  selection = [...new Set(ids)];
  $('houses').querySelector(`option[value="${batchValue}"]`)?.remove();
  for (const id of selection) {
    if (![...$('houses').options].some(option => option.value === id)) $('houses').add(new Option(id, id));
  }
  if (selection.length > 1) {
    $('houses').add(new Option(`All ${selection.length} prepared buildings`, batchValue), 1);
    $('houses').value = batchValue;
  } else if (selection.length) $('houses').value = selection[0];
  $('selection-note').textContent = selection.length ? `${selection.length} ${selection.length === 1 ? 'building ready' : 'buildings ready'}.` : 'Prepare a building on the map to begin.';
  syncAnalysisAction();
}
async function loadHouses() {
  try {
    updateSelection((await request('/api/houses')).map(item => item.id));
    if ($('status').dataset.source === 'houses') notice('status');
  }
  catch { $('selection-note').textContent = 'Saved buildings could not be loaded.'; retryNotice('status', 'Check the connection and try again.', loadHouses, 'houses'); }
}
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.data?.type !== 'building-data-acquired') return;
  updateSelection(event.data.houses || []); notice('status');
  $('settings-title').focus({preventScroll: true});
  if (matchMedia('(max-width: 800px)').matches) $('settings-title').scrollIntoView({block: 'start'});
});
$('houses').onchange = syncAnalysisAction;
function syncPowerInput() {
  const basis = $('power-basis').value;
  $('power-label').hidden = basis === 'geometry';
  $('power-value').required = basis !== 'geometry';
  $('power-name').textContent = basis === 'dc_capacity_kwp' ? 'Installed capacity (kWp)' : 'Module wattage (W)';
}
$('power-basis').onchange = syncPowerInput;
function config() {
  const values = {year: Number($('year').value), context_half_extent: Number($('context').value),
    general_loss_percent: Number($('loss').value), dc_ac_ratio: Number($('dc-ac').value),
    inverter_efficiency_percent: Number($('efficiency').value)};
  const basis = $('power-basis').value;
  if (basis !== 'geometry') {
    const power = Number($('power-value').value);
    if (!(power > 0)) throw new Error('Enter a positive module wattage or installed capacity.');
    values[basis] = power;
  }
  if ($('inverter').value) values.inverter_ac_kw = Number($('inverter').value);
  return values;
}
$('analysis-form').addEventListener('invalid', event => {
  if ($('advanced').contains(event.target)) $('advanced').open = true;
}, true);
let review = null, reviewVersion = 0, resultJob = null;
const gallery = new ImageGallery($('image-gallery'), $('gallery-status'), id => {
  if (!review) return;
  review.choices[review.house] = id;
  review.confirmed.delete(review.house);
  updateGalleryAction();
});
function closeGallery() {
  if (submitting) return;
  reviewVersion++; gallery.clear(); review = null; $('image-dialog').close();
}
$('close-gallery').onclick = closeGallery;
$('cancel-gallery').onclick = closeGallery;
$('image-dialog').addEventListener('cancel', event => { event.preventDefault(); closeGallery(); });
$('gallery-footprint').onchange = () => $('image-gallery').classList.toggle('hide-footprints', !$('gallery-footprint').checked);
function updateGalleryAction() {
  if (!review) return;
  const remaining = review.ids.filter(id => id !== review.house && !review.confirmed.has(id));
  $('confirm-gallery').disabled = submitting || !review.info || !gallery.ready.has(review.choices[review.house]);
  $('confirm-gallery').textContent = submitting ? 'Starting analysis…' : remaining.length ? 'Confirm & next building' : review.mode === 'change' ? 'Use image & rerun' : `Confirm & analyze${review.ids.length > 1 ? ` ${review.ids.length} buildings` : ''}`;
  $('gallery-note').textContent = review.ids.length > 1 ? `${review.confirmed.size} of ${review.ids.length} buildings confirmed. Review each building before analysis.` : 'The selected image will be used to reconstruct the roof before solar analysis.';
  for (const button of $('gallery-buildings').children) {
    button.classList.toggle('active', button.dataset.house === review.house);
    button.setAttribute('aria-pressed', String(button.dataset.house === review.house));
    button.disabled = submitting;
  }
  $('close-gallery').disabled = $('cancel-gallery').disabled = submitting;
}
async function openGallery(ids, values, mode = 'new', choices = {}) {
  if (submitting) return;
  closeGallery();
  review = {ids, values, mode, choices: {...choices}, confirmed: new Set(), cache: new Map(), house: null, info: null};
  $('gallery-buildings').hidden = ids.length < 2;
  $('gallery-buildings').replaceChildren(...ids.map(id => {
    const button = document.createElement('button'); button.className = 'secondary'; button.textContent = id; button.dataset.house = id;
    button.onclick = () => reviewHouse(id); return button;
  }));
  $('image-dialog').showModal();
  await reviewHouse(ids[0]);
}
async function reviewHouse(id) {
  const state = review, version = ++reviewVersion;
  gallery.clear(); state.house = id; state.info = null;
  notice('gallery-error'); $('retry-gallery').hidden = true;
  $('gallery-title').textContent = `Choose an orthophoto · ${id}`;
  $('gallery-status').className = ''; $('gallery-status').textContent = 'Preparing available images…';
  updateGalleryAction();
  try {
    let info = state.cache.get(id);
    if (!info) {
      const job = await request(`/api/houses/${encodeURIComponent(id)}/imagery`, {method: 'POST'});
      for (;;) {
        if (version !== reviewVersion) return;
        const status = await request(`/api/imagery-jobs/${job.id}`);
        if (version !== reviewVersion) return;
        if (status.status === 'completed') { info = status.result; break; }
        if (['failed', 'interrupted'].includes(status.status)) throw Error(status.error || 'Images could not be prepared.');
        $('gallery-status').textContent = status.messages?.at(-1) || 'Preparing available images…';
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      state.cache.set(id, info);
    }
    if (version !== reviewVersion) return;
    state.info = info;
    if (!info.candidates.some(item => item.id === state.choices[id])) state.choices[id] = info.selected_id;
    gallery.show(id, info, state.choices[id]);
    updateGalleryAction();
  } catch (error) {
    if (version !== reviewVersion) return;
    $('gallery-status').textContent = error.message; $('gallery-status').className = 'error'; $('retry-gallery').hidden = false;
  }
}
$('retry-gallery').onclick = () => { if (review) reviewHouse(review.house); };
$('confirm-gallery').onclick = async () => {
  if (!review || submitting || !gallery.ready.has(review.choices[review.house])) return;
  review.confirmed.add(review.house);
  const next = review.ids.find(id => !review.confirmed.has(id));
  if (next) { await reviewHouse(next); return; }
  const state = review;
  submitting = true; syncAnalysisAction(); updateGalleryAction(); notice('gallery-error');
  try {
    const run = await request('/api/solar/runs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({houses: state.ids, config: state.values, image_choices: state.choices})});
    submitting = false; closeGallery();
    clearResult(); activeRun = run.id; location.hash = 'analyses';
    await refreshRuns(run.id);
    notice('history-status', `Analysis started for ${state.ids.length} ${state.ids.length === 1 ? 'building' : 'buildings'}.`);
  } catch (error) { notice('gallery-error', `Analysis could not start. ${error.message}`, true); }
  finally { submitting = false; syncAnalysisAction(); updateGalleryAction(); }
};
$('change-image').onclick = () => {
  if (resultJob) openGallery([resultJob.house_id], resultJob.config, 'change', resultJob.image_choices || {});
};
$('analysis-form').onsubmit = async event => {
  event.preventDefault();
  if (submitting) return;
  try {
    const values = config(), ids = $('houses').value === batchValue ? selection : [$('houses').value];
    if (!ids.length || !ids[0]) throw new Error('Prepare a building before starting an analysis.');
    notice('status'); await openGallery([...ids], values);
  } catch (error) { notice('status', error.message, true); }
};
function showView() {
  currentView = location.hash === '#analyses' ? 'analyses' : 'find';
  $('find-view').hidden = currentView !== 'find'; $('analyses-view').hidden = currentView !== 'analyses';
  for (const view of ['find', 'analyses']) {
    if (view === currentView) $(`nav-${view}`).setAttribute('aria-current', 'page');
    else $(`nav-${view}`).removeAttribute('aria-current');
  }
  if (currentView === 'analyses') {
    if (current) requestAnimationFrame(render);
    else if (activeRun) pollRun(activeRun, runEpoch);
  } else {
    document.querySelector('iframe').contentWindow?.dispatchEvent(new Event('resize'));
  }
}
window.addEventListener('hashchange', showView);
document.querySelector('.skip-link').onclick = event => {
  event.preventDefault(); $('workspace').focus();
};
function clearResult() {
  clearTimeout(countTimer); countTimer = null; preparingCounts = false;
  resultEpoch++; current = null; openedJob = null; resultJob = null;
  $('results').hidden = true; $('results').removeAttribute('aria-busy');
  $('download').removeAttribute('href'); $('design').replaceChildren(); notice('result-status');
  $('panel-control').hidden = true; notice('design-status');
}
function clearRun() {
  cancelPoll(); pollRequest++;
  clearResult(); activeRun = null; configuredRun = null;
  $('job-rows').replaceChildren(); $('remove-run').disabled = true;
}
async function refreshRuns(selected) {
  cancelPoll();
  const epoch = ++runEpoch;
  $('run-summary').textContent = 'Loading analyses…';
  try {
    const runs = await request('/api/solar/runs');
    if (epoch !== runEpoch) return;
    $('analysis-count').textContent = runs.length; $('analysis-count').hidden = !runs.length;
    $('run-toolbar').hidden = !runs.length; $('new-analysis').hidden = !runs.length; $('history-empty').hidden = !!runs.length;
    $('report-list').hidden = !runs.length;
    $('runs').replaceChildren(new Option('Select an analysis run', ''));
    for (const run of runs) $('runs').add(new Option(`${new Date(run.created_at).toLocaleString()} · ${run.houses.length} ${run.houses.length === 1 ? 'building' : 'buildings'}`, run.id));
    const preferred = selected || activeRun;
    const id = runs.some(run => run.id === preferred) ? preferred : runs[0]?.id;
    if (id) { clearResult(); $('runs').value = id; activeRun = id; await pollRun(id, epoch); }
    else { clearRun(); $('run-summary').textContent = ''; }
  } catch (error) {
    if (epoch !== runEpoch) return;
    $('run-summary').textContent = ''; $('history-empty').hidden = true;
    retryNotice('history-status', `Analyses could not be loaded. ${error.message}`, () => { notice('history-status'); refreshRuns(selected); });
  }
}
$('runs').onchange = () => {
  runEpoch++; clearResult(); activeRun = $('runs').value;
  if (activeRun) pollRun(activeRun, runEpoch);
  else { clearRun(); $('run-summary').textContent = 'Select an analysis run.'; }
};
async function refreshTrash() {
  try {
    const items = await request('/api/solar/trash'); $('trash-list').replaceChildren();
    $('removed-count').textContent = items.length ? `(${items.length})` : '';
    for (const item of items) {
      const row = document.createElement('li'), label = document.createElement('span'); label.textContent = item.label;
      const undo = document.createElement('button'); undo.className = 'secondary'; undo.textContent = 'Undo'; undo.dataset.removed = item.id; undo.setAttribute('aria-label', `Undo removal: ${item.label}`);
      undo.onclick = async () => {
        undo.disabled = true; undo.textContent = 'Restoring…';
        try {
          const restored = await request(`/api/solar/${item.kind}/${item.id}/restore`, {method: 'POST'});
          notice('history-status', 'Restored.'); await refreshRuns(restored.run_id); await refreshTrash(); $('runs').focus({preventScroll: true});
        } catch (error) { notice('history-status', error.message, true); undo.disabled = false; undo.textContent = 'Undo'; }
      };
      row.append(label, undo); $('trash-list').append(row);
    }
    if (!items.length) { const row = document.createElement('li'); row.className = 'muted'; row.textContent = 'No removed items.'; $('trash-list').append(row); }
  } catch (error) { retryNotice('history-status', `Removed items could not be loaded. ${error.message}`, refreshTrash); }
}
async function removeItem(kind, id, button) {
  button.disabled = true;
  try {
    await request(`/api/solar/${kind}/${id}`, {method: 'DELETE'});
    runEpoch++; clearResult(); if (kind === 'runs') activeRun = null;
    notice('history-status', `${kind === 'runs' ? 'Run' : 'Report'} removed. Undo is available in Removed items.`);
    $('removed-items').open = true; await refreshRuns(); await refreshTrash();
    $('trash-list').querySelector(`[data-removed="${id}"]`)?.focus({preventScroll: true});
  } catch (error) { notice('history-status', error.message, true); button.disabled = false; }
}
$('remove-run').onclick = () => { if (activeRun) removeItem('runs', activeRun, $('remove-run')); };
async function openResult(job) {
  clearResult(); const epoch = resultEpoch, run = activeRun;
  notice('result-status', `Loading report for ${job.house_id}…`); $('results').setAttribute('aria-busy', 'true');
  try {
    const result = await request(job.result_url);
    if (epoch !== resultEpoch || run !== activeRun) return;
    current = result; openedJob = job.id; resultJob = job; notice('result-status'); $('results').removeAttribute('aria-busy');
    $('download').href = job.result_url; $('download').download = `${job.house_id}-solar.json`;
    $('result-building').textContent = job.house_id; $('results').hidden = false;
    selectedDesign = current.designs.installed ? 'installed' : 'clean_slate';
    configureDesigns();
    for (const row of $('job-rows').rows) row.setAttribute('aria-selected', String(row.dataset.job === job.id));
    if (currentView === 'analyses') render();
  } catch (error) {
    if (epoch !== resultEpoch || run !== activeRun) return;
    $('results').removeAttribute('aria-busy'); retryNotice('result-status', `Report could not be loaded. ${error.message}`, () => openResult(job));
  }
}
function hasCountDesigns() {
  return current?.provenance?.method_revision === 8 && !!current.designs_by_count?.[current.designs.clean_slate?.panel_count];
}
function configureDesigns() {
  const labels = {installed: 'Installed panels', relocated: 'Same-count relocation', clean_slate: 'New design'};
  $('design').replaceChildren();
  for (const key of Object.keys(current.designs)) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'secondary'; button.dataset.design = key;
    button.textContent = labels[key] || key;
    button.onclick = () => { selectedDesign = key; render(); };
    $('design').append(button);
  }
  const maximum = current.designs.clean_slate?.panel_count ?? 0;
  $('panel-count').max = maximum; $('panel-count').value = maximum;
  $('panel-maximum').textContent = `${maximum} panels · roof capacity`;
  $('panel-count').disabled = !hasCountDesigns() || maximum === 0;
}
async function prepareCounts(start = true, epoch = resultEpoch) {
  if (!current || epoch !== resultEpoch) return;
  preparingCounts = true;
  notice('design-status', 'Preparing adjustable designs… Showing the saved maximum layout for now.');
  try {
    const status = await request(`/api/solar/jobs/${openedJob}/counts`, start ? {method: 'POST'} : undefined);
    if (epoch !== resultEpoch) return;
    if (status.status === 'failed') throw new Error(status.message || 'Adjustable designs could not be prepared.');
    if (status.status === 'complete') {
      const result = await request(status.result_url);
      if (epoch !== resultEpoch) return;
      current = result; preparingCounts = false; notice('design-status');
      $('download').href = status.result_url;
      configureDesigns(); render();
    } else {
      notice('design-status', `${status.message || 'Preparing adjustable designs…'} Showing the saved maximum layout for now.`);
      countTimer = setTimeout(() => prepareCounts(false, epoch), 1500);
    }
  } catch (error) {
    if (epoch !== resultEpoch) return;
    // Keep preparation marked pending so rendering does not immediately retry a failed request.
    retryNotice('design-status', error.message, () => prepareCounts(true, epoch));
  }
}
function cancelPoll() {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
}
async function pollRun(id, epoch = runEpoch) {
  cancelPoll();
  const pending = ++pollRequest;
  try {
    const run = await request(`/api/solar/runs/${id}`);
    if (id !== activeRun || epoch !== runEpoch || pending !== pollRequest) return;
    if ($('history-status').dataset.source === 'poll') notice('history-status');
    $('remove-run').disabled = run.status !== 'finished';
    $('remove-run').title = run.status === 'finished' ? '' : 'Wait for all analyses in this run to finish.';
    if (configuredRun !== id) {
      const values = run.config;
      for (const [field, key] of [['year','year'],['context','context_half_extent'],['loss','general_loss_percent'],['dc-ac','dc_ac_ratio'],['efficiency','inverter_efficiency_percent']]) $(field).value = values[key];
      $('inverter').value = values.inverter_ac_kw ?? '';
      const basis = values.module_wattage_w != null ? 'module_wattage_w' : values.dc_capacity_kwp != null ? 'dc_capacity_kwp' : 'geometry';
      $('power-basis').value = basis; $('power-value').value = values[basis] ?? ''; syncPowerInput(); configuredRun = id;
    }
    const active = run.houses.length - run.complete_count - run.failed_count;
    $('run-summary').textContent = `${run.complete_count} complete${active ? ` · ${active} in progress` : ''}${run.failed_count ? ` · ${run.failed_count} failed` : ''} · ${run.houses.length} ${run.houses.length === 1 ? 'building' : 'buildings'}`;
    $('job-rows').replaceChildren();
    for (const job of run.jobs) {
      const row = document.createElement('tr'); row.dataset.job = job.id; row.setAttribute('aria-selected', String(openedJob === job.id));
      row.insertCell().textContent = job.house_id;
      const status = row.insertCell(), state = document.createElement('span'); state.className = 'job-state'; state.dataset.state = job.status;
      state.textContent = {complete: 'Complete', failed: 'Failed', queued: 'Queued', running: 'Running'}[job.status] || job.status; status.append(state);
      if (job.message && job.message.toLowerCase() !== job.status && job.message !== 'Solar analysis complete') {
        const progress = document.createElement('span'); progress.className = 'job-progress'; progress.textContent = job.message; status.append(progress);
      }
      const cell = row.insertCell();
      if (job.status === 'complete') {
        const button = document.createElement('button'); button.textContent = 'Open report'; button.className = 'secondary'; button.setAttribute('aria-label', `Open report for ${job.house_id}`);
        button.onclick = () => openResult(job); cell.append(button);
      }
      const remove = document.createElement('button'); remove.textContent = 'Remove'; remove.className = 'secondary'; remove.setAttribute('aria-label', `Remove report for ${job.house_id}`);
      remove.disabled = !['complete', 'failed'].includes(job.status); remove.title = remove.disabled ? 'Wait for this analysis to finish.' : 'Remove report';
      remove.onclick = () => removeItem('jobs', job.id, remove); cell.append(remove); $('job-rows').append(row);
    }
    if (!openedJob && currentView === 'analyses') { const first = run.jobs.find(job => job.status === 'complete'); if (first) await openResult(first); }
    if (run.status !== 'finished' && pending === pollRequest) {
      cancelPoll();
      pollTimer = setTimeout(() => {
        pollTimer = null;
        if (activeRun === id && epoch === runEpoch) pollRun(id, epoch);
      }, 2000);
    }
  } catch (error) {
    if (id === activeRun && epoch === runEpoch && pending === pollRequest)
      retryNotice('history-status', `Progress could not be loaded. ${error.message}`, () => pollRun(id, epoch), 'poll');
  }
}
function render() {
  if (!current) return;
  const adjustable = selectedDesign === 'clean_slate';
  $('panel-control').hidden = !adjustable;
  for (const button of $('design').children) button.setAttribute('aria-pressed', String(button.dataset.design === selectedDesign));
  const count = Number($('panel-count').value);
  $('panel-count-value').value = `${count} ${count === 1 ? 'panel' : 'panels'}`;
  $('panel-count').setAttribute('aria-valuetext', `${count} of ${$('panel-count').max} panels`);
  const d = adjustable && hasCountDesigns() ? current.designs_by_count[count] : current.designs[selectedDesign];
  const s = current.scene, state = $('state').value, origin = s.origin;
  if (adjustable && !hasCountDesigns() && !preparingCounts) prepareCounts();
  const metrics = d.states?.[state];
  $('summary').textContent = metrics ? `${d.panel_count} ${d.panel_count === 1 ? 'panel' : 'panels'} · ${Math.round(metrics.modeled_kwh).toLocaleString()} kWh AC · ${d.electrical.dc_capacity_kwp.toFixed(2)} kWp · ${current.hours_simulated.toLocaleString()} weather hours` : 'No modules fit this layout.';
  $('notes').textContent = [current.provenance?.method_revision !== 8 ? 'Saved result from an earlier calculation version. Run the analysis again to use the current method.' : '',
    d.retained_installed ? 'Existing panel positions retained for the same-count comparison.' : '',
    current.count_preparation ? 'Adjustable designs were recalculated with the current method using cached inputs. The original report is retained.' : '',
    current.partial_period ? 'Partial weather period; these totals are not annual.' : '', current.context.enabled ? 'Neighborhood shading includes terrain, buildings and vegetation.' : 'Neighborhood shading is disabled.',
    d.panel_count === 0 ? '' : d.snow_basis === 'snow-free placement estimate' ? 'Snow losses are excluded from this placement estimate.' : current.snow.available ? 'MeteoSwiss snow observations applied.' : `Snow model unavailable: ${current.snow.reason || 'no observations'}`,
    d.same_count_feasible === false ? `Requested ${d.requested_panel_count} modules; only ${d.panel_count} fit the specified clearances.` : ''].filter(Boolean).join('\n');
  if (typeof Plotly === 'undefined') { notice('result-status', 'Charts could not load. Reload this page to retry, or download the result as JSON.', true); return; }
  const xyz = points => [0, 1, 2].map(k => points.map(point => point[k] - origin[k]));
  const [x, y, z] = xyz(s.vertices), traces = [{type: 'mesh3d', x, y, z, i: s.faces.map(f => f[0]), j: s.faces.map(f => f[1]), k: s.faces.map(f => f[2]), color: '#a9bfba', opacity: .8, hoverinfo: 'skip', name: 'Roof'}];
  const [sx, sy, sz] = xyz(s.samples);
  traces.push({type: 'scatter3d', mode: 'markers', x: sx, y: sy, z: sz, marker: {size: d.panel_count ? 1 : 2, color: s.irradiation_kwh_m2[state], colorscale: 'YlOrRd', colorbar: {title: {text: 'kWh/m²'}}, opacity: d.panel_count ? .2 : .65}, name: 'Roof irradiation', hovertemplate: '%{marker.color:.1f} kWh/m²<extra></extra>'});
  for (const corners of d.corners) {
    const [px, py, pz] = xyz(corners);
    traces.push({type: 'mesh3d', x: px, y: py, z: pz, i: [0, 0], j: [1, 2], k: [2, 3], color: '#1c4e79', opacity: 1, hoverinfo: 'skip', showlegend: false});
    traces.push({type: 'scatter3d', mode: 'lines', x: [...px, px[0]], y: [...py, py[0]], z: [...pz, pz[0]], line: {color: '#adcdd9', width: 3}, hoverinfo: 'skip', showlegend: false});
  }
  const font = {family: 'system-ui, sans-serif', color: '#202c2a', size: 12};
  Plotly.react('scene', traces, {font, showlegend: false, margin: {l: 0, r: 0, b: 0, t: 16}, scene: {aspectmode: 'data', xaxis: {title: {text: 'East (m)'}}, yaxis: {title: {text: 'North (m)'}}, zaxis: {title: {text: 'Height (m)'}}}, uirevision: 'roof'}, {responsive: true, displaylogo: false});
  Plotly.react('chart', d.monthly ? [{type: 'bar', x: d.monthly.months, y: d.monthly.ac_kwh[state], marker: {color: '#175f59'}}] : [],
    {font, title: {text: 'Monthly modeled AC energy', font: {size: 16}}, xaxis: {type: 'category', tickvals: d.monthly?.months, ticktext: d.monthly?.months.map(month => new Date(`${month}-01T00:00:00Z`).toLocaleDateString(undefined, {month: 'short', timeZone: 'UTC'}))}, yaxis: {title: {text: 'kWh'}, gridcolor: '#e2e9e5'}, margin: {t: 56, l: 64, r: 16, b: 40}}, {responsive: true, displaylogo: false});
}
let renderFrame = null;
$('panel-count').oninput = () => {
  const count = Number($('panel-count').value);
  $('panel-count-value').value = `${count} ${count === 1 ? 'panel' : 'panels'}`;
  if (renderFrame !== null) cancelAnimationFrame(renderFrame);
  renderFrame = requestAnimationFrame(() => { renderFrame = null; render(); });
};
$('state').onchange = render;
showView(); loadHouses(); refreshRuns(); refreshTrash();
