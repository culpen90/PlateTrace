'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const ACTIVE = new Set(['queued', 'pending', 'running', 'researching', 'starting', 'cancelling']);
  const FINAL = new Set(['completed', 'complete', 'succeeded', 'failed', 'error', 'cancelled', 'canceled', 'interrupted', 'incomplete']);
  const state = {
    config: null, run: null, stream: null, poll: null, sourceRefresh: null,
    eventIds: new Set(), eventCount: 0, submitting: false, modelRequest: 0,
    selection: 0, runListRequest: 0, providers: [],
  };

  const el = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null) node.textContent = String(content);
    return node;
  };
  const readable = (value) => typeof value === 'string' ? value : value === undefined || value === null ? '' : JSON.stringify(value, null, 2);
  const statusOf = (run) => String(run?.status || 'queued').toLowerCase();
  const providerLabel = (id) => state.providers.find((p) => p.id === id)?.label || (id === 'demo' ? 'Demo' : id === 'ollama' ? 'Ollama' : id === 'openrouter' ? 'OpenRouter' : id || 'Agent');
  const isDemo = (id) => ['demo', 'fixture'].includes(id);
  const safeUrl = (value) => {
    try {
      const url = new URL(value);
      return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch { return null; }
  };
  const showAlert = (id, message) => {
    $(id).textContent = message || '';
    $(id).hidden = !message;
  };
  const timeLabel = (value, compact = true) => {
    const time = new Date(value);
    if (Number.isNaN(time.getTime())) return '';
    return compact ? time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }) : time.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  async function api(path, options = {}) {
    const response = await fetch(path, {
      credentials: 'same-origin',
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    let body;
    try { body = await response.json(); }
    catch { throw new Error(response.ok ? 'The server returned an unreadable response.' : `The server returned HTTP ${response.status}.`); }
    if (!response.ok) {
      const detail = body.error || body.detail || body.message || `Request failed (HTTP ${response.status}).`;
      throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item) => `${(item.loc || []).filter((part) => part !== 'body').join(' → ')}: ${item.msg || readable(item)}`).join('\n') : readable(detail));
    }
    return body;
  }

  function setConnection(online) {
    const pill = $('connection-status');
    pill.classList.toggle('offline', !online);
    pill.replaceChildren(el('span', 'live-dot'), document.createTextNode(online ? 'Workspace online' : 'Server unavailable'));
  }

  function setBusy() {
    const active = state.run && ACTIVE.has(statusOf(state.run));
    $('start-button').disabled = state.submitting || Boolean(active);
    $('start-button').firstElementChild.textContent = state.submitting ? 'Starting research…' : active ? 'Research in progress…' : 'Start research';
    $('demo-button').disabled = state.submitting || Boolean(active);
    $('stop-button').hidden = !active;
    $('stop-button').disabled = statusOf(state.run) === 'cancelling';
    $('stop-button').textContent = statusOf(state.run) === 'cancelling' ? 'Stopping…' : '■ Stop run';
  }

  function renderStatus(status) {
    const normalized = String(status || 'queued').toLowerCase();
    const badge = $('run-status');
    badge.textContent = normalized.replaceAll('_', ' ');
    badge.className = `status-badge ${/^[a-z]+$/.test(normalized) ? normalized : 'idle'}`;
    const active = ACTIVE.has(normalized);
    $('agent-dot').classList.toggle('muted', !active);
    $('agent-dot').parentElement.classList.toggle('processing', active);
    $('agent-state').textContent = active ? (normalized === 'cancelling' ? 'Stopping the agent' : 'Agent researching') : normalized === 'completed' || normalized === 'complete' || normalized === 'succeeded' ? 'Research complete' : normalized === 'failed' || normalized === 'error' ? 'Agent needs attention' : 'Agent stopped';
    setBusy();
  }

  async function configure() {
    try {
      const config = await api('/api/config');
      state.config = config;
      state.providers = Array.isArray(config.providers) ? config.providers : [];
      if (state.providers.length) {
        $('provider').replaceChildren(...state.providers.map((provider) => {
          const option = el('option', null, provider.label);
          option.value = provider.id;
          return option;
        }));
        $('provider').value = config.defaults?.provider || state.providers[0].id;
      }
      $('model-id').value = config.defaults?.model || state.providers.find((p) => p.id === $('provider').value)?.default_model || '';
      $('max-steps').max = String(Number(config.limits?.max_steps) || 40);
      $('max-steps').value = String(Math.min(12, Number($('max-steps').max)));
      $('app-version').textContent = `PlateTrace ${config.version || ''} · local edition`;
      const terminal = config.terminal || {};
      $('enable-terminal').disabled = !terminal.available;
      $('enable-terminal').checked = Boolean(terminal.available);
      $('terminal-note').textContent = terminal.available ? 'Internet-enabled Docker workspace. The agent chooses commands; host files and credentials stay outside.' : `${terminal.reason || 'Docker terminal is unavailable.'} Web research tools remain available.`;
      $('search-state').textContent = config.search?.available ? `Web search · ${config.search.provider || 'ready'}` : 'Web research · public websites';
      setConnection(true);
      showAlert('global-alert', '');
      updateProvider(false);
    } catch (error) {
      setConnection(false);
      showAlert('global-alert', `Cannot connect to the local server. ${error.message} Start PlateTrace, then refresh this page.`);
      $('enable-terminal').disabled = true;
      $('terminal-note').textContent = 'Connect to the local server to check terminal availability.';
    }
  }

  function updateProvider(resetModel = true) {
    const provider = $('provider').value;
    const demo = isDemo(provider);
    const configured = state.providers.find((p) => p.id === provider);
    state.modelRequest += 1;
    $('api-key-field').hidden = provider !== 'openrouter';
    $('model-id').disabled = demo;
    $('refresh-models').hidden = demo;
    $('model-options').replaceChildren();
    $('model-message').hidden = true;
    if (resetModel) $('model-id').value = configured?.default_model || (provider === 'ollama' ? 'qwen3:8b' : provider === 'openrouter' ? 'openai/gpt-4.1-mini' : 'demo');
    $('provider-description').textContent = demo ? 'A labeled sample run. No model calls or live web research.' : provider === 'ollama' ? 'Your local model chooses its own research steps. It needs tool-calling support.' : 'Your research context is sent to the selected cloud provider through OpenRouter. Choose a model with tool-calling support.';
  }

  async function loadModels() {
    const provider = $('provider').value;
    if (isDemo(provider)) return;
    const request = ++state.modelRequest;
    $('refresh-models').disabled = true;
    $('refresh-models').textContent = 'Loading…';
    $('model-message').textContent = 'Checking available models…';
    $('model-message').hidden = false;
    try {
      const data = await api(`/api/models?provider=${encodeURIComponent(provider)}`);
      if (request !== state.modelRequest) return;
      const models = Array.isArray(data.models) ? data.models : [];
      $('model-options').replaceChildren(...models.map((model) => {
        const option = el('option', null, model.name || model.id);
        option.value = model.id;
        return option;
      }));
      $('model-message').textContent = data.error || (models.length ? `${models.length} models available. Choose one from the model field or enter an ID.` : 'No models returned. You can enter a model ID directly.');
    } catch (error) {
      if (request === state.modelRequest) $('model-message').textContent = `${error.message} You can also enter a model ID directly.`;
    } finally {
      $('refresh-models').disabled = false;
      $('refresh-models').textContent = 'Refresh ↻';
    }
  }

  function readForm() {
    let records = [];
    if ($('records').value.trim()) {
      try { records = JSON.parse($('records').value); }
      catch { throw new Error('Authorized vehicle records must be valid JSON. Enter an array of vehicle record objects.'); }
      if (!Array.isArray(records) || records.some((record) => !record || typeof record !== 'object' || Array.isArray(record))) throw new Error('Authorized vehicle records must be a JSON array of objects.');
      const allowed = new Set(['plate', 'jurisdiction', 'vin', 'make', 'model', 'year', 'fuel', 'color']);
      for (const record of records) {
        const unknown = Object.keys(record).filter((key) => !allowed.has(key));
        if (unknown.length) throw new Error(`Unsupported record field${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}. Use only the vehicle fields listed below the records box.`);
      }
    }
    const sourceUrls = $('source-urls').value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (sourceUrls.some((url) => !safeUrl(url))) throw new Error('Each starting URL must be a complete http:// or https:// address, one per line.');
    const vin = $('vin').value.trim().toUpperCase();
    if (vin && !/^[A-HJ-NPR-Z0-9]{17}$/.test(vin)) throw new Error('A VIN must contain exactly 17 letters and digits, excluding I, O, and Q.');
    const provider = $('provider').value;
    const modelId = $('model-id').value.trim();
    if (!isDemo(provider) && !modelId) throw new Error('Enter a model ID for your selected research engine.');
    const data = {
      plate: $('plate').value.trim().toUpperCase(), jurisdiction: $('jurisdiction').value.trim(),
      objective: $('objective').value.trim(), provider, model_id: modelId,
      purpose: $('purpose').value, authorized: $('authorized').checked,
      source_urls: sourceUrls, records, enable_terminal: $('enable-terminal').checked && !$('enable-terminal').disabled,
      max_steps: Number($('max-steps').value),
    };
    if (!data.plate || !data.jurisdiction) throw new Error('Enter a license plate and its issuing jurisdiction.');
    if (!Number.isInteger(data.max_steps) || data.max_steps < 2 || data.max_steps > (Number(state.config?.limits?.max_steps) || 40)) throw new Error(`Choose a research turn limit between 2 and ${state.config?.limits?.max_steps || 40}.`);
    if (!data.authorized) throw new Error('Confirm that you are authorized to research this vehicle and use the records you provide.');
    if (vin) data.vin = vin;
    if ($('make').value.trim()) data.make = $('make').value.trim();
    if ($('vehicle-model').value.trim()) data.model = $('vehicle-model').value.trim();
    if ($('year').value) data.year = Number($('year').value);
    if (provider === 'openrouter' && $('api-key').value.trim()) data.api_key = $('api-key').value.trim();
    return data;
  }

  async function startRun(event) {
    event.preventDefault();
    if (state.submitting || (state.run && ACTIVE.has(statusOf(state.run)))) return;
    showAlert('form-error', '');
    try {
      const payload = readForm();
      state.submitting = true;
      setBusy();
      const result = await api('/api/runs', { method: 'POST', body: JSON.stringify(payload) });
      if (!result.id) throw new Error('The server did not return a research run ID.');
      $('api-key').value = '';
      await selectRun(result.id, { ...payload, api_key: undefined, id: result.id, status: 'queued', sources: [], events: [] });
      loadRuns();
    } catch (error) {
      showAlert('form-error', error.message);
    } finally {
      state.submitting = false;
      setBusy();
    }
  }

  function resetEvents() {
    state.eventIds.clear();
    state.eventCount = 0;
    $('activity-log').replaceChildren();
    $('event-count').textContent = '0';
  }

  function closeStream() {
    if (state.stream) state.stream.close();
    state.stream = null;
    clearInterval(state.poll);
    clearTimeout(state.sourceRefresh);
    state.poll = null;
    state.sourceRefresh = null;
  }

  function renderReport(report) {
    if (!report) {
      const status = statusOf(state.run);
      $('report-summary').textContent = ACTIVE.has(status) ? 'The agent is gathering evidence. Follow its searches, page reads, and terminal commands in the activity log below.' : status === 'failed' || status === 'error' ? 'This run stopped before a research brief was completed. Details are available in the activity log.' : 'This run has no completed research brief. You can inspect the steps and any sources collected below.';
      $('findings').replaceChildren();
      $('limitations-section').hidden = true;
      $('next-steps-section').hidden = true;
      return;
    }
    $('report-summary').textContent = readable(report.summary);
    $('findings').replaceChildren(...(Array.isArray(report.findings) ? report.findings : []).map((finding) => {
      const card = el('article', 'finding');
      card.append(el('h3', null, finding.title || 'Finding'), el('p', null, readable(finding.detail || finding.description)));
      const ids = Array.isArray(finding.source_ids) ? finding.source_ids : [];
      if (ids.length) {
        const citations = el('div', 'finding-citations');
        ids.forEach((id) => {
          const citation = el('a', 'citation', `↗ ${id}`);
          citation.href = `#source-${encodeURIComponent(String(id))}`;
          citation.setAttribute('aria-label', `Read source ${id}`);
          citations.append(citation);
        });
        card.append(citations);
      }
      return card;
    }));
    for (const [field, listId, sectionId] of [['limitations', 'limitations', 'limitations-section'], ['next_steps', 'next-steps', 'next-steps-section']]) {
      const values = Array.isArray(report[field]) ? report[field] : [];
      $(listId).replaceChildren(...values.map((value) => el('li', null, readable(value))));
      $(sectionId).hidden = !values.length;
    }
  }

  function renderSources(sources) {
    const entries = Array.isArray(sources) ? sources : [];
    $('source-count').textContent = String(entries.length);
    const list = $('source-list');
    if (!entries.length) {
      const empty = el('div', 'source-empty');
      empty.append(el('span', null, '◎'), el('p', null, 'Source links and excerpts will appear as your agent gathers evidence.'));
      list.replaceChildren(empty);
      return;
    }
    list.replaceChildren(...entries.map((source) => {
      const card = el('article', 'source-card');
      card.id = `source-${encodeURIComponent(String(source.id))}`;
      const heading = el('div', 'source-heading');
      heading.append(el('span', 'source-id', source.id));
      const title = el('h3');
      const url = safeUrl(source.url);
      if (url) {
        const link = el('a', null, `${source.title || new URL(url).hostname} ↗`);
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        title.append(link);
      } else title.textContent = source.title || 'Supplied vehicle record';
      heading.append(title);
      card.append(heading);
      if (url) card.append(el('p', 'source-domain', new URL(url).hostname));
      if (source.excerpt) card.append(el('p', 'source-excerpt', readable(source.excerpt)));
      if (source.kind || source.retrieved_at) card.append(el('span', 'source-kind', [source.kind?.replaceAll('_', ' '), source.retrieved_at ? `Retrieved ${timeLabel(source.retrieved_at, false)}` : ''].filter(Boolean).join(' · ')));
      return card;
    }));
  }

  function renderRun(run, replay = true) {
    state.run = run;
    $('empty-report').hidden = true;
    $('active-report').hidden = false;
    $('run-plate').textContent = [run.plate, run.jurisdiction].filter(Boolean).join(' / ');
    $('run-engine').textContent = [providerLabel(run.provider), run.model_id].filter(Boolean).join(' · ');
    $('export-markdown').href = `/api/runs/${encodeURIComponent(run.id)}/export?format=markdown`;
    $('export-json').href = `/api/runs/${encodeURIComponent(run.id)}/export?format=json`;
    renderStatus(statusOf(run));
    renderReport(run.report);
    renderSources(run.sources);
    showAlert('run-error', run.error ? readable(run.error) : '');
    if (replay && Array.isArray(run.events)) run.events.forEach((event) => appendEvent(event, false));
  }

  function eventDisplay(event) {
    const data = event.data;
    if (event.type === 'tool_start') return { label: data?.name || 'TOOL', text: data?.arguments?.command ? `$ ${data.arguments.command}` : data?.arguments?.query || data?.arguments?.url || 'Starting tool', detail: data?.arguments };
    if (event.type === 'tool_result') return { label: data?.name || 'RESULT', text: data?.result?.error ? readable(data.result.error) : data?.result?.summary || data?.result?.title || 'Tool returned results', detail: data?.result };
    if (event.type === 'report') return { label: 'BRIEF READY', text: 'Research brief assembled with findings and source references.' };
    if (event.type === 'status') return { label: 'STATUS', text: data?.message || data?.status || readable(data) };
    if (event.type === 'error') return { label: 'ATTENTION', text: data?.message || data?.error || readable(data) };
    if (event.type === 'done') return { label: 'FINISHED', text: data?.message || data?.status || 'Research run finished.' };
    return { label: event.type === 'note' ? 'AGENT' : String(event.type || 'ACTIVITY').replaceAll('_', ' '), text: data?.message || data?.text || readable(data) };
  }

  function appendEvent(event, live = true) {
    if (!event || typeof event !== 'object') return;
    const key = event.id !== undefined ? String(event.id) : `${event.type}:${event.time}:${readable(event.data)}`;
    if (state.eventIds.has(key)) return;
    state.eventIds.add(key);
    state.eventCount += 1;
    $('event-count').textContent = String(state.eventCount);
    const log = $('activity-log');
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 70;
    const display = eventDisplay(event);
    const row = el('div', `log-event ${event.type === 'error' ? 'error' : ''}`);
    row.append(el('time', 'event-time', timeLabel(event.time)));
    const body = el('div', 'event-body');
    body.append(el('span', 'event-label', String(display.label).toUpperCase()), el('span', null, readable(display.text)));
    if (display.detail !== undefined) {
      const details = el('details');
      details.append(el('summary', null, event.type === 'tool_start' ? 'View arguments' : 'View result'), el('pre', null, readable(display.detail)));
      body.append(details);
    }
    row.append(body);
    log.append(row);
    if (nearBottom || !live) log.scrollTop = log.scrollHeight;
    if (!live || !state.run) return;
    if (event.type === 'report') {
      state.run.report = event.data?.report || event.data;
      renderReport(state.run.report);
    }
    if (event.type === 'status') {
      const status = typeof event.data === 'string' ? event.data : event.data?.status;
      if (status) {
        state.run.status = status;
        renderStatus(status);
      }
    }
    if (event.type === 'error') showAlert('run-error', readable(event.data?.error || event.data?.message || event.data));
    if (event.type === 'tool_result' && !state.sourceRefresh) {
      state.sourceRefresh = setTimeout(() => { state.sourceRefresh = null; refreshCurrentRun(); }, 500);
    }
    if (event.type === 'done') {
      closeStream();
      refreshCurrentRun().then(() => loadRuns());
    }
  }

  async function refreshCurrentRun() {
    const id = state.run?.id;
    const selection = state.selection;
    if (!id) return;
    try {
      const run = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (selection !== state.selection || state.run?.id !== id) return;
      renderRun(run);
      if (FINAL.has(statusOf(run))) closeStream();
      setConnection(true);
    } catch (error) {
      if (selection === state.selection) showAlert('run-error', `Unable to refresh this run. ${error.message}`);
    }
  }

  function beginPolling(id, selection) {
    if (state.poll) return;
    state.poll = setInterval(() => {
      if (state.run?.id !== id || selection !== state.selection) return;
      refreshCurrentRun();
    }, 3000);
  }

  function connectEvents(id, selection) {
    if (!window.EventSource) { beginPolling(id, selection); return; }
    const stream = new EventSource(`/api/runs/${encodeURIComponent(id)}/events`);
    state.stream = stream;
    for (const type of ['status', 'tool_start', 'tool_result', 'note', 'report', 'error', 'done']) {
      stream.addEventListener(type, (message) => {
        if (selection !== state.selection || state.run?.id !== id || !message.data) return;
        try { appendEvent(JSON.parse(message.data)); }
        catch { /* A malformed event must not break subsequent events. */ }
      });
    }
    stream.onopen = () => {
      if (selection !== state.selection) return;
      clearInterval(state.poll);
      state.poll = null;
      setConnection(true);
    };
    stream.onerror = (event) => {
      if (event.data || selection !== state.selection) return;
      if (FINAL.has(statusOf(state.run))) { closeStream(); return; }
      beginPolling(id, selection);
    };
  }

  async function selectRun(id, provisional) {
    closeStream();
    const selection = ++state.selection;
    resetEvents();
    if (provisional) renderRun(provisional);
    showAlert('global-alert', '');
    try {
      const run = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (selection !== state.selection) return;
      renderRun(run);
      if (!FINAL.has(statusOf(run))) connectEvents(id, selection);
      document.querySelectorAll('.history-row').forEach((row) => row.classList.toggle('selected', row.dataset.id === id));
    } catch (error) {
      if (selection === state.selection) {
        showAlert('global-alert', `Unable to open this research run. ${error.message}`);
        if (provisional) connectEvents(id, selection);
      }
    }
  }

  async function loadRuns() {
    const request = ++state.runListRequest;
    try {
      const data = await api('/api/runs');
      if (request !== state.runListRequest) return;
      const runs = Array.isArray(data.runs) ? data.runs : [];
      const list = $('run-list');
      if (!runs.length) { list.replaceChildren(el('p', 'history-empty', 'Your research history will appear here.')); return; }
      list.replaceChildren(...runs.map((run) => {
        const row = el('button', `history-row${state.run?.id === run.id ? ' selected' : ''}`);
        row.type = 'button';
        row.dataset.id = run.id;
        row.setAttribute('aria-label', `Open research for ${run.plate}, ${run.jurisdiction}, ${run.status}`);
        const vehicle = el('span', 'vehicle-history', run.plate || 'Vehicle research');
        vehicle.append(el('small', null, run.jurisdiction));
        const status = statusOf(run);
        row.append(vehicle, el('span', null, providerLabel(run.provider)), el('span', 'history-time', timeLabel(run.created_at, false)), el('span', `history-status status-badge ${/^[a-z]+$/.test(status) ? status : 'idle'}`, status));
        row.addEventListener('click', () => {
          selectRun(run.id);
          $('report-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        return row;
      }));
    } catch (error) {
      if (request === state.runListRequest) $('run-list').replaceChildren(el('p', 'history-empty', `History is unavailable. ${error.message}`));
    }
  }

  async function stopRun() {
    const id = state.run?.id;
    if (!id) return;
    $('stop-button').disabled = true;
    try {
      await api(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST', body: '{}' });
      if (state.run?.id !== id) return;
      state.run.status = 'cancelling';
      renderStatus('cancelling');
      await refreshCurrentRun();
      loadRuns();
    } catch (error) {
      if (state.run?.id === id) { showAlert('run-error', `Unable to stop the run. ${error.message}`); setBusy(); }
    }
  }

  function fillDemo() {
    showAlert('form-error', '');
    const demo = state.providers.find((p) => isDemo(p.id));
    if (state.providers.length && !demo) {
      showAlert('form-error', 'The sample research engine is not enabled on this server. Choose an available engine to start a live run.');
      return;
    }
    $('provider').value = demo?.id || 'demo';
    updateProvider();
    $('plate').value = 'DEMO123';
    $('jurisdiction').value = 'Example jurisdiction';
    $('objective').value = 'Find publicly available vehicle details, specifications, and recall information.';
    $('make').value = 'Example Motors';
    $('vehicle-model').value = 'Touring';
    $('year').value = '2020';
    $('vin').value = '';
    $('source-urls').value = '';
    $('records').value = JSON.stringify([{ plate: 'DEMO123', jurisdiction: 'Example jurisdiction', make: 'Example Motors', model: 'Touring', year: 2020 }], null, 2);
    $('purpose').value = 'authorized_dataset';
    $('authorized').checked = true;
    $('enable-terminal').checked = false;
    $('research-form').requestSubmit();
  }

  $('research-form').addEventListener('submit', startRun);
  $('provider').addEventListener('change', () => updateProvider());
  $('refresh-models').addEventListener('click', loadModels);
  $('refresh-runs').addEventListener('click', loadRuns);
  $('demo-button').addEventListener('click', fillDemo);
  $('stop-button').addEventListener('click', stopRun);
  window.addEventListener('beforeunload', closeStream);
  Promise.allSettled([configure(), loadRuns()]);
})();
