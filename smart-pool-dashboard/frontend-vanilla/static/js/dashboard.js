// ═══════════════════════════════════════════════════════════
// R26-IT-143 — Unified Dashboard
// Socket patterns preserved from Component 3's dashboard.js,
// extended to update all four module cards from one state object.
// ═══════════════════════════════════════════════════════════

const socket = io();

// ── Connection ──────────────────────────────────────────────
socket.on('connect', () => setConn(true));
socket.on('disconnect', () => setConn(false));

function setConn(ok) {
  document.getElementById('connDot').className = 'status-dot ' + (ok ? 'connected' : 'disconnected');
  document.getElementById('connText').textContent = ok ? 'Connected' : 'Disconnected';
}

// ── Controls ────────────────────────────────────────────────
function startDetection() { socket.emit('start_detection'); }
function stopDetection()  { socket.emit('stop_detection'); }

// Client-side session-stats tracking (peak swimmers / alerts this
// session). The backend already tracks a fuller session summary for
// the PDF report (see /api/session/summary), but that isn't part of
// the live state broadcast — these two small counters just mirror
// what's visible in the state updates this page has already received,
// reset each time a new session starts.
let peakSwimmers = 0;
let sessionAlertCount = 0;
let lastSeenAlertKey; // undefined = no state observed yet this session

socket.on('detection_started', () => {
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  peakSwimmers = 0;
  sessionAlertCount = 0;
  lastSeenAlertKey = undefined;
  setText('peakSwimmers', 0);
  setText('sessionAlerts', 0);
});
socket.on('detection_stopped', () => {
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
});
socket.on('camera_error', (d) => {
  const ov = document.getElementById('cameraOverlay');
  ov.style.display = 'flex';
  ov.textContent = d.message;
});

// ── Frame + state updates ───────────────────────────────────
socket.on('frame_update', (data) => {
  const feed = document.getElementById('cameraFeed');
  feed.src = 'data:image/jpeg;base64,' + data.frame;
  document.getElementById('cameraOverlay').style.display = 'none';
  updateDashboard(data.state);
});
socket.on('state_update', updateDashboard);

socket.on('drowning_alert', () => {
  flashCard('drowningCard');
  playBeep();
});

// ── Renderers ───────────────────────────────────────────────
function updateDashboard(s) {
  if (!s) return;
  setText('fps', s.fps);
  setText('frameNum', s.frame_num);
  setText('sessionTime', s.session_time);

  // Session stats strip (client-side running totals, reset on start)
  if (typeof s.crowd.swimmer_count === 'number') {
    peakSwimmers = Math.max(peakSwimmers, s.crowd.swimmer_count);
  }
  setText('peakSwimmers', peakSwimmers);
  const currentTopAlertKey = s.alerts.length
    ? s.alerts[0].time + '|' + s.alerts[0].module + '|' + s.alerts[0].message
    : null;
  if (lastSeenAlertKey === undefined) {
    // First observation this session: establish a baseline without
    // counting it (avoids counting alerts that already existed before
    // this page/session started).
    lastSeenAlertKey = currentTopAlertKey;
  } else if (currentTopAlertKey !== null && currentTopAlertKey !== lastSeenAlertKey) {
    sessionAlertCount++;
    lastSeenAlertKey = currentTopAlertKey;
  }
  setText('sessionAlerts', sessionAlertCount);

  // Crowd
  setText('swimmerCount', s.crowd.swimmer_count);
  setText('densityLevel', s.crowd.density_level);
  setText('batherLoad', s.crowd.bather_load);
  setText('maintRec', s.crowd.maintenance_recommendation);
  setText('crowdModel', s.crowd.model_status);

  // Drowning
  setText('swimming', s.drowning.swimming);
  setText('drowningCount', s.drowning.drowning);
  setText('outOfWater', s.drowning.out_of_water);
  setText('totalDrownAlerts', s.drowning.total_alerts);
  setText('drownModel', s.drowning.model_status);
  setPill('drownStatus', s.drowning.status, s.drowning.status === 'SAFE' ? 'safe' : 'danger');

  // Garbage
  setText('garbageCount', s.garbage.objects_detected);
  setText('garbageStatus', s.garbage.alert_status);
  setText('garbageEvents', s.garbage.total_events);
  setText('garbageModel', s.garbage.model_status);
  setText('garbageLabels', s.garbage.object_labels.length
    ? s.garbage.object_labels.join(', ') : 'No objects detected');

  // Garbage intent (source + throwing-motion intent classification)
  const gRisk = s.garbage.risk_level || 'NORMAL';
  setPill('garbageRisk', gRisk,
    gRisk === 'HIGH' ? 'danger' : (gRisk === 'MEDIUM' ? 'warn' : 'safe'));
  if (s.garbage.alert_status === 'ALERT') {
    setText('garbageSource', 'Source: ' + s.garbage.source);
    setText('garbageIntent',
      ' · Intent: ' + s.garbage.intent + ' (' + s.garbage.intent_confidence + '%)');
  } else {
    setText('garbageSource', 'No intrusion detected');
    setText('garbageIntent', '');
  }

  // Water quality
  setText('ph', fmt(s.water_quality.ph));
  setText('temperature', fmt(s.water_quality.temperature));
  setText('chlorine', fmt(s.water_quality.chlorine));
  setText('turbidity', fmt(s.water_quality.turbidity));
  setText('tds', fmt(s.water_quality.tds));
  setText('sensorConn', s.water_quality.sensor_connected ? 'connected' : 'offline');
  setText('waterModel', s.water_quality.model_status);
  const w = s.water_quality.status;
  setPill('waterStatus', w, w === 'SAFE' ? 'safe' : (w === 'CRITICAL' ? 'danger' : 'warn'));

  // Decision engine
  const risk = document.getElementById('overallRisk');
  risk.textContent = s.decision.overall_risk;
  risk.className = 'risk-badge ' + s.decision.overall_risk.toLowerCase();
  setPill('maintUrgency', s.decision.maintenance_urgency,
    s.decision.maintenance_urgency === 'LOW' ? 'safe'
      : (s.decision.maintenance_urgency === 'IMMEDIATE' ? 'danger' : 'warn'));
  const notes = document.getElementById('decisionNotes');
  notes.innerHTML = s.decision.notes.length
    ? s.decision.notes.map(n => `<li>${n}</li>`).join('')
    : '<li class="muted">No cross-module concerns</li>';

  // Alerts
  const list = document.getElementById('alertList');
  list.innerHTML = s.alerts.length
    ? s.alerts.map(a =>
        `<li class="alert-item ${a.severity}"><b>${a.time}</b> · ${a.module}: ${a.message}</li>`
      ).join('')
    : '<li class="muted">No alerts yet</li>';
}

// ── PDF report ──────────────────────────────────────────────
function generateReport() {
  const btn = document.getElementById('reportBtn');
  if (!btn) return;
  const label = btn.querySelector('.btn-label');
  const err = document.getElementById('reportError');

  btn.disabled = true;
  if (label) label.textContent = 'Generating…';
  if (err) { err.textContent = ''; err.classList.remove('show'); }

  fetch('/api/report')
    .then(async (resp) => {
      if (!resp.ok) {
        let msg = 'Report generation failed';
        try {
          const data = await resp.json();
          if (data && data.error) msg = data.error;
        } catch (e) { /* keep default message */ }
        throw new Error(msg);
      }
      const cd = resp.headers.get('Content-Disposition') || '';
      const match = /filename="?([^";]+)"?/.exec(cd);
      const filename = match ? match[1] : 'pool_session_report.pdf';
      const blob = await resp.blob();
      return { blob, filename };
    })
    .then(({ blob, filename }) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    })
    .catch((e) => {
      if (err) {
        err.textContent = e.message || 'Report generation failed';
        err.classList.add('show');
      }
    })
    .finally(() => {
      btn.disabled = false;
      if (label) label.textContent = 'Generate Report';
    });
}

// ── Helpers ─────────────────────────────────────────────────
function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function setPill(id, text, cls) {
  const e = document.getElementById(id);
  if (e) { e.textContent = text; e.className = 'pill ' + cls; }
}
function fmt(v) { return (v === null || v === undefined) ? '–' : v; }
function flashCard(id) {
  const c = document.getElementById(id);
  c.classList.add('flash');
  setTimeout(() => c.classList.remove('flash'), 1500);
}
function playBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    o.frequency.value = 1000;
    o.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.4);
  } catch (e) { /* audio not available */ }
}
