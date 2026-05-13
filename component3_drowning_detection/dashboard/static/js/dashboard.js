// ═══════════════════════════════════════════════
// AI POOL MONITORING — DASHBOARD JAVASCRIPT
// ═══════════════════════════════════════════════

const socket = io();
let isRunning = false;

// ── CONNECTION ───────────────────────────────
socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    console.log('Disconnected');
    updateConnectionStatus(false);
});

// ── INITIAL STATE ────────────────────────────
socket.on('initial_state', (state) => {
    updateDashboard(state);
});

// ── FRAME UPDATE ─────────────────────────────
socket.on('frame_update', (data) => {
    // Update camera feed
    const feed = document.getElementById('cameraFeed');
    feed.src = 'data:image/jpeg;base64,' + data.frame;

    // Hide overlay when feed starts
    const overlay = document.getElementById('cameraOverlay');
    overlay.style.display = 'none';

    // Update dashboard stats
    updateDashboard(data.state);
});

// ── DROWNING ALERT ───────────────────────────
socket.on('drowning_alert', (alert) => {
    showDrowningAlert(alert);
    addAlertToHistory(alert);
    playAlertSound();
});

// ── DETECTION STARTED/STOPPED ────────────────
socket.on('detection_started', () => {
    isRunning = true;
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    console.log('Detection started');
});

socket.on('detection_stopped', () => {
    isRunning = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('cameraOverlay').style.display = 'flex';
    console.log('Detection stopped');
});

socket.on('camera_error', (data) => {
    alert('Camera Error: ' + data.message +
          '\n\nMake sure IP Webcam is running on your phone.');
});

// ═══════════════════════════════════════════════
// UPDATE FUNCTIONS
// ═══════════════════════════════════════════════

function updateDashboard(state) {
    // Counts
    setText('swimmingCount', state.swimming);
    setText('drowningCount', state.drowning);
    setText('outWaterCount', state.out_of_water);
    setText('totalAlerts',   state.total_alerts);
    setText('fpsDisplay',    state.fps);
    setText('sessionTime',   state.session_time);
    setText('frameNum',      state.frame_num);

    // Status
    updateStatus(state.status, state.drowning);

    // Drowning stat highlight
    const drowningCard = document.querySelector(
        '.stat-item.drowning');
    if (state.drowning > 0) {
        drowningCard.classList.add('active');
    } else {
        drowningCard.classList.remove('active');
    }

    // Alert overlay
    const alertOverlay = document.getElementById(
        'alertOverlay');
    alertOverlay.style.display =
        state.drowning > 0 ? 'block' : 'none';
}

function updateStatus(status, drowningCount) {
    const card = document.getElementById('statusCard');
    const icon = document.getElementById('statusIcon');
    const text = document.getElementById('statusText');

    if (status === 'DANGER' || drowningCount > 0) {
        card.className = 'card status-card danger';
        icon.textContent = '⚠️';
        text.textContent = 'DANGER';
        text.className = 'status-text danger';
    } else {
        card.className = 'card status-card safe';
        icon.textContent = '✓';
        text.textContent = 'SAFE';
        text.className = 'status-text';
    }
}

function updateConnectionStatus(connected) {
    const dot  = document.querySelector('.status-dot');
    const text = document.getElementById('connectionText');

    if (connected) {
        dot.className  = 'status-dot connected';
        text.textContent = 'Connected';
        text.style.color = '#00ff88';
    } else {
        dot.className  = 'status-dot disconnected';
        text.textContent = 'Disconnected';
        text.style.color = '#888';
    }
}

function showDrowningAlert(alert) {
    // Flash page border
    document.body.style.outline = '4px solid red';
    setTimeout(() => {
        document.body.style.outline = 'none';
    }, 2000);
}

function addAlertToHistory(alert) {
    const list = document.getElementById('alertsList');

    // Remove no-alerts message
    const noAlerts = list.querySelector('.no-alerts');
    if (noAlerts) noAlerts.remove();

    // Create alert item
    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `
        <div class="alert-item-left">
            <span class="alert-icon">🚨</span>
            <span class="alert-time">${alert.time}</span>
            <span class="alert-desc">
                Drowning detected in pool
            </span>
        </div>
        <span class="alert-badge">
            ${alert.count} person(s)
        </span>
    `;

    list.insertBefore(item, list.firstChild);
}

function clearAlerts() {
    const list = document.getElementById('alertsList');
    list.innerHTML = `
        <div class="no-alerts">
            ✓ No drowning alerts in current session
        </div>
    `;
}

function playAlertSound() {
    try {
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880;
        osc.type = 'square';
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(
            0.001, ctx.currentTime + 0.5);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.5);
    } catch(e) {}
}

// ═══════════════════════════════════════════════
// CONTROL FUNCTIONS
// ═══════════════════════════════════════════════

function startDetection() {
    socket.emit('start_detection');
    document.getElementById('streamUrl').textContent =
        window.location.hostname === 'localhost'
        ? 'Connecting...'
        : 'Connecting...';
}

function stopDetection() {
    socket.emit('stop_detection');
}

// ═══════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════
function updateClock() {
    const now = new Date();
    document.getElementById('currentTime').textContent =
        now.toLocaleTimeString('en-US', {
            hour:   '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
}

setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}