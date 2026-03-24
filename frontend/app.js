/**
 * app.js — Frontend client for Real-Time Speech Transcription (v2).
 *
 * Handles:
 *   - Dynamic model dropdown populated from /config API
 *   - Dynamic language dropdown based on selected model's supported languages
 *   - Microphone capture via getUserMedia (float32 PCM, 16kHz mono)
 *   - WebSocket streaming to the FastAPI backend
 *   - Real-time transcript display and audio visualizer
 */

// ── DOM Elements ───────────────────────────────────────────────────────
const DOM = {
    // Settings
    nsSelect: document.getElementById('nsSelect'),
    vadSelect: document.getElementById('vadSelect'),
    sttSelect: document.getElementById('sttSelect'),
    langSelect: document.getElementById('langSelect'),
    chunkInput: document.getElementById('chunkInput'),
    settingsPanel: document.getElementById('settingsPanel'),

    // Controls
    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    btnClear: document.getElementById('btnClear'),

    // Transcript
    transcriptText: document.getElementById('transcriptText'),
    placeholder: document.getElementById('transcriptPlaceholder'),

    // Status bar
    statLatency: document.getElementById('statLatency'),
    statChunk: document.getElementById('statChunk'),
    statNS: document.getElementById('statNS'),
    statVAD: document.getElementById('statVAD'),
    statSTT: document.getElementById('statSTT'),
    statLang: document.getElementById('statLang'),
    statDevice: document.getElementById('statDevice'),

    // Connection
    badge: document.getElementById('connectionBadge'),
    badgeText: document.querySelector('.badge-text'),
    logoIcon: document.querySelector('.logo-icon'),

    // Visualizer
    canvas: document.getElementById('visualizerCanvas'),
};

// ── State ──────────────────────────────────────────────────────────────
let ws = null;
let mediaStream = null;
let audioContext = null;
let processorNode = null;
let analyserNode = null;
let isStreaming = false;
let animationFrameId = null;

const TARGET_SAMPLE_RATE = 16000;

function getChunkDurationMs() {
    const val = parseInt(DOM.chunkInput.value, 10);
    return Math.min(5000, Math.max(250, isNaN(val) ? 1000 : val));
}

// Server configuration (fetched at startup)
let serverConfig = null;

// ── Initialization ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await fetchServerConfig();
    setupEventListeners();
    setupVisualizer();
});

/**
 * Fetch configuration from the server and populate UI dropdowns.
 */
async function fetchServerConfig() {
    try {
        const resp = await fetch('/config');
        serverConfig = await resp.json();

        // Update device badge
        const device = serverConfig.device || 'cpu';
        DOM.statDevice.textContent = device.toUpperCase();
        DOM.statDevice.classList.toggle('gpu', device === 'cuda');
        DOM.statDevice.classList.toggle('cpu', device === 'cpu');

        // Check DeepFilterNet availability
        if (!serverConfig.deepfilter_available) {
            // Remove DeepFilterNet3 option, keep only None
            const nsOpt = DOM.nsSelect.querySelector('option[value="deepfilter"]');
            if (nsOpt) nsOpt.remove();
        }

        // Populate STT model dropdown
        populateSTTModels(serverConfig.stt_models, serverConfig.default_stt_model);

        // Populate language dropdown based on default model
        updateLanguageDropdown();

        console.log('Server config loaded:', serverConfig);
    } catch (err) {
        console.warn('Could not fetch server config:', err);
    }
}

/**
 * Populate the STT model dropdown with optgroups.
 */
function populateSTTModels(sttModels, defaultModel) {
    const select = DOM.sttSelect;
    select.innerHTML = '';

    // Group models by their group name
    const groups = {};
    for (const [key, model] of Object.entries(sttModels)) {
        const group = model.group || 'Other';
        if (!groups[group]) groups[group] = [];
        groups[group].push({ key, ...model });
    }

    // Create optgroups
    for (const [groupName, models] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = groupName;

        for (const model of models) {
            const option = document.createElement('option');
            option.value = model.key;
            option.textContent = model.label;
            if (model.key === defaultModel) option.selected = true;
            optgroup.appendChild(option);
        }

        select.appendChild(optgroup);
    }
}

/**
 * Update the language dropdown based on the selected STT model's supported languages.
 */
function updateLanguageDropdown() {
    if (!serverConfig) return;

    const modelKey = DOM.sttSelect.value;
    const modelConfig = serverConfig.stt_models[modelKey];
    if (!modelConfig) return;

    const allLanguages = serverConfig.all_languages;
    const supportedLangs = modelConfig.languages;
    const select = DOM.langSelect;

    // Remember current selection
    const currentLang = select.value;
    select.innerHTML = '';

    if (supportedLangs === 'all') {
        // Show all languages
        for (const [code, name] of Object.entries(allLanguages)) {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            select.appendChild(option);
        }
    } else {
        // Show only supported languages
        for (const code of supportedLangs) {
            const name = allLanguages[code] || code;
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            select.appendChild(option);
        }
    }

    // Restore previous selection if still available, otherwise pick first
    if ([...select.options].some(o => o.value === currentLang)) {
        select.value = currentLang;
    } else {
        select.selectedIndex = 0;
    }

    updateStatusBar();
}

/**
 * Wire up all user interactions.
 */
function setupEventListeners() {
    DOM.btnStart.addEventListener('click', startStreaming);
    DOM.btnStop.addEventListener('click', stopStreaming);
    DOM.btnClear.addEventListener('click', clearTranscript);

    // Update status bar when settings change
    DOM.nsSelect.addEventListener('change', updateStatusBar);
    DOM.vadSelect.addEventListener('change', updateStatusBar);
    DOM.langSelect.addEventListener('change', updateStatusBar);
    DOM.chunkInput.addEventListener('input', updateStatusBar);
    DOM.sttSelect.addEventListener('change', () => {
        updateLanguageDropdown();
        updateStatusBar();
    });

    updateStatusBar();
}

// ── Settings Helpers ──────────────────────────────────────────────────

function getSettings() {
    return {
        ns_enabled: DOM.nsSelect.value !== 'none',
        vad_model: DOM.vadSelect.value,
        stt_model: DOM.sttSelect.value,
        language: DOM.langSelect.value,
    };
}

function updateStatusBar() {
    const s = getSettings();
    DOM.statNS.textContent = DOM.nsSelect.value === 'none' ? 'OFF' : 'DeepFilter';
    DOM.statVAD.textContent = s.vad_model;
    DOM.statSTT.textContent = s.stt_model;
    DOM.statLang.textContent = s.language;
    DOM.statChunk.textContent = `${getChunkDurationMs()} ms`;
}

// ── Streaming Control ─────────────────────────────────────────────

function disableSettings(disabled) {
    DOM.nsSelect.disabled = disabled;
    DOM.vadSelect.disabled = disabled;
    DOM.sttSelect.disabled = disabled;
    DOM.langSelect.disabled = disabled;
    DOM.chunkInput.disabled = disabled;
}

async function startStreaming() {
    if (isStreaming) return;

    try {
        // 1. Get microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: { ideal: TARGET_SAMPLE_RATE },
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: true,
            }
        });

        // 2. Create audio processing pipeline
        audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
        const source = audioContext.createMediaStreamSource(mediaStream);

        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);

        const bufferSize = 4096;
        processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

        let audioBuffer = [];
        const chunkMs = getChunkDurationMs();
        const samplesPerChunk = Math.floor(TARGET_SAMPLE_RATE * chunkMs / 1000);

        processorNode.onaudioprocess = (event) => {
            if (!isStreaming) return;
            const inputData = event.inputBuffer.getChannelData(0);
            audioBuffer.push(...inputData);

            while (audioBuffer.length >= samplesPerChunk) {
                const chunk = new Float32Array(audioBuffer.splice(0, samplesPerChunk));
                sendAudioChunk(chunk);
            }
        };

        source.connect(processorNode);
        processorNode.connect(audioContext.destination);

        // 3. Open WebSocket
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

        ws.onopen = () => {
            const settings = getSettings();
            ws.send(JSON.stringify(settings));
            console.log('WebSocket connected, config sent:', settings);
        };

        ws.onmessage = (event) => {
            handleServerMessage(JSON.parse(event.data));
        };

        ws.onerror = (err) => console.error('WebSocket error:', err);
        ws.onclose = () => {
            console.log('WebSocket closed.');
            if (isStreaming) stopStreaming();
        };

        // 4. Update UI
        isStreaming = true;
        DOM.btnStart.disabled = true;
        DOM.btnStop.disabled = false;
        disableSettings(true);
        setConnectionStatus(true);
        DOM.placeholder.classList.add('hidden');
        DOM.logoIcon.classList.add('recording');
        startVisualizer();

    } catch (err) {
        console.error('Failed to start streaming:', err);
        if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') {
            alert('Could not access microphone. Please allow microphone permission.');
        } else {
            alert(`Error starting stream: ${err.message}`);
        }
    }
}

function stopStreaming() {
    isStreaming = false;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('STOP');
        ws.close();
    }
    ws = null;

    if (processorNode) { processorNode.disconnect(); processorNode = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

    DOM.btnStart.disabled = false;
    DOM.btnStop.disabled = true;
    disableSettings(false);
    setConnectionStatus(false);
    DOM.logoIcon.classList.remove('recording');
    stopVisualizer();
}

function sendAudioChunk(chunk) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(chunk.buffer);
    }
}

function handleServerMessage(msg) {
    if (msg.type === 'config_ack') {
        console.log('Pipeline configured:', msg.status);
        return;
    }
    if (msg.type === 'status') {
        console.log('Server status:', msg.message);
        // Show loading message in transcript area
        const statusDiv = document.createElement('div');
        statusDiv.className = 'line status-line';
        statusDiv.textContent = `⏳ ${msg.message}`;
        statusDiv.style.opacity = '0.5';
        statusDiv.style.fontStyle = 'italic';
        DOM.transcriptText.appendChild(statusDiv);
        return;
    }
    if (msg.type === 'transcription') {
        DOM.statLatency.textContent = `${msg.latency} ms`;
        if (msg.text && msg.text.trim()) {
            appendTranscript(msg.text.trim());
        }
    }
}

// ── Transcript ────────────────────────────────────────────────────────

function appendTranscript(text) {
    const line = document.createElement('div');
    line.className = 'line';
    line.textContent = text;
    DOM.transcriptText.appendChild(line);
    const container = DOM.transcriptText.parentElement;
    container.scrollTop = container.scrollHeight;
}

function clearTranscript() {
    DOM.transcriptText.innerHTML = '';
    DOM.placeholder.classList.remove('hidden');
}

// ── UI Helpers ────────────────────────────────────────────────────────

function setConnectionStatus(connected) {
    DOM.badge.classList.toggle('connected', connected);
    DOM.badgeText.textContent = connected ? 'Streaming' : 'Disconnected';
}



// ── Audio Visualizer ──────────────────────────────────────────────────

function setupVisualizer() {
    const canvas = DOM.canvas;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    drawIdleVisualizer(ctx, rect.width, rect.height);

    window.addEventListener('resize', () => {
        const r = canvas.parentElement.getBoundingClientRect();
        canvas.width = r.width * dpr;
        canvas.height = r.height * dpr;
        ctx.scale(dpr, dpr);
        if (!isStreaming) drawIdleVisualizer(ctx, r.width, r.height);
    });
}

function drawIdleVisualizer(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(124, 100, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
}

function startVisualizer() {
    if (!analyserNode) return;

    const canvas = DOM.canvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!isStreaming) return;
        animationFrameId = requestAnimationFrame(draw);
        analyserNode.getByteFrequencyData(dataArray);

        const rect = canvas.parentElement.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        ctx.clearRect(0, 0, w, h);

        const barCount = Math.min(bufferLength, 80);
        const barWidth = w / barCount;
        const gap = 2;

        for (let i = 0; i < barCount; i++) {
            const value = dataArray[i] / 255;
            const barHeight = value * h * 0.85;
            const hue = 255 - value * 40;
            ctx.fillStyle = `hsla(${hue}, 70%, 65%, ${0.4 + value * 0.6})`;
            ctx.fillRect(i * barWidth + gap / 2, (h - barHeight) / 2, barWidth - gap, barHeight || 1);
        }
    }

    draw();
}

function stopVisualizer() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    const canvas = DOM.canvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    drawIdleVisualizer(ctx, rect.width, rect.height);
}
