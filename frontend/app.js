/**
 * app.js - Frontend client for real-time speech transcription.
 *
 * Handles:
 *   - Dynamic model dropdowns populated from /config
 *   - STT model gating when pyannote diarization is selected
 *   - Microphone capture via getUserMedia (float32 PCM, 16kHz mono)
 *   - WebSocket streaming to the FastAPI backend
 *   - Transcript upserts and late speaker label updates
 *   - Audio visualizer and live status bar
 */

const DOM = {
    nsSelect: document.getElementById('nsSelect'),
    vadSelect: document.getElementById('vadSelect'),
    sttSelect: document.getElementById('sttSelect'),
    diarizationSelect: document.getElementById('diarizationSelect'),
    diarizationHint: document.getElementById('diarizationHint'),
    langSelect: document.getElementById('langSelect'),
    chunkInput: document.getElementById('chunkInput'),

    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    btnClear: document.getElementById('btnClear'),

    transcriptText: document.getElementById('transcriptText'),
    placeholder: document.getElementById('transcriptPlaceholder'),

    statLatency: document.getElementById('statLatency'),
    statChunk: document.getElementById('statChunk'),
    statNS: document.getElementById('statNS'),
    statVAD: document.getElementById('statVAD'),
    statSTT: document.getElementById('statSTT'),
    statSD: document.getElementById('statSD'),
    statLang: document.getElementById('statLang'),
    statDevice: document.getElementById('statDevice'),

    badge: document.getElementById('connectionBadge'),
    badgeText: document.querySelector('.badge-text'),
    logoIcon: document.querySelector('.logo-icon'),

    canvas: document.getElementById('visualizerCanvas'),
};

let ws = null;
let mediaStream = null;
let audioContext = null;
let processorNode = null;
let analyserNode = null;
let isStreaming = false;
let animationFrameId = null;
let serverConfig = null;

const transcriptNodes = new Map();
const TARGET_SAMPLE_RATE = 16000;

function getChunkDurationMs() {
    const value = parseInt(DOM.chunkInput.value, 10);
    return Math.min(5000, Math.max(250, Number.isNaN(value) ? 1000 : value));
}

document.addEventListener('DOMContentLoaded', async () => {
    await fetchServerConfig();
    setupEventListeners();
    setupVisualizer();
});

async function fetchServerConfig() {
    try {
        const response = await fetch('/config');
        serverConfig = await response.json();

        const device = serverConfig.device || 'cpu';
        DOM.statDevice.textContent = device.toUpperCase();
        DOM.statDevice.classList.toggle('gpu', device === 'cuda');
        DOM.statDevice.classList.toggle('cpu', device === 'cpu');

        if (!serverConfig.deepfilter_available) {
            const option = DOM.nsSelect.querySelector('option[value="deepfilter"]');
            if (option) option.remove();
        }

        populateDiarizationModels(
            serverConfig.diarization_models || {},
            serverConfig.default_diarization_model || 'none',
        );
        populateSTTModels(
            serverConfig.stt_models || {},
            serverConfig.default_stt_model,
        );
        applyDiarizationModelConstraints();
        updateLanguageDropdown();
        updateStatusBar();

        console.log('Server config loaded:', serverConfig);
    } catch (err) {
        console.warn('Could not fetch server config:', err);
    }
}

function populateDiarizationModels(models, defaultModel) {
    DOM.diarizationSelect.innerHTML = '';

    for (const [key, model] of Object.entries(models)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = model.label || key;
        option.disabled = model.available === false;
        if (key === defaultModel && !option.disabled) {
            option.selected = true;
        }
        DOM.diarizationSelect.appendChild(option);
    }

    if (DOM.diarizationSelect.selectedIndex < 0 && DOM.diarizationSelect.options.length > 0) {
        DOM.diarizationSelect.selectedIndex = 0;
    }
}

function populateSTTModels(sttModels, defaultModel) {
    DOM.sttSelect.innerHTML = '';

    const groups = {};
    for (const [key, model] of Object.entries(sttModels)) {
        const group = model.group || 'Other';
        if (!groups[group]) groups[group] = [];
        groups[group].push({ key, ...model });
    }

    for (const [groupName, models] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = groupName;

        for (const model of models) {
            const option = document.createElement('option');
            option.value = model.key;
            option.textContent = model.label;
            option.dataset.diarizationSupported = String(!!model.diarization_supported);
            if (model.key === defaultModel) {
                option.selected = true;
            }
            optgroup.appendChild(option);
        }

        DOM.sttSelect.appendChild(optgroup);
    }
}

function applyDiarizationModelConstraints() {
    if (!serverConfig) return;

    const selectedModel = DOM.diarizationSelect.value;
    const modelInfo = serverConfig.diarization_models?.[selectedModel];
    if (selectedModel !== 'none' && modelInfo?.available === false) {
        DOM.diarizationSelect.value = 'none';
    }

    const diarizationModel = DOM.diarizationSelect.value;
    const diarizationActive = diarizationModel !== 'none';
    const sttOptions = [...DOM.sttSelect.querySelectorAll('option')];

    for (const option of sttOptions) {
        const supported = option.dataset.diarizationSupported === 'true';
        option.disabled = diarizationActive && !supported;
    }

    const selectedOption = DOM.sttSelect.selectedOptions[0];
    if (!selectedOption || selectedOption.disabled) {
        const firstEnabled = sttOptions.find(option => !option.disabled);
        if (firstEnabled) {
            DOM.sttSelect.value = firstEnabled.value;
        }
    }

    updateDiarizationHint();
}

function updateDiarizationHint() {
    const diarizationModel = DOM.diarizationSelect.value;
    if (!serverConfig) {
        DOM.diarizationHint.textContent = 'STT stays live while speaker labels resolve in the background.';
        return;
    }

    if (diarizationModel !== 'none') {
        const label = serverConfig.diarization_models?.[diarizationModel]?.label || diarizationModel;
        DOM.diarizationHint.textContent = `Only diarization-supported STT models are enabled. Speaker labels start as loading... and update later. (${label})`;
        return;
    }

    const anyAvailable = Object.entries(serverConfig.diarization_models || {})
        .some(([key, model]) => key !== 'none' && model.available !== false);

    if (!anyAvailable) {
        DOM.diarizationHint.textContent = 'No diarization backends available. Install pyannote.audio, nemo_toolkit, or speechbrain.';
        return;
    }

    DOM.diarizationHint.textContent = 'STT stays live while speaker labels resolve in the background.';
}

function updateLanguageDropdown() {
    if (!serverConfig) return;

    const modelKey = DOM.sttSelect.value;
    const modelConfig = serverConfig.stt_models?.[modelKey];
    if (!modelConfig) return;

    const allLanguages = serverConfig.all_languages || {};
    const supportedLangs = modelConfig.languages;
    const currentLang = DOM.langSelect.value;
    DOM.langSelect.innerHTML = '';

    if (supportedLangs === 'all') {
        for (const [code, name] of Object.entries(allLanguages)) {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            DOM.langSelect.appendChild(option);
        }
    } else {
        for (const code of supportedLangs) {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = allLanguages[code] || code;
            DOM.langSelect.appendChild(option);
        }
    }

    if ([...DOM.langSelect.options].some(option => option.value === currentLang)) {
        DOM.langSelect.value = currentLang;
    } else {
        DOM.langSelect.selectedIndex = 0;
    }
}

function setupEventListeners() {
    DOM.btnStart.addEventListener('click', startStreaming);
    DOM.btnStop.addEventListener('click', stopStreaming);
    DOM.btnClear.addEventListener('click', clearTranscript);

    DOM.nsSelect.addEventListener('change', updateStatusBar);
    DOM.vadSelect.addEventListener('change', updateStatusBar);
    DOM.langSelect.addEventListener('change', updateStatusBar);
    DOM.chunkInput.addEventListener('input', updateStatusBar);

    DOM.sttSelect.addEventListener('change', () => {
        updateLanguageDropdown();
        updateStatusBar();
    });

    DOM.diarizationSelect.addEventListener('change', () => {
        applyDiarizationModelConstraints();
        updateLanguageDropdown();
        updateStatusBar();
    });

    updateStatusBar();
}

function getSettings() {
    return {
        ns_enabled: DOM.nsSelect.value !== 'none',
        vad_model: DOM.vadSelect.value,
        stt_model: DOM.sttSelect.value,
        diarization_model: DOM.diarizationSelect.value,
        language: DOM.langSelect.value,
    };
}

function updateStatusBar(status = null) {
    const settings = status || getSettings();
    DOM.statNS.textContent = (settings.ns_enabled || DOM.nsSelect.value !== 'none') ? 'DeepFilter' : 'OFF';
    DOM.statVAD.textContent = settings.vad_model || DOM.vadSelect.value;
    DOM.statSTT.textContent = settings.stt_model || DOM.sttSelect.value;
    DOM.statSD.textContent = settings.diarization_model || DOM.diarizationSelect.value;
    DOM.statLang.textContent = settings.language || DOM.langSelect.value;
    DOM.statChunk.textContent = `${getChunkDurationMs()} ms`;
}

function applyServerStatus(status) {
    if (!status) return;

    if (status.stt_model && serverConfig?.stt_models?.[status.stt_model]) {
        DOM.sttSelect.value = status.stt_model;
    }
    if (status.language) {
        updateLanguageDropdown();
        if ([...DOM.langSelect.options].some(option => option.value === status.language)) {
            DOM.langSelect.value = status.language;
        }
    }
    if (status.vad_model) {
        DOM.vadSelect.value = status.vad_model;
    }
    if (typeof status.ns_enabled === 'boolean') {
        DOM.nsSelect.value = status.ns_enabled ? 'deepfilter' : 'none';
    }
    if (status.diarization_model) {
        DOM.diarizationSelect.value = status.diarization_model;
        applyDiarizationModelConstraints();
    }

    updateStatusBar(status);
}

function disableSettings(disabled) {
    DOM.nsSelect.disabled = disabled;
    DOM.vadSelect.disabled = disabled;
    DOM.sttSelect.disabled = disabled;
    DOM.diarizationSelect.disabled = disabled;
    DOM.langSelect.disabled = disabled;
    DOM.chunkInput.disabled = disabled;
}

async function startStreaming() {
    if (isStreaming) return;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: { ideal: TARGET_SAMPLE_RATE },
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: true,
            },
        });

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

        isStreaming = true;
        DOM.btnStart.disabled = true;
        DOM.btnStop.disabled = false;
        disableSettings(true);
        setConnectionStatus(true);
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

    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

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
        applyServerStatus(msg.status);
        return;
    }

    if (msg.type === 'config_update') {
        applyServerStatus(msg.status);
        return;
    }

    if (msg.type === 'status') {
        appendStatusLine(msg.message);
        return;
    }

    if (msg.type === 'transcription') {
        DOM.statLatency.textContent = `${msg.latency} ms`;
        if (Array.isArray(msg.items)) {
            msg.items.forEach(upsertTranscriptItem);
        } else if (msg.text && msg.text.trim()) {
            upsertTranscriptItem({
                id: `legacy_${Date.now()}`,
                text: msg.text.trim(),
                speaker: null,
                speaker_pending: false,
            });
        }
        return;
    }

    if (msg.type === 'speaker_update' && Array.isArray(msg.updates)) {
        msg.updates.forEach(applySpeakerUpdate);
    }
}

function createTranscriptNode(item) {
    const line = document.createElement('div');
    line.className = 'line transcript-line';
    line.dataset.id = item.id;

    const speaker = document.createElement('span');
    speaker.className = 'speaker-chip';

    const text = document.createElement('span');
    text.className = 'line-text';

    line.appendChild(speaker);
    line.appendChild(text);
    DOM.transcriptText.appendChild(line);

    transcriptNodes.set(item.id, { line, speaker, text });
    return transcriptNodes.get(item.id);
}

function upsertTranscriptItem(item) {
    if (!item || !item.id || !item.text) return;

    DOM.placeholder.classList.add('hidden');

    const node = transcriptNodes.get(item.id) || createTranscriptNode(item);
    node.text.textContent = item.text;
    renderSpeakerState(node.speaker, item.speaker, !!item.speaker_pending);

    const container = DOM.transcriptText.parentElement;
    container.scrollTop = container.scrollHeight;
}

function applySpeakerUpdate(update) {
    if (!update || !update.id) return;

    const node = transcriptNodes.get(update.id);
    if (!node) return;

    renderSpeakerState(node.speaker, update.speaker, !!update.speaker_pending);
}

function renderSpeakerState(speakerNode, speaker, pending) {
    if (!speaker && !pending) {
        speakerNode.textContent = '';
        speakerNode.classList.add('hidden');
        speakerNode.classList.remove('pending');
        return;
    }

    speakerNode.classList.remove('hidden');
    speakerNode.classList.toggle('pending', pending);
    speakerNode.textContent = pending ? 'loading...' : speaker;
}

function appendStatusLine(message) {
    const line = document.createElement('div');
    line.className = 'line status-line';
    line.textContent = `Loading: ${message}`;
    DOM.transcriptText.appendChild(line);
    DOM.placeholder.classList.add('hidden');

    const container = DOM.transcriptText.parentElement;
    container.scrollTop = container.scrollHeight;
}

function clearTranscript() {
    transcriptNodes.clear();
    DOM.transcriptText.innerHTML = '';
    DOM.placeholder.classList.remove('hidden');
}

function setConnectionStatus(connected) {
    DOM.badge.classList.toggle('connected', connected);
    DOM.badgeText.textContent = connected ? 'Streaming' : 'Disconnected';
}

function setupVisualizer() {
    const canvas = DOM.canvas;
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!isStreaming) {
            drawIdleVisualizer(ctx, rect.width, rect.height);
        }
    }

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function drawIdleVisualizer(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(124, 100, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
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
        const width = rect.width;
        const height = rect.height;
        ctx.clearRect(0, 0, width, height);

        const barCount = Math.min(bufferLength, 80);
        const barWidth = width / barCount;
        const gap = 2;

        for (let i = 0; i < barCount; i += 1) {
            const value = dataArray[i] / 255;
            const barHeight = value * height * 0.85;
            const hue = 255 - value * 40;
            ctx.fillStyle = `hsla(${hue}, 70%, 65%, ${0.4 + value * 0.6})`;
            ctx.fillRect(
                i * barWidth + gap / 2,
                (height - barHeight) / 2,
                barWidth - gap,
                barHeight || 1,
            );
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
