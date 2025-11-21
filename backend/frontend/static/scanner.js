const steps = [
  { id: 'frente', label: 'Mirar al frente', hint: 'Mira fijo a la cámara.' },
  { id: 'derecha', label: 'Girar a la derecha', hint: 'Gira suavemente tu cabeza a la derecha.' },
  { id: 'izquierda', label: 'Girar a la izquierda', hint: 'Gira ahora hacia la izquierda.' },
  { id: 'arriba', label: 'Levantar la cabeza', hint: 'Levanta la barbilla y mira ligeramente hacia arriba.' },
  { id: 'abajo', label: 'Bajar la cabeza', hint: 'Baja la mirada con la cabeza hacia abajo.' },
  { id: 'sonreir', label: 'Sonreír', hint: 'Sonríe mostrando los dientes.' },
];

let currentStep = 0;
let embeddings = [];
let previews = [];
let stream = null;
let liveLoop = false;
let busy = false;
const LIVE_INTERVAL = 900; // ms between live evaluations

const video = document.getElementById('scan-video');
const overlay = document.getElementById('scan-overlay');
const startBtn = document.getElementById('scan-start');
const captureBtn = document.getElementById('scan-capture');
const statusBadge = document.getElementById('scan-status');
const stepsList = document.getElementById('scan-steps');
const hintBox = document.getElementById('scan-hint');
const previewsBox = document.getElementById('scan-previews');
const resultBox = document.getElementById('scan-result');
const progressBadge = document.getElementById('scan-progress');
const resetBtn = document.getElementById('scan-reset');
const personSelect = document.getElementById('scan-person');
const saveCheck = document.getElementById('scan-save');
const liveCheck = document.getElementById('scan-live');
const refreshCheck = document.getElementById('scan-refresh');
const autoCheck = document.getElementById('scan-auto');
const deviceSelect = document.getElementById('scan-device');

const overlayText = document.querySelector('#scan-overlay span');

const cameraHints = {
  insecure:
    'El navegador bloqueó la cámara en HTTP. Abre el panel en https:// o usa localhost para permitir el acceso.',
  denied: 'Permiso de cámara denegado. Autoriza el uso de cámara en el navegador y vuelve a intentarlo.',
  notfound: 'No se encontraron dispositivos de cámara disponibles.',
};

function setStatus(text, color = 'secondary') {
  statusBadge.textContent = text;
  statusBadge.className = `badge bg-${color}`;
}

function renderSteps() {
  if (!stepsList) return;
  stepsList.innerHTML = '';
  steps.forEach((step, idx) => {
    const active = idx === currentStep;
    const done = idx < currentStep;
    const li = document.createElement('div');
    li.className = 'list-group-item d-flex align-items-center justify-content-between';
    li.innerHTML = `
      <div>
        <div class="fw-semibold ${done ? 'text-success' : ''}">${step.label}</div>
        <div class="text-muted small">${step.hint}</div>
      </div>
      <span class="badge ${done ? 'bg-success' : active ? 'bg-primary' : 'bg-secondary-subtle text-dark'}">${done ? 'OK' : active ? 'Ahora' : 'Pendiente'}</span>
    `;
    stepsList.appendChild(li);
  });
  progressBadge.textContent = `${Math.min(currentStep, steps.length)}/${steps.length}`;
  if (hintBox) hintBox.querySelector('div').textContent = steps[currentStep]?.hint || 'Comienza el flujo de escaneo.';
}

function addPreview(dataUrl, step, quality) {
  const col = document.createElement('div');
  col.className = 'col-6 col-lg-4';
  col.innerHTML = `
    <div class="card h-100 shadow-sm">
      <img src="${dataUrl}" class="card-img-top" alt="${step.label}" />
      <div class="card-body py-2">
        <div class="small">${step.label}</div>
        <div class="text-muted small">Calidad ${(quality || 0).toFixed(2)}</div>
      </div>
    </div>
  `;
  previewsBox.appendChild(col);
}

async function loadPersons() {
  if (!personSelect) return;
  personSelect.innerHTML = '<option value="">Sin persona (solo prueba)</option>';
  try {
    const resp = await fetch('/api/persons');
    const persons = await resp.json();
    persons.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.full_name} ${p.list_tag ? '(' + p.list_tag + ')' : ''}`;
      personSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('persons load failed', e);
  }
}

async function loadDevices() {
  if (!deviceSelect || !navigator.mediaDevices?.enumerateDevices) return;
  deviceSelect.innerHTML = '<option value="">Predeterminada</option>';
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter((d) => d.kind === 'videoinput');
    videos.forEach((v, idx) => {
      const opt = document.createElement('option');
      opt.value = v.deviceId;
      opt.textContent = v.label || `Cámara ${idx + 1}`;
      deviceSelect.appendChild(opt);
    });
    if (!videos.length) setStatus(cameraHints.notfound, 'danger');
  } catch (err) {
    console.error('enumerateDevices failed', err);
  }
}

function getUserMedia(constraints) {
  if (navigator.mediaDevices?.getUserMedia) return navigator.mediaDevices.getUserMedia(constraints);
  const legacy =
    navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
  if (legacy) return new Promise((resolve, reject) => legacy.call(navigator, constraints, resolve, reject));
  return null;
}

async function startCamera() {
  if (!window.isSecureContext && location.hostname !== 'localhost') {
    setStatus(cameraHints.insecure, 'danger');
    overlay.hidden = false;
    if (overlayText) overlayText.textContent = cameraHints.insecure;
    return;
  }

  const gum = getUserMedia({ video: true, audio: false });
  if (!gum || typeof gum.then !== 'function') {
    setStatus('Este navegador no soporta cámara (getUserMedia)', 'danger');
    overlay.hidden = false;
    if (overlayText) overlayText.textContent = cameraHints.notfound;
    return;
  }

  try {
    const constraints = { video: { facingMode: 'user' }, audio: false };
    if (deviceSelect?.value) constraints.video = { deviceId: { exact: deviceSelect.value } };
    stream = await getUserMedia(constraints);
    video.srcObject = stream;
    overlay.hidden = true;
    captureBtn.disabled = false;
    setStatus('En vivo', 'success');
    if (autoCheck?.checked) startLiveLoop();
  } catch (err) {
    console.error('camera error', err);
    let msg = 'Sin cámara';
    if (!window.isSecureContext) msg = cameraHints.insecure;
    else if (err.name === 'NotAllowedError') msg = cameraHints.denied;
    else if (err.name === 'NotFoundError') msg = cameraHints.notfound;
    setStatus(msg, 'danger');
    overlay.hidden = false;
    if (overlayText) overlayText.textContent = msg;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  overlay.hidden = false;
  if (overlayText) overlayText.textContent = 'Cámara detenida';
  captureBtn.disabled = true;
  setStatus('Cámara detenida', 'secondary');
  stopLiveLoop();
}

function stopLiveLoop() {
  liveLoop = false;
}

function snapshot() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob((blob) => resolve({ blob, dataUrl: canvas.toDataURL('image/jpeg') }), 'image/jpeg', 0.9));
}

function renderResult(payload) {
  if (!resultBox) return;
  if (!payload) {
    resultBox.innerHTML = '<p class="text-muted mb-0">Aún no se combinan embeddings.</p>';
    return;
  }
  const conf = payload.similarity ? (payload.similarity * 100).toFixed(1) : null;
  resultBox.innerHTML = `
    <div class="d-flex flex-column gap-2">
      <div class="fw-semibold">Embedding combinado (${payload.mode || 'median'})</div>
      <div class="text-muted small">${payload.count || 0} muestras · ${payload.embedding ? payload.embedding.length : 0} dims</div>
      ${payload.saved_embedding_id ? `<span class="badge bg-success">Guardado en BD (#${payload.saved_embedding_id})</span>` : ''}
      ${payload.similarity ? `<div class="text-muted small">Confianza referencia: ${conf}%</div>` : ''}
    </div>
  `;
}

function describeReason(reason) {
  switch (reason) {
    case 'no_face':
      return 'No se detectaron rostros válidos';
    case 'sin_landmarks':
      return 'No se pudo estimar el giro (acércate o mejora la luz)';
    case 'accion_no_reconocida':
      return 'Acción no reconocida';
    case 'lenient':
      return 'Movimiento aceptado en modo tolerante';
    case 'spoof':
      return 'Liveness falló, intenta de nuevo';
    default:
      return 'Movimiento insuficiente para el paso';
  }
}

async function sendFrame(step, shot) {
  const formData = new FormData();
  formData.append('image', shot.blob, `${step.id}.jpg`);
  formData.append('action', step.id);
  if (liveCheck?.checked) formData.append('live_check', 'true');
  if (refreshCheck?.checked) formData.append('refresh_bank', 'true');
  if (saveCheck?.checked) formData.append('save_photo', 'true');
  const personId = personSelect?.value;
  if (personId) formData.append('person_id', personId);

  const resp = await fetch('/api/scan/frame', { method: 'POST', body: formData });
  const payload = await resp.json();
  return { ok: resp.ok, payload };
}

async function handleFrame(step) {
  if (!stream || busy) return;
  busy = true;
  setStatus('Procesando...', 'warning');

  const shot = await snapshot();
  if (!shot.blob) {
    setStatus('Sin frame', 'danger');
    busy = false;
    return;
  }

  try {
    const { ok, payload } = await sendFrame(step, shot);
    if (payload.ok) {
      embeddings.push(payload.embedding);
      previews.push(payload.embedding);
      addPreview(shot.dataUrl, step, payload.quality || 0);
      currentStep += 1;
      setStatus('Paso validado', 'success');
      renderSteps();
      if (currentStep >= steps.length) {
        await finalizeScan();
      }
    } else {
      const reason = payload.reason || (ok ? 'sin_match' : 'error');
      const color = reason === 'no_face' ? 'warning' : 'danger';
      setStatus(payload.message || describeReason(reason), color);
    }
  } catch (err) {
    console.error('scan frame failed', err);
    setStatus('Error de red', 'danger');
  } finally {
    busy = false;
  }
}

function startLiveLoop() {
  if (liveLoop) return;
  liveLoop = true;
  const loop = async () => {
    if (!liveLoop || currentStep >= steps.length) return;
    const step = steps[currentStep];
    if (step) await handleFrame(step);
    setTimeout(loop, LIVE_INTERVAL);
  };
  loop();
}

async function captureStep() {
  if (!stream) {
    setStatus('Sin cámara', 'danger');
    return;
  }
  const step = steps[currentStep];
  if (!step) return;
  captureBtn.disabled = true;
  await handleFrame(step);
  captureBtn.disabled = false;
}

async function finalizeScan() {
  if (!embeddings.length) {
    setStatus('Sin embeddings', 'danger');
    return;
  }
  setStatus('Combinando...', 'primary');
  stopLiveLoop();
  try {
    const resp = await fetch('/api/scan/finalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeddings, mode: 'median', person_id: personSelect?.value || null }),
    });
    const payload = await resp.json();
    if (!resp.ok) {
      setStatus(payload.error || 'No se pudo combinar', 'danger');
      return;
    }
    setStatus('Embedding listo', 'success');
    renderResult(payload);
  } catch (err) {
    console.error('finalize failed', err);
    setStatus('Error de red', 'danger');
  }
}

function resetFlow() {
  stopLiveLoop();
  busy = false;
  embeddings = [];
  previews = [];
  currentStep = 0;
  previewsBox.innerHTML = '';
  renderSteps();
  renderResult(null);
  setStatus('Listo', 'secondary');
}

if (startBtn) startBtn.addEventListener('click', startCamera);
if (captureBtn) captureBtn.addEventListener('click', captureStep);
if (resetBtn) resetBtn.addEventListener('click', resetFlow);
if (autoCheck)
  autoCheck.addEventListener('change', () => {
    if (autoCheck.checked && stream) startLiveLoop();
    else stopLiveLoop();
  });

renderSteps();
loadPersons();
loadDevices();
