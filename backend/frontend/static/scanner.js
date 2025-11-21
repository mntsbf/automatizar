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

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    overlay.hidden = true;
    captureBtn.disabled = false;
    setStatus('En vivo', 'success');
  } catch (err) {
    console.error('camera error', err);
    setStatus('Sin cámara', 'danger');
    overlay.hidden = false;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  overlay.hidden = false;
  captureBtn.disabled = true;
  setStatus('Cámara detenida', 'secondary');
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

async function captureStep() {
  if (!stream) {
    setStatus('Sin cámara', 'danger');
    return;
  }
  const step = steps[currentStep];
  if (!step) return;
  setStatus('Procesando...', 'warning');
  captureBtn.disabled = true;
  const shot = await snapshot();
  if (!shot.blob) {
    setStatus('Sin frame', 'danger');
    captureBtn.disabled = false;
    return;
  }

  const formData = new FormData();
  formData.append('image', shot.blob, `${step.id}.jpg`);
  formData.append('action', step.id);
  if (liveCheck?.checked) formData.append('live_check', 'true');
  if (refreshCheck?.checked) formData.append('refresh_bank', 'true');
  if (saveCheck?.checked) formData.append('save_photo', 'true');
  const personId = personSelect?.value;
  if (personId) formData.append('person_id', personId);

  try {
    const resp = await fetch('/api/scan/frame', { method: 'POST', body: formData });
    const payload = await resp.json();
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
      setStatus(payload.message || 'Movimiento insuficiente', 'danger');
    }
  } catch (err) {
    console.error('scan frame failed', err);
    setStatus('Error de red', 'danger');
  } finally {
    captureBtn.disabled = false;
  }
}

async function finalizeScan() {
  if (!embeddings.length) {
    setStatus('Sin embeddings', 'danger');
    return;
  }
  setStatus('Combinando...', 'primary');
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

renderSteps();
loadPersons();
