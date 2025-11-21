const form = document.getElementById('lookup-form');
const resultBox = document.getElementById('lookup-result');
const alertBox = document.getElementById('lookup-alert');
const bankBadge = document.getElementById('bank-version');

function showAlert(kind, text) {
  if (!alertBox) return;
  alertBox.className = `alert alert-${kind}`;
  alertBox.textContent = text;
  alertBox.classList.remove('d-none');
}

function renderUnknown(payload) {
  const similarity = payload?.similaridad_maxima || 0;
  resultBox.innerHTML = `
    <div class="d-flex align-items-center gap-3">
      <i class="bi bi-question-circle text-warning fs-2"></i>
      <div>
        <div class="fw-semibold">Sin coincidencias</div>
        <div class="text-muted small">Máx. similitud: ${(similarity * 100).toFixed(1)}%</div>
      </div>
    </div>
  `;
  showAlert('warning', payload?.mensaje || 'No se encontró una coincidencia bajo el umbral.');
}

function renderMatch(payload) {
  const persona = payload.persona;
  const percent = (persona.confianza * 100).toFixed(1);
  const distancia = (persona.distancia ?? 0).toFixed(4);
  const tag = persona.list_tag ? `<span class="badge bg-danger-subtle text-danger ms-2">${persona.list_tag}</span>` : '';
  resultBox.innerHTML = `
    <div class="d-flex flex-column gap-2">
      <div class="d-flex align-items-center gap-2">
        <i class="bi bi-person-badge-fill text-success fs-3"></i>
        <div>
          <div class="fw-semibold">${persona.nombre}${tag}</div>
          <div class="text-muted small">ID ${persona.id} · Confianza ${percent}% · Distancia ${distancia}</div>
        </div>
      </div>
      ${payload.live === false ? '<div class="badge bg-warning text-dark">Anti-spoofing: revisar</div>' : ''}
      ${payload.diagnosticos ? `<pre class="bg-light p-2 rounded small mb-0">${JSON.stringify(payload.diagnosticos, null, 2)}</pre>` : ''}
    </div>
  `;
  showAlert('success', 'Coincidencia encontrada en la base de datos.');
}

async function submitLookup(event) {
  event.preventDefault();
  if (!form || !resultBox) return;
  alertBox.classList.add('d-none');
  resultBox.textContent = 'Procesando...';

  const formData = new FormData(form);
  const liveCheck = form.querySelector('#live_check');
  if (liveCheck && liveCheck.checked) formData.set('live_check', 'true');

  try {
    const resp = await fetch('/buscar-persona', {
      method: 'POST',
      body: formData,
    });
    const payload = await resp.json();
    if (payload.bank_version && bankBadge) {
      bankBadge.textContent = `Banco v${payload.bank_version}`;
    }

    if (!resp.ok) {
      showAlert('danger', payload.error || payload.mensaje || 'No se pudo procesar la imagen.');
      resultBox.textContent = 'Intenta nuevamente con otra foto.';
      return;
    }

    if (payload.resultado === 'match') {
      renderMatch(payload);
    } else if (payload.resultado === 'sin_rostro') {
      showAlert('warning', payload.mensaje || 'No se detectaron rostros válidos.');
      resultBox.textContent = 'No se detectaron rostros en la foto enviada.';
    } else {
      renderUnknown(payload);
    }
  } catch (err) {
    console.error('lookup failed', err);
    showAlert('danger', 'Error inesperado procesando la imagen.');
    resultBox.textContent = 'No se pudo completar la búsqueda.';
  }
}

if (form) {
  form.addEventListener('submit', submitLookup);
}
