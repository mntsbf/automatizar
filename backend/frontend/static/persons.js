async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function riskBadge(level) {
  const map = {
    critical: 'badge-risk-critical',
    warning: 'badge-risk-warning',
    info: 'badge-risk-info',
  };
  return `<span class="badge ${map[level] || map.info}">${level === 'critical' ? 'Lista negra' : level === 'warning' ? 'Lista gris' : 'Normal'}</span>`;
}

function renderPersons(persons) {
  const tbody = document.getElementById('persons-table');
  tbody.innerHTML = '';
  persons.forEach((p) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-semibold">${p.full_name}</td>
      <td>${p.role || '—'}</td>
      <td>${riskBadge(p.risk_level)}</td>
      <td><span class="badge bg-dark-subtle text-dark">${p.embeddings?.length || 0}</span></td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('person-total').textContent = `${persons.length} registradas`;
}

async function loadPersons() {
  const persons = await fetchJSON('/api/persons');
  renderPersons(persons);
  const select = document.getElementById('photo-person');
  select.innerHTML = '';
  persons.forEach((p) => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.full_name;
    select.appendChild(opt);
  });
}

async function handlePersonSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const embeddingsText = form.embeddings.value.trim();
  let embeddings = [];
  if (embeddingsText) {
    try {
      embeddings = JSON.parse(embeddingsText);
    } catch (e) {
      alert('Embeddings debe ser un JSON válido');
      return;
    }
  }
  const payload = {
    full_name: form.full_name.value,
    role: form.role.value,
    embeddings,
  };
  await fetchJSON('/api/persons', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  form.reset();
  loadPersons();
}

async function handlePhotoSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const personId = form.person_id.value;
  const res = await fetch(`/api/persons/${personId}/embedding`, {
    method: 'POST',
    body: data,
  });
  const status = document.getElementById('photo-status');
  if (res.ok) {
    status.textContent = 'Embedding agregado';
    form.reset();
    loadPersons();
  } else {
    const payload = await res.json().catch(() => ({}));
    status.textContent = payload.error || 'No se pudo procesar la foto';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadPersons();
  document.getElementById('persons-refresh').addEventListener('click', loadPersons);
  document.getElementById('person-form').addEventListener('submit', handlePersonSubmit);
  document.getElementById('photo-form').addEventListener('submit', handlePhotoSubmit);
});
