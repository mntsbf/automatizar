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
  return `<span class="badge ${map[level] || map.info}">${level === 'critical' ? 'Crítica' : level === 'warning' ? 'Vigilancia' : 'Normal'}</span>`;
}

async function loadSites() {
  const sites = await fetchJSON('/api/sites');
  const select = document.getElementById('camera-site');
  select.innerHTML = '';
  sites.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.name;
    select.appendChild(opt);
  });
}

async function loadCameras() {
  const cameras = await fetchJSON('/api/cameras');
  document.getElementById('camera-total').textContent = `${cameras.length} registradas`;

  // map of alerts last 24h
  const alerts = await fetchJSON('/api/alerts?limit=500');
  const lastDay = Date.now() - 24 * 60 * 60 * 1000;
  const byCamera = {};
  alerts.forEach((a) => {
    if (new Date(a.created_at).getTime() >= lastDay && a.camera?.id) {
      byCamera[a.camera.id] = (byCamera[a.camera.id] || 0) + 1;
    }
  });

  const tbody = document.getElementById('cameras-table');
  tbody.innerHTML = '';
  cameras.forEach((c) => {
    const tr = document.createElement('tr');
    const count = byCamera[c.id] || 0;
    tr.innerHTML = `
      <td class="fw-semibold">${c.name}</td>
      <td>${c.site_name || 'Sin sede'}</td>
      <td><code>${c.rtsp_url}</code></td>
      <td><span class="badge ${count > 0 ? 'bg-success-subtle text-success' : 'bg-secondary-subtle text-secondary'}">${count} / 24h</span></td>
    `;
    tbody.appendChild(tr);
  });
}

async function handleCameraSubmit(event) {
  event.preventDefault();
  event.stopPropagation();
  const form = event.target;
  if (!form.checkValidity()) {
    form.classList.add('was-validated');
    return;
  }
  const payload = {
    name: form.name.value,
    rtsp_url: form.rtsp_url.value,
    site_id: Number(form.site_id.value),
  };
  await fetchJSON('/api/cameras', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  form.reset();
  form.classList.remove('was-validated');
  loadCameras();
}

window.addEventListener('DOMContentLoaded', () => {
  loadSites().then(loadCameras);
  document.getElementById('cameras-refresh').addEventListener('click', loadCameras);
  document.getElementById('camera-form').addEventListener('submit', handleCameraSubmit);
});
