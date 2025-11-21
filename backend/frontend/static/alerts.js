let alertsSource = null;

function riskBadge(level, text) {
  const map = {
    critical: 'badge-risk-critical',
    warning: 'badge-risk-warning',
    info: 'badge-risk-info',
  };
  return `<span class="badge ${map[level] || map.info}">${text}</span>`;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function populateSelect(selectId, items, placeholder) {
  const select = document.getElementById(selectId);
  select.innerHTML = '';
  const defaultOpt = document.createElement('option');
  defaultOpt.value = '';
  defaultOpt.textContent = placeholder;
  select.appendChild(defaultOpt);
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.value;
    opt.textContent = item.label;
    select.appendChild(opt);
  });
}

function renderAlerts(alerts) {
  const tbody = document.getElementById('alerts-table');
  tbody.innerHTML = '';
  alerts.forEach((a) => {
    const tr = document.createElement('tr');
    const riskLabel = a.risk_level === 'critical' ? 'Crítica' : a.risk_level === 'warning' ? 'Vigilancia' : 'Normal';
    tr.innerHTML = `
      <td>${new Date(a.created_at).toLocaleString()}</td>
      <td>${a.camera?.site?.name || a.camera?.site_name || 'Sin sede'}</td>
      <td>${a.camera?.name || '—'}</td>
      <td>${a.person?.full_name || 'Desconocido'}</td>
      <td>${riskBadge(a.risk_level, riskLabel)}</td>
      <td>${(a.similarity * 100).toFixed(1)}%</td>
      <td>${a.message || ''}</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('alerts-count').textContent = `${alerts.length} eventos`;
}

async function loadFilters() {
  const [sites, persons] = await Promise.all([fetchJSON('/api/sites'), fetchJSON('/api/persons')]);
  populateSelect('filter-site', sites.map((s) => ({ value: s.id, label: s.name })), 'Todas');
  populateSelect('filter-person', persons.map((p) => ({ value: p.id, label: p.full_name })), 'Todas');
}

async function loadAlerts() {
  const risk = document.getElementById('filter-risk').value;
  const site = document.getElementById('filter-site').value;
  const person = document.getElementById('filter-person').value;
  const params = new URLSearchParams();
  if (risk) params.append('risk', risk);
  if (site) params.append('site_id', site);
  if (person) params.append('person_id', person);
  params.append('limit', '100');
  const data = await fetchJSON(`/api/alerts?${params.toString()}`);
  renderAlerts(data);
}

function toggleLive() {
  const btn = document.getElementById('alerts-live');
  if (alertsSource) {
    alertsSource.close();
    alertsSource = null;
    btn.classList.remove('active');
    btn.innerHTML = '<i class="bi bi-broadcast-pin me-1"></i>Live';
    return;
  }
  alertsSource = new EventSource('/api/dashboard/tiempo-real');
  alertsSource.onmessage = (event) => {
    const payload = JSON.parse(event.data || '{}');
    if (payload.type === 'alert') {
      loadAlerts();
    }
  };
  alertsSource.onerror = () => {
    alertsSource?.close();
    alertsSource = null;
  };
  btn.classList.add('active');
  btn.innerHTML = '<i class="bi bi-broadcast me-1"></i>Live ON';
}

function resetFilters() {
  document.getElementById('filter-risk').value = '';
  document.getElementById('filter-site').value = '';
  document.getElementById('filter-person').value = '';
  loadAlerts();
}

window.addEventListener('DOMContentLoaded', () => {
  loadFilters().then(loadAlerts);
  document.getElementById('alerts-refresh').addEventListener('click', loadAlerts);
  document.getElementById('alerts-live').addEventListener('click', toggleLive);
  document.getElementById('alerts-reset').addEventListener('click', resetFilters);
});
