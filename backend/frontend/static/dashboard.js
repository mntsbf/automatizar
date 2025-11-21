async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

function riskBadge(level, text) {
  const map = {
    critical: 'badge-risk-critical',
    warning: 'badge-risk-warning',
    info: 'badge-risk-info',
  };
  return `<span class="badge ${map[level] || map.info}">${text}</span>`;
}

let chartInstance = null;
let eventSource = null;
let livePreferred = true;

function renderChart(labels, values) {
  const ctx = document.getElementById('alertsChart');
  if (chartInstance) chartInstance.destroy();
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Alertas',
          data: values,
          tension: 0.35,
          fill: true,
          borderColor: '#0d6efd',
          backgroundColor: 'rgba(13, 110, 253, 0.12)',
          pointBackgroundColor: '#0d6efd',
          pointBorderWidth: 0,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
        x: { grid: { display: false } },
      },
    },
  });
}

async function loadStats() {
  const data = await fetchJSON('/api/dashboard/estadisticas');
  document.getElementById('stat-alerts-today').textContent = data.alerts_today;
  document.getElementById('stat-critical').textContent = data.critical_alerts;
  document.getElementById('stat-cameras-active').textContent = data.cameras_active;
  document.getElementById('stat-cameras-total').textContent = data.cameras_total;
  document.getElementById('stat-sites').textContent = data.sites_with_alerts;
  document.getElementById('series-total').textContent = `${data.series_total} totales`;
  document.getElementById('notification-badge').textContent = data.critical_alerts;

  const topSites = document.getElementById('top-sites');
  topSites.innerHTML = '';
  data.matches_by_site.forEach((row) => {
    const item = document.createElement('div');
    item.className = 'list-group-item d-flex justify-content-between align-items-center';
    item.innerHTML = `<div class="fw-semibold">${row.site || 'Sin sede'}</div><span class="badge bg-dark-subtle text-dark">${row.count}</span>`;
    topSites.appendChild(item);
  });

  const labels = data.alerts_series.map((p) => p.label);
  const values = data.alerts_series.map((p) => p.count);
  renderChart(labels, values);
}

function renderAlertsTable(alerts) {
  const tbody = document.getElementById('alerts-table');
  tbody.innerHTML = '';
  alerts.forEach((alert) => {
    const tr = document.createElement('tr');
    const risk = alert.risk_level || 'info';
    const thumb = alert.thumbnail_url || 'https://via.placeholder.com/64x64.png?text=Face';
    tr.innerHTML = `
      <td>${new Date(alert.created_at).toLocaleString()}</td>
      <td>${alert.site || 'Sin sitio'}</td>
      <td>${alert.camera || '—'}</td>
      <td>${alert.person || 'Desconocido'}</td>
      <td>${riskBadge(risk, alert.risk_label)}</td>
      <td>${(alert.similarity * 100).toFixed(1)}%</td>
      <td><img class="thumbnail-face" src="${thumb}" alt="rostro" /></td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadAlerts() {
  const data = await fetchJSON('/api/dashboard/ultimas-alertas');
  renderAlertsTable(data.alerts);
}

function connectLive() {
  if (!window.EventSource) return;
  eventSource = new EventSource('/api/dashboard/tiempo-real');
  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data || '{}');
    if (payload.type === 'alert') {
      loadStats();
      loadAlerts();
    }
  };
  eventSource.onerror = () => {
    eventSource.close();
    eventSource = null;
  };
}

function toggleLive() {
  const button = document.getElementById('live-toggle');
  if (eventSource) {
    eventSource.close();
    eventSource = null;
    button.classList.remove('active');
    button.innerHTML = '<i class="bi bi-broadcast-pin me-1"></i>Live';
  } else {
    connectLive();
    button.classList.add('active');
    button.innerHTML = '<i class="bi bi-broadcast me-1"></i>Live ON';
  }
}

async function loadLivePreference() {
  try {
    const settings = await fetchJSON('/api/settings');
    livePreferred = String(settings.live_refresh || '').toLowerCase() !== 'false';
  } catch (e) {
    livePreferred = true;
  }
  if (livePreferred) {
    toggleLive();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadAlerts();
  document.getElementById('refresh-stats').addEventListener('click', loadStats);
  document.getElementById('refresh-alerts').addEventListener('click', loadAlerts);
  document.getElementById('live-toggle').addEventListener('click', toggleLive);
  loadLivePreference();
});
