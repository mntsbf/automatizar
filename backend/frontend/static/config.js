async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function applySettings(settings) {
  const form = document.getElementById('settings-form');
  form.tolerance.value = Number(settings.tolerance || 0.72);
  form.margin.value = Number(settings.margin || 0.15);
  form.live_refresh.checked = String(settings.live_refresh || '').toLowerCase() === 'true';
  form.theme.value = settings.theme || 'dark';
}

async function loadSettings() {
  const data = await fetchJSON('/api/settings');
  applySettings(data);
}

async function saveSettings(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    tolerance: form.tolerance.value,
    margin: form.margin.value,
    live_refresh: form.live_refresh.checked,
    theme: form.theme.value,
  };
  await fetchJSON('/api/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const status = document.getElementById('settings-status');
  status.textContent = 'Guardado';
  setTimeout(() => (status.textContent = ''), 2000);
}

window.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  document.getElementById('settings-form').addEventListener('submit', saveSettings);
});
