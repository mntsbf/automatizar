const api = axios.create({ baseURL: '/api' });

const siteForm = document.getElementById('site-form');
const cameraForm = document.getElementById('camera-form');
const personForm = document.getElementById('person-form');
const cameraSiteSelect = document.getElementById('camera-site');

siteForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(siteForm);
  await api.post('/sites', Object.fromEntries(formData.entries()));
  siteForm.reset();
  await refreshSites();
});

cameraForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(cameraForm);
  await api.post('/cameras', Object.fromEntries(formData.entries()));
  cameraForm.reset();
  await Promise.all([refreshSites(), refreshCameras()]);
});

personForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(personForm);
  const embedding = formData.get('embedding');
  const payload = {
    full_name: formData.get('full_name'),
    role: formData.get('role'),
    embeddings: embedding ? [embedding] : [],
  };
  await api.post('/persons', payload);
  personForm.reset();
});

async function refreshSites() {
  const { data } = await api.get('/sites');
  const list = document.getElementById('site-list');
  list.innerHTML = '';
  cameraSiteSelect.innerHTML = '';
  data.forEach((site) => {
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    li.innerHTML = `<span>${site.name}</span><span class="badge bg-secondary">${site.location || '—'}</span>`;
    list.appendChild(li);

    const option = document.createElement('option');
    option.value = site.id;
    option.textContent = site.name;
    cameraSiteSelect.appendChild(option);
  });
}

async function refreshCameras() {
  const { data } = await api.get('/cameras');
  const list = document.getElementById('camera-list');
  list.innerHTML = '';
  data.forEach((cam) => {
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    li.innerHTML = `<span>${cam.name}</span><span class="badge bg-info">${cam.site_name || 'Sin sitio'}</span>`;
    list.appendChild(li);
  });
}

async function refreshAlerts() {
  const { data } = await api.get('/alerts');
  const list = document.getElementById('alerts');
  list.innerHTML = '';
  data.forEach((alert) => {
    const item = document.createElement('div');
    item.className = 'list-group-item';
    const person = alert.person ? alert.person.full_name : 'Desconocido';
    item.innerHTML = `
      <div class="fw-bold">${person} (${(alert.similarity * 100).toFixed(1)}%)</div>
      <div class="text-muted small">${alert.message}</div>
      <div class="small">${new Date(alert.created_at).toLocaleString()}</div>
    `;
    list.appendChild(item);
  });
}

document.getElementById('refresh-alerts').addEventListener('click', refreshAlerts);
document.getElementById('refresh-inventory').addEventListener('click', async () => {
  await Promise.all([refreshSites(), refreshCameras()]);
});

refreshSites().then(refreshCameras).then(refreshAlerts);
