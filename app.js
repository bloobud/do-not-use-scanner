
// app.js
// Usable v1: People list + multiple samples/person + scan batch + preview boxes
// Requires face-api.js models hosted at MODEL_URL.

const MODEL_URL = "https://dashing-bavarois-f3bc73.netlify.app/models"; // <-- CHANGE THIS

// Elements
const el = (id) => document.getElementById(id);

const modelStatus = el("modelStatus");

const personName = el("personName");
const btnAddPerson = el("btnAddPerson");
const btnClearAll = el("btnClearAll");

const peopleCount = el("peopleCount");
const peopleList = el("peopleList");
const personSelect = el("personSelect");

const enrollInput = el("enrollInput");
const enrollDrop = el("enrollDrop");
const enrollStatus = el("enrollStatus");
const crops = el("crops");

const scanInput = el("scanInput");
const btnScan = el("btnScan");
const scanStatus = el("scanStatus");
const results = el("results");

const threshold = el("threshold");
const thrVal = el("thrVal");

const previewCanvas = el("previewCanvas");

const btnExport = el("btnExport");
const btnImport = el("btnImport");
const importFile = el("importFile");

const helpDialog = el("helpDialog");
const btnHelp = el("btnHelp");
const btnCloseHelp = el("btnCloseHelp");

// Data
let modelsReady = false;

// people = [{id,name,samples:number[][]}]
let people = loadPeople();

// ---------- Helpers ----------
function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function setStatus(target, msg) {
  target.textContent = msg || "";
}

function savePeople() {
  localStorage.setItem("dnu_people_v1", JSON.stringify(people));
  renderPeople();
  updateButtons();
}

function loadPeople() {
  try {
    const raw = localStorage.getItem("dnu_people_v1");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

async function fileToImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  URL.revokeObjectURL(url);
  return img;
}

function bestMatchToPeople(descriptor, thr) {
  let best = { hit: false, personName: "", personId: "", dist: Infinity };

  for (const p of people) {
    for (const s of (p.samples || [])) {
      const d = dist(descriptor, s);
      if (d < best.dist) {
        best = { hit: d <= thr, personName: p.name, personId: p.id, dist: d };
      }
    }
  }
  return best;
}

function updateButtons() {
  btnAddPerson.disabled = !personName.value.trim();
  btnScan.disabled = !modelsReady || scanInput.files.length === 0 || people.length === 0 || totalSamples() === 0;
}

function totalSamples() {
  return people.reduce((sum, p) => sum + (p.samples?.length || 0), 0);
}

function renderPeople() {
  peopleCount.textContent = `${people.length} people • ${totalSamples()} samples`;

  // Select options
  personSelect.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = people.length ? "Select a person…" : "Add a person to begin";
  personSelect.appendChild(opt0);

  for (const p of people) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.samples?.length || 0})`;
    personSelect.appendChild(opt);
  }

  // List cards
  peopleList.innerHTML = "";
  for (const p of people) {
    const card = document.createElement("div");
    card.className = "person";

    const top = document.createElement("div");
    top.className = "person__top";

    const left = document.createElement("div");
    left.innerHTML = `
      <div class="person__name">${escapeHtml(p.name)}</div>
      <div class="person__meta">${(p.samples?.length || 0)} sample(s)</div>
    `;

    const right = document.createElement("div");
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn--danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = () => {
      people = people.filter(x => x.id !== p.id);
      savePeople();
    };
    right.appendChild(delBtn);

    top.appendChild(left);
    top.appendChild(right);

    const actions = document.createElement("div");
    actions.className = "person__actions";

    const selectBtn = document.createElement("button");
    selectBtn.className = "btn btn--ghost";
    selectBtn.textContent = "Select";
    selectBtn.onclick = () => {
      personSelect.value = p.id;
      setStatus(enrollStatus, `Selected: ${p.name}. Upload photos and click face crops to add samples.`);
    };

    const clearSamplesBtn = document.createElement("button");
    clearSamplesBtn.className = "btn btn--ghost";
    clearSamplesBtn.textContent = "Clear Samples";
    clearSamplesBtn.onclick = () => {
      p.samples = [];
      savePeople();
    };

    actions.appendChild(selectBtn);
    actions.appendChild(clearSamplesBtn);

    card.appendChild(top);
    card.appendChild(actions);
    peopleList.appendChild(card);
  }
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, m => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"
  }[m]));
}

// ---------- Models ----------
async function loadModels() {
  setStatus(modelStatus, "Loading models…");
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    modelsReady = true;
    setStatus(modelStatus, "Models loaded ✓");
  } catch (e) {
    console.error(e);
    setStatus(modelStatus, "Model load failed. Check MODEL_URL + that .bin files exist.");
  }
  updateButtons();
}

loadModels();
renderPeople();
updateButtons();

// ---------- People ----------
personName.addEventListener("input", updateButtons);

btnAddPerson.addEventListener("click", () => {
  const name = personName.value.trim();
  if (!name) return;
  people.push({ id: uid(), name, samples: [] });
  personName.value = "";
  savePeople();
});

btnClearAll.addEventListener("click", () => {
  people = [];
  savePeople();
  setStatus(enrollStatus, "Cleared.");
  setStatus(scanStatus, "Cleared.");
  crops.innerHTML = "";
  results.innerHTML = "";
  clearPreview();
});

// ---------- Enroll (drag/drop + file input) ----------
function setupDropzone(dropEl, onFiles) {
  dropEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropEl.classList.add("dragover");
  });
  dropEl.addEventListener("dragleave", () => dropEl.classList.remove("dragover"));
  dropEl.addEventListener("drop", (e) => {
    e.preventDefault();
    dropEl.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files || []).filter(f => f.type.startsWith("image/"));
    onFiles(files);
  });
}

setupDropzone(enrollDrop, async (files) => {
  if (!files.length) return;
  await handleEnrollFiles(files);
});

enrollInput.addEventListener("change", async () => {
  const files = Array.from(enrollInput.files || []);
  await handleEnrollFiles(files);
  enrollInput.value = "";
});

async function handleEnrollFiles(files) {
  if (!modelsReady) return setStatus(enrollStatus, "Models not ready yet.");
  const pid = personSelect.value;
  if (!pid) return setStatus(enrollStatus, "Select a person first (left panel or dropdown).");

  const person = people.find(p => p.id === pid);
  if (!person) return setStatus(enrollStatus, "Selected person not found.");

  crops.innerHTML = "";
  setStatus(enrollStatus, `Detecting faces for: ${person.name}…`);

  let found = 0;

  for (const file of files) {
    const img = await fileToImage(file);

    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!detections.length) continue;

    // draw image into tmp canvas for cropping
    const tmp = document.createElement("canvas");
    tmp.width = img.width; tmp.height = img.height;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(img, 0, 0);

    for (const det of detections) {
      found++;
      const box = det.detection.box;
      const pad = Math.max(10, Math.round(Math.min(box.width, box.height) * 0.15));
      const x = Math.max(0, Math.floor(box.x - pad));
      const y = Math.max(0, Math.floor(box.y - pad));
      const w = Math.min(img.width - x, Math.floor(box.width + pad * 2));
      const h = Math.min(img.height - y, Math.floor(box.height + pad * 2));

      const crop = document.createElement("canvas");
      crop.width = 180; crop.height = 180;
      crop.getContext("2d").drawImage(tmp, x, y, w, h, 0, 0, crop.width, crop.height);
      const dataUrl = crop.toDataURL("image/jpeg", 0.86);

      const descriptor = Array.from(det.descriptor);

      const tile = document.createElement("div");
      tile.className = "crop";

      const im = document.createElement("img");
      im.src = dataUrl;

      const btn = document.createElement("button");
      btn.textContent = "Add sample";
      btn.onclick = () => {
        person.samples.push(descriptor);
        savePeople();
        btn.textContent = "Added ✓";
        btn.classList.add("added");
        btn.disabled = true;
        setStatus(enrollStatus, `Added sample to ${person.name}. Total: ${person.samples.length}`);
      };

      tile.appendChild(im);
      tile.appendChild(btn);
      crops.appendChild(tile);
    }
  }

  setStatus(enrollStatus, found ? `Found ${found} face(s). Click crops to add samples to ${person.name}.` : "No faces found.");
}

// ---------- Scan ----------
threshold.addEventListener("input", () => {
  thrVal.textContent = threshold.value;
});

scanInput.addEventListener("change", () => updateButtons());

btnScan.addEventListener("click", async () => {
  if (!modelsReady) return setStatus(scanStatus, "Models not ready yet.");
  const files = Array.from(scanInput.files || []);
  if (!files.length) return;

  const thr = parseFloat(threshold.value);
  results.innerHTML = "";
  setStatus(scanStatus, "Scanning…");

  let flagged = 0;

  for (const file of files) {
    const img = await fileToImage(file);

    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptors();

    const matches = detections.map(det => {
      const desc = Array.from(det.descriptor);
      return bestMatchToPeople(desc, thr);
    });

    const isFlagged = matches.some(m => m.hit);
    if (isFlagged) flagged++;

    const row = document.createElement("div");
    row.className = "result";

    const left = document.createElement("div");
    left.innerHTML = `
      <div class="result__name">${escapeHtml(file.name)}</div>
      <div class="muted small">${detections.length} face(s) detected</div>
    `;

    const right = document.createElement("div");
    right.style.display = "flex";
    right.style.gap = "10px";
    right.style.alignItems = "center";

    const tag = document.createElement("span");
    tag.className = `tag ${isFlagged ? "tag--bad" : "tag--good"}`;
    tag.textContent = isFlagged ? "FLAGGED" : "CLEAR";

    const btnPrev = document.createElement("button");
    btnPrev.className = "btn btn--ghost";
    btnPrev.textContent = "Preview";
    btnPrev.onclick = () => {
      drawPreview(img, detections, matches);
    };

    right.appendChild(tag);
    right.appendChild(btnPrev);

    row.appendChild(left);
    row.appendChild(right);
    results.appendChild(row);

    if (isFlagged && flagged === 1) {
      drawPreview(img, detections, matches);
    }
  }

  setStatus(scanStatus, `Done. Flagged ${flagged} / ${files.length}. Threshold: ${thr.toFixed(2)}`);
});

function clearPreview() {
  const ctx = previewCanvas.getContext("2d");
  previewCanvas.width = 1;
  previewCanvas.height = 1;
  ctx.clearRect(0, 0, 1, 1);
}

function drawPreview(img, detections, matches) {
  const ctx = previewCanvas.getContext("2d");
  const maxW = 1100;
  const scale = Math.min(1, maxW / img.width);

  previewCanvas.width = Math.round(img.width * scale);
  previewCanvas.height = Math.round(img.height * scale);

  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);

  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  ctx.textBaseline = "top";

  detections.forEach((det, idx) => {
    const b = det.detection.box;
    const x = b.x * scale, y = b.y * scale, w = b.width * scale, h = b.height * scale;

    const m = matches[idx];
    const hit = m?.hit;

    ctx.strokeStyle = hit ? "#b00020" : "#00b374";
    ctx.strokeRect(x, y, w, h);

    const label = hit ? `RESTRICTED: ${m.personName}` : "face";
    const pad = 4;
    const tw = ctx.measureText(label).width;

    ctx.fillStyle = hit ? "#b00020" : "#00b374";
    ctx.fillRect(x, y, tw + pad * 2, 18 + pad);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + pad, y + pad);
  });
}

// ---------- Export / Import ----------
btnExport.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ version: 1, people }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dnu-people.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

btnImport.addEventListener("click", () => importFile.click());

importFile.addEventListener("change", async () => {
  const f = importFile.files?.[0];
  if (!f) return;
  try {
    const parsed = JSON.parse(await f.text());
    if (!parsed?.people || !Array.isArray(parsed.people)) throw new Error("Invalid format");
    people = parsed.people;
    savePeople();
    setStatus(enrollStatus, "Imported people list.");
    setStatus(scanStatus, "Imported people list.");
  } catch (e) {
    console.error(e);
    setStatus(enrollStatus, "Import failed (bad JSON).");
  } finally {
    importFile.value = "";
    updateButtons();
  }
});

// ---------- Help dialog ----------
btnHelp.addEventListener("click", () => helpDialog.showModal());
btnCloseHelp.addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", (e) => {
  const rect = helpDialog.getBoundingClientRect();
  const inDialog = (
    e.clientX >= rect.left && e.clientX <= rect.right &&
    e.clientY >= rect.top && e.clientY <= rect.bottom
  );
  if (!inDialog) helpDialog.close();
});
