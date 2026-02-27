// app.js — Multi-person scanning + filters + confidence + buckets + preview labels
// Models are served locally in same Netlify site: /models

const MODEL_URL = "/models";

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

const scanPeople = el("scanPeople");
const btnAll = el("btnAll");
const btnNone = el("btnNone");

const scanInput = el("scanInput");
const btnScan = el("btnScan");
const scanStatus = el("scanStatus");

const threshold = el("threshold");
const thrVal = el("thrVal");

const resultsFlagged = el("resultsFlagged");
const resultsPossible = el("resultsPossible");
const resultsClear = el("resultsClear");

const countFlagged = el("countFlagged");
const countPossible = el("countPossible");
const countClear = el("countClear");

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
// scanSelection = { [personId]: boolean }
let people = loadPeople();
let scanSelection = loadSelection();

// Performance knobs
const DETECTOR = {
  inputSize: 512,         // 320 = faster, 512 = better
  scoreThreshold: 0.35    // lower = more sensitive
};
const BORDERLINE_BAND = 0.06; // threshold -> threshold+band = "Possible"
const MAX_FACES_PER_IMAGE = 20; // sanity cap

// ---------- Helpers ----------
function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function setStatus(target, msg) {
  target.textContent = msg || "";
}

function savePeople() {
  localStorage.setItem("dnu_people_v2", JSON.stringify(people));
  // Keep selection in sync (new people default to checked)
  syncSelection();
  renderAll();
}

function loadPeople() {
  try {
    const raw = localStorage.getItem("dnu_people_v2") || localStorage.getItem("dnu_people_v1");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSelection() {
  localStorage.setItem("dnu_selection_v2", JSON.stringify(scanSelection));
}

function loadSelection() {
  try {
    const raw = localStorage.getItem("dnu_selection_v2");
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function totalSamples() {
  return people.reduce((sum, p) => sum + (p.samples?.length || 0), 0);
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, m => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"
  }[m]));
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

// distance -> a rough % (not scientifically “probability”, but usable UX)
function distanceToConfidence(d, thr) {
  // 0.0 => 100%, thr => ~60%, thr+band => ~40%, bigger => lower
  const band = BORDERLINE_BAND;
  const max = thr + band + 0.20;
  const clamped = Math.max(0, Math.min(1, 1 - (d / max)));
  return Math.round(clamped * 100);
}

async function fileToImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  URL.revokeObjectURL(url);
  return img;
}

function yieldToUI() {
  return new Promise(r => setTimeout(r, 0));
}

function selectedPeople() {
  return people.filter(p => scanSelection[p.id] !== false); // default true
}

function syncSelection() {
  // New people default selected
  const known = new Set(people.map(p => p.id));
  for (const p of people) {
    if (!(p.id in scanSelection)) scanSelection[p.id] = true;
  }
  // Remove deleted people
  for (const id of Object.keys(scanSelection)) {
    if (!known.has(id)) delete scanSelection[id];
  }
  saveSelection();
}

// For scanning: compute best match per person, return top matches
function matchesForDescriptor(descriptor, thr) {
  const pool = selectedPeople();
  const hits = [];
  const possibles = [];

  for (const p of pool) {
    let best = Infinity;
    for (const s of (p.samples || [])) {
      const d = dist(descriptor, s);
      if (d < best) best = d;
      // small early-exit
      if (best < 0.20) break;
    }
    if (!isFinite(best)) continue;

    if (best <= thr) {
      hits.push({ personId: p.id, name: p.name, dist: best, confidence: distanceToConfidence(best, thr), level: "flagged" });
    } else if (best <= thr + BORDERLINE_BAND) {
      possibles.push({ personId: p.id, name: p.name, dist: best, confidence: distanceToConfidence(best, thr), level: "possible" });
    }
  }

  hits.sort((a,b) => a.dist - b.dist);
  possibles.sort((a,b) => a.dist - b.dist);

  return { hits, possibles };
}

// ---------- Rendering ----------
function renderPeople() {
  peopleCount.textContent = `${people.length} people • ${totalSamples()} samples`;

  // Dropdown
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

  // Left list
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

function renderScanPeople() {
  scanPeople.innerHTML = "";

  if (!people.length) {
    const div = document.createElement("div");
    div.className = "muted small";
    div.textContent = "Add people to enable scanning filters.";
    scanPeople.appendChild(div);
    return;
  }

  for (const p of people) {
    const row = document.createElement("label");
    row.className = "chk";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = scanSelection[p.id] !== false;
    box.onchange = () => {
      scanSelection[p.id] = box.checked;
      saveSelection();
      updateButtons();
    };

    const name = document.createElement("div");
    name.className = "chk__name";
    name.textContent = p.name;

    const meta = document.createElement("div");
    meta.className = "chk__meta";
    meta.textContent = `${p.samples?.length || 0} samples`;

    row.appendChild(box);
    row.appendChild(name);
    row.appendChild(meta);
    scanPeople.appendChild(row);
  }
}

function clearBuckets() {
  resultsFlagged.innerHTML = "";
  resultsPossible.innerHTML = "";
  resultsClear.innerHTML = "";
  countFlagged.textContent = "0";
  countPossible.textContent = "0";
  countClear.textContent = "0";
}

function updateButtons() {
  btnAddPerson.disabled = !personName.value.trim();
  const pool = selectedPeople();
  const poolSamples = pool.reduce((s,p)=>s+(p.samples?.length||0),0);
  btnScan.disabled = !modelsReady || scanInput.files.length === 0 || pool.length === 0 || poolSamples === 0;
}

function renderAll() {
  renderPeople();
  renderScanPeople();
  updateButtons();
}

renderAll();

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
    setStatus(modelStatus, "Model load failed. Check /models and that .bin files exist.");
  }
  updateButtons();
}

loadModels();

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
  scanSelection = {};
  localStorage.removeItem("dnu_people_v2");
  localStorage.removeItem("dnu_selection_v2");
  renderAll();
  setStatus(enrollStatus, "Cleared.");
  setStatus(scanStatus, "Cleared.");
  crops.innerHTML = "";
  clearBuckets();
  clearPreview();
});

btnAll.addEventListener("click", () => {
  for (const p of people) scanSelection[p.id] = true;
  saveSelection();
  renderScanPeople();
  updateButtons();
});
btnNone.addEventListener("click", () => {
  for (const p of people) scanSelection[p.id] = false;
  saveSelection();
  renderScanPeople();
  updateButtons();
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
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions(DETECTOR))
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!detections.length) continue;

    // cap to avoid crazy crowd shots
    const trimmed = detections.slice(0, MAX_FACES_PER_IMAGE);

    const tmp = document.createElement("canvas");
    tmp.width = img.width; tmp.height = img.height;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(img, 0, 0);

    for (const det of trimmed) {
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

    await yieldToUI();
  }

  setStatus(enrollStatus, found ? `Found ${found} face(s). Click crops to add samples to ${person.name}.` : "No faces found.");
}

// ---------- Scan ----------
threshold.addEventListener("input", () => {
  thrVal.textContent = threshold.value;
});
scanInput.addEventListener("change", updateButtons);

btnScan.addEventListener("click", async () => {
  if (!modelsReady) return setStatus(scanStatus, "Models not ready yet.");

  const files = Array.from(scanInput.files || []);
  if (!files.length) return;

  const pool = selectedPeople();
  const poolSamples = pool.reduce((s,p)=>s+(p.samples?.length||0),0);
  if (!pool.length || poolSamples === 0) {
    return setStatus(scanStatus, "No selected people with samples. Check your scan filters + enroll at least 1 sample.");
  }

  const thr = parseFloat(threshold.value);
  const possibleThr = thr + BORDERLINE_BAND;

  clearBuckets();
  setStatus(scanStatus, `Scanning ${files.length} photo(s)…`);

  let flaggedCount = 0;
  let possibleCount = 0;
  let clearCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    setStatus(scanStatus, `Scanning ${i+1} / ${files.length}…`);
    const img = await fileToImage(file);

    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions(DETECTOR))
      .withFaceLandmarks()
      .withFaceDescriptors();

    const trimmed = detections.slice(0, MAX_FACES_PER_IMAGE);

    // For each face, compute matches to selected people
    const faceResults = trimmed.map(det => {
      const desc = Array.from(det.descriptor);
      const m = matchesForDescriptor(desc, thr);
      // choose best candidate among hits, else possibles
      const best = m.hits[0] || m.possibles[0] || null;
      return { det, ...m, best };
    });

    // Determine image status
    const imageHits = [];
    const imagePossibles = [];

    for (const fr of faceResults) {
      if (fr.hits.length) imageHits.push(...fr.hits);
      else if (fr.possibles.length) imagePossibles.push(...fr.possibles);
    }

    // Deduplicate by person (keep best distance per person)
    function dedupe(list) {
      const map = new Map();
      for (const item of list) {
        const prev = map.get(item.personId);
        if (!prev || item.dist < prev.dist) map.set(item.personId, item);
      }
      return Array.from(map.values()).sort((a,b)=>a.dist-b.dist);
    }
    const uniqHits = dedupe(imageHits);
    const uniqPoss = dedupe(imagePossibles);

    let status = "clear";
    if (uniqHits.length) status = "flagged";
    else if (uniqPoss.length) status = "possible";

    // Build label string
    const names = (status === "flagged" ? uniqHits : status === "possible" ? uniqPoss : [])
      .slice(0, 3)
      .map(x => `${x.name} (${x.confidence}%)`);
    const more = ((status === "flagged" ? uniqHits : status === "possible" ? uniqPoss : [])).length - 3;
    const label = names.length ? `${names.join(", ")}${more > 0 ? ` +${more}` : ""}` : "—";

    // Render row into correct bucket
    const row = document.createElement("div");
    row.className = "result";

    const left = document.createElement("div");
    left.innerHTML = `
      <div class="result__name">${escapeHtml(file.name)}</div>
      <div class="result__sub">${trimmed.length} face(s) detected • ${status === "clear" ? "No matches" : label}</div>
    `;

    const right = document.createElement("div");
    right.className = "result__right";

    const tag = document.createElement("span");
    tag.className = `tag ${
      status === "flagged" ? "tag--bad" : status === "possible" ? "tag--warn" : "tag--good"
    }`;
    tag.textContent = status.toUpperCase();

    const btnPrev = document.createElement("button");
    btnPrev.className = "btn btn--ghost btn--sm";
    btnPrev.textContent = "Preview";
    btnPrev.onclick = () => {
      drawPreview(img, faceResults, thr, possibleThr);
    };

    right.appendChild(tag);
    right.appendChild(btnPrev);

    row.appendChild(left);
    row.appendChild(right);

    if (status === "flagged") {
      resultsFlagged.appendChild(row);
      flaggedCount++;
      // auto-preview first flagged
      if (flaggedCount === 1) drawPreview(img, faceResults, thr, possibleThr);
    } else if (status === "possible") {
      resultsPossible.appendChild(row);
      possibleCount++;
      // if nothing flagged yet, auto-preview first possible
      if (flaggedCount === 0 && possibleCount === 1) drawPreview(img, faceResults, thr, possibleThr);
    } else {
      resultsClear.appendChild(row);
      clearCount++;
      // if nothing else yet, auto-preview first clear
      if (flaggedCount === 0 && possibleCount === 0 && clearCount === 1) drawPreview(img, faceResults, thr, possibleThr);
    }

    countFlagged.textContent = String(flaggedCount);
    countPossible.textContent = String(possibleCount);
    countClear.textContent = String(clearCount);

    await yieldToUI();
  }

  setStatus(
    scanStatus,
    `Done. Flagged ${flaggedCount} • Possible ${possibleCount} • Clear ${clearCount}. Threshold ${thr.toFixed(2)} (Possible up to ${ (thr + BORDERLINE_BAND).toFixed(2) }).`
  );
});

function clearPreview() {
  const ctx = previewCanvas.getContext("2d");
  previewCanvas.width = 1;
  previewCanvas.height = 1;
  ctx.clearRect(0, 0, 1, 1);
}

function drawPreview(img, faceResults, thr, possibleThr) {
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

  for (const fr of faceResults) {
    const b = fr.det.detection.box;
    const x = b.x * scale, y = b.y * scale, w = b.width * scale, h = b.height * scale;

    // choose best match (flagged first, else possible)
    const best = fr.best;

    let level = "clear";
    if (best?.dist <= thr) level = "flagged";
    else if (best?.dist <= possibleThr) level = "possible";

    const stroke =
      level === "flagged" ? "#b00020" :
      level === "possible" ? "#d9a23a" :
      "#00b374";

    ctx.strokeStyle = stroke;
    ctx.strokeRect(x, y, w, h);

    let label = "face";
    if (level === "flagged") label = `RESTRICTED: ${best.name} (${best.confidence}%)`;
    else if (level === "possible") label = `POSSIBLE: ${best.name} (${best.confidence}%)`;

    const pad = 4;
    const tw = ctx.measureText(label).width;

    ctx.fillStyle = stroke;
    ctx.fillRect(x, y, tw + pad * 2, 18 + pad);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + pad, y + pad);
  }
}

// ---------- Export / Import ----------
btnExport.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ version: 2, people }, null, 2)], { type: "application/json" });
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
    syncSelection();
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

// Keep selection sane on load
syncSelection();
renderAll();
