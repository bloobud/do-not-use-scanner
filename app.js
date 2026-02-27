// app.js — DNU Scanner
// - Only labels the SINGLE closest match from enrolled people
// - Only draws boxes for matched faces
// - Filters bogus detections (prevents "arm/torso is a face")
// - Upscales images to help small faces
// Models must be at /models on the SAME site.

const MODEL_URL = "/models";

// -------------------- Elements --------------------
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
const resultsClear = el("resultsClear");

const countFlagged = el("countFlagged");
const countClear = el("countClear");

const previewCanvas = el("previewCanvas");

const btnExport = el("btnExport");
const btnImport = el("btnImport");
const importFile = el("importFile");

const helpDialog = el("helpDialog");
const btnHelp = el("btnHelp");
const btnCloseHelp = el("btnCloseHelp");

// -------------------- Data --------------------
let modelsReady = false;

let people = loadPeople();            // [{id, name, samples:number[][]}]
let scanSelection = loadSelection();  // { [id]: boolean }

// -------------------- Settings --------------------
// Detector settings: keep normal pass strict-ish; fallback slightly looser (but not crazy).
const DETECTOR_ENROLL = { inputSize: 608, scoreThreshold: 0.12 };
const DETECTOR_SCAN   = { inputSize: 608, scoreThreshold: 0.12 };

// Fallback: more sensitive, but still avoids tons of bogus detections.
const DETECTOR_FALLBACK = { inputSize: 608, scoreThreshold: 0.18 };

const ENROLL_MIN_SIDE = 700;
const SCAN_MIN_SIDE   = 900;

const MAX_FACES_PER_IMAGE = 30;

// Start strict for “only show real matches”
const DEFAULT_THRESHOLD = 0.55;

// Plausibility filters (these kill "arm face" boxes)
const MIN_FACE_PX = 40;            // too small → ignore
const MIN_DET_SCORE = 0.35;        // too low confidence → ignore
const MIN_AR = 0.65;               // aspect ratio lower bound
const MAX_AR = 1.60;               // aspect ratio upper bound

// -------------------- Helpers --------------------
function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}
function setStatus(target, msg) {
  target.textContent = msg || "";
}
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"
  }[m]));
}

function savePeople() {
  localStorage.setItem("dnu_people_v2", JSON.stringify(people));
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
function selectedPeople() {
  // default true
  return people.filter((p) => scanSelection[p.id] !== false);
}
function syncSelection() {
  const known = new Set(people.map((p) => p.id));
  for (const p of people) {
    if (!(p.id in scanSelection)) scanSelection[p.id] = true;
  }
  for (const id of Object.keys(scanSelection)) {
    if (!known.has(id)) delete scanSelection[id];
  }
  saveSelection();
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

// UX-only number (not a true probability)
function distanceToConfidence(d, thr) {
  const max = thr + 0.25;
  const clamped = Math.max(0, Math.min(1, 1 - d / max));
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
  return new Promise((r) => setTimeout(r, 0));
}

// Upscale for better detection of small faces
function imageToDetectionCanvas(img, targetMinSide) {
  const minSide = Math.min(img.width, img.height);
  const scale = minSide < targetMinSide ? (targetMinSide / minSide) : 1;

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  return { canvas, scale };
}

async function detectAll(canvasOrImg, detectorOpts) {
  return await faceapi
    .detectAllFaces(canvasOrImg, new faceapi.TinyFaceDetectorOptions(detectorOpts))
    .withFaceLandmarks()
    .withFaceDescriptors();
}

// Filters bogus detections (prevents "arm as face")
function isPlausibleFace(det, detScale, img) {
  const b = det.detection.box;
  const score = det.detection.score ?? 0;

  // map to original image coords
  const w = b.width / detScale;
  const h = b.height / detScale;

  if (score < MIN_DET_SCORE) return false;

  const minSide = Math.min(w, h);
  if (minSide < MIN_FACE_PX) return false;

  const ar = w / h;
  if (ar < MIN_AR || ar > MAX_AR) return false;

  // ignore absurdly huge boxes
  if (w > img.width * 0.75 || h > img.height * 0.75) return false;

  return true;
}

// ✅ Closest match only (or null if no one passes threshold)
function bestMatchForDescriptor(descriptor, thr) {
  const pool = selectedPeople();

  let best = null;

  for (const p of pool) {
    if (!p.samples?.length) continue;

    let bestDist = Infinity;

    for (const s of p.samples) {
      const d = dist(descriptor, s);
      if (d < bestDist) bestDist = d;
      if (bestDist < 0.20) break; // early exit
    }

    if (!isFinite(bestDist)) continue;

    if (!best || bestDist < best.dist) {
      best = {
        personId: p.id,
        name: p.name,
        dist: bestDist,
        confidence: distanceToConfidence(bestDist, thr),
      };
    }
  }

  if (!best || best.dist > thr) return null;
  return best;
}

function dedupeNames(bestMatches) {
  const set = new Set(bestMatches.map((b) => b.name));
  return Array.from(set);
}

// -------------------- Rendering --------------------
function clearBuckets() {
  resultsFlagged.innerHTML = "";
  resultsClear.innerHTML = "";
  countFlagged.textContent = "0";
  countClear.textContent = "0";
}

function updateButtons() {
  btnAddPerson.disabled = !personName.value.trim();

  const pool = selectedPeople();
  const poolSamples = pool.reduce((s, p) => s + (p.samples?.length || 0), 0);

  btnScan.disabled =
    !modelsReady ||
    (scanInput.files?.length || 0) === 0 ||
    pool.length === 0 ||
    poolSamples === 0;
}

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
      people = people.filter((x) => x.id !== p.id);
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
      setStatus(enrollStatus, `Cleared samples for ${p.name}.`);
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

function renderAll() {
  renderPeople();
  renderScanPeople();
  updateButtons();
}
renderAll();

// -------------------- Model loading --------------------
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
    modelsReady = false;
    setStatus(modelStatus, "Model load failed. Check /models and that .bin files exist.");
  }
  updateButtons();
}
loadModels();

// -------------------- Defaults --------------------
if (threshold) {
  threshold.value = String(DEFAULT_THRESHOLD);
  thrVal.textContent = String(DEFAULT_THRESHOLD);
}

// -------------------- People controls --------------------
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

// -------------------- Threshold UI --------------------
threshold.addEventListener("input", () => {
  thrVal.textContent = threshold.value;
});

// -------------------- Dropzone --------------------
function setupDropzone(dropEl, onFiles) {
  dropEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropEl.classList.add("dragover");
  });
  dropEl.addEventListener("dragleave", () => dropEl.classList.remove("dragover"));
  dropEl.addEventListener("drop", (e) => {
    e.preventDefault();
    dropEl.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files || []).filter((f) => f.type.startsWith("image/"));
    onFiles(files);
  });
}
setupDropzone(enrollDrop, async (files) => {
  if (!files.length) return;
  await handleEnrollFiles(files);
});

// -------------------- Enroll --------------------
enrollInput.addEventListener("change", async () => {
  const files = Array.from(enrollInput.files || []);
  await handleEnrollFiles(files);
  enrollInput.value = "";
});

async function handleEnrollFiles(files) {
  if (!modelsReady) return setStatus(enrollStatus, "Models not ready yet.");
  const pid = personSelect.value;
  if (!pid) return setStatus(enrollStatus, "Select a person first (left panel or dropdown).");

  const person = people.find((p) => p.id === pid);
  if (!person) return setStatus(enrollStatus, "Selected person not found.");

  crops.innerHTML = "";
  setStatus(enrollStatus, `Detecting faces for: ${person.name}…`);

  let found = 0;

  for (const file of files) {
    const img = await fileToImage(file);

    const { canvas, scale } = imageToDetectionCanvas(img, ENROLL_MIN_SIDE);

    let detections = await detectAll(canvas, DETECTOR_ENROLL);

    if (!detections.length) {
      // try fallback if nothing was found
      detections = await detectAll(canvas, DETECTOR_FALLBACK);
    }

    if (!detections.length) continue;

    const trimmed = detections.slice(0, MAX_FACES_PER_IMAGE);

    for (const det of trimmed) {
      if (!isPlausibleFace(det, scale, img)) continue;

      found++;
      const box = det.detection.box;

      // map detection coords (upscaled) back to original
      const ox = box.x / scale;
      const oy = box.y / scale;
      const ow = box.width / scale;
      const oh = box.height / scale;

      // generous padding so descriptor sees more of the face
      const pad = Math.max(12, Math.round(Math.min(ow, oh) * 0.18));
      const x = Math.max(0, Math.floor(ox - pad));
      const y = Math.max(0, Math.floor(oy - pad));
      const w = Math.min(img.width - x, Math.floor(ow + pad * 2));
      const h = Math.min(img.height - y, Math.floor(oh + pad * 2));

      const crop = document.createElement("canvas");
      crop.width = 180;
      crop.height = 180;
      crop.getContext("2d").drawImage(img, x, y, w, h, 0, 0, crop.width, crop.height);
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

  if (!found) {
    setStatus(
      enrollStatus,
      "No faces found. Try a clearer/closer photo (face larger, less blur)."
    );
  } else {
    setStatus(enrollStatus, `Found ${found} face(s). Click crops to add samples to ${person.name}.`);
  }
}

// -------------------- Scan --------------------
scanInput.addEventListener("change", updateButtons);

btnScan.addEventListener("click", async () => {
  if (!modelsReady) return setStatus(scanStatus, "Models not ready yet.");

  const files = Array.from(scanInput.files || []);
  if (!files.length) return;

  const pool = selectedPeople();
  const poolSamples = pool.reduce((s, p) => s + (p.samples?.length || 0), 0);
  if (!pool.length || poolSamples === 0) {
    return setStatus(scanStatus, "No selected people with samples. Check filters and enroll samples first.");
  }

  const thr = parseFloat(threshold.value);

  clearBuckets();
  setStatus(scanStatus, `Scanning ${files.length} photo(s)…`);

  let flaggedCount = 0;
  let clearCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    setStatus(scanStatus, `Scanning ${i + 1} / ${files.length}…`);

    const img = await fileToImage(file);

    const { canvas, scale } = imageToDetectionCanvas(img, SCAN_MIN_SIDE);

    let detections = await detectAll(canvas, DETECTOR_SCAN);

    if (!detections.length) {
      detections = await detectAll(canvas, DETECTOR_FALLBACK);
    }

    const trimmed = detections.slice(0, MAX_FACES_PER_IMAGE);

    // compute best match only for plausible detections
    const faceResults = [];
    for (const det of trimmed) {
      if (!isPlausibleFace(det, scale, img)) continue;

      const desc = Array.from(det.descriptor);
      const best = bestMatchForDescriptor(desc, thr);
      faceResults.push({ det, best, scale });
    }

    const matchedFaces = faceResults.filter((fr) => fr.best);

    const status = matchedFaces.length ? "flagged" : "clear";

    const names = matchedFaces.length ? dedupeNames(matchedFaces.map((m) => m.best)) : [];
    const label = names.length
      ? `${names.slice(0, 3).join(", ")}${names.length > 3 ? ` +${names.length - 3}` : ""}`
      : "—";

    const row = document.createElement("div");
    row.className = "result";

    const left = document.createElement("div");
    left.innerHTML = `
      <div class="result__name">${escapeHtml(file.name)}</div>
      <div class="result__sub">
        ${trimmed.length} face(s) detected • ${status === "clear" ? "No matches" : `Matched: ${escapeHtml(label)}`}
      </div>
    `;

    const right = document.createElement("div");
    right.className = "result__right";

    const tag = document.createElement("span");
    tag.className = `tag ${status === "flagged" ? "tag--bad" : "tag--good"}`;
    tag.textContent = status === "flagged" ? "MATCH" : "CLEAR";

    const btnPrev = document.createElement("button");
    btnPrev.className = "btn btn--ghost btn--sm";
    btnPrev.textContent = "Preview";
    btnPrev.onclick = () => drawPreview(img, faceResults);

    right.appendChild(tag);
    right.appendChild(btnPrev);

    row.appendChild(left);
    row.appendChild(right);

    if (status === "flagged") {
      resultsFlagged.appendChild(row);
      flaggedCount++;
      if (flaggedCount === 1) drawPreview(img, faceResults);
    } else {
      resultsClear.appendChild(row);
      clearCount++;
      if (flaggedCount === 0 && clearCount === 1) drawPreview(img, faceResults);
    }

    countFlagged.textContent = String(flaggedCount);
    countClear.textContent = String(clearCount);

    await yieldToUI();
  }

  setStatus(
    scanStatus,
    `Done. Matched ${flaggedCount} • Clear ${clearCount}. Threshold ${thr.toFixed(2)}.`
  );
});

// -------------------- Preview --------------------
function clearPreview() {
  const ctx = previewCanvas.getContext("2d");
  previewCanvas.width = 1;
  previewCanvas.height = 1;
  ctx.clearRect(0, 0, 1, 1);
}

function drawPreview(img, faceResults) {
  const ctx = previewCanvas.getContext("2d");
  const maxW = 1100;
  const scaleCanvas = Math.min(1, maxW / img.width);

  previewCanvas.width = Math.round(img.width * scaleCanvas);
  previewCanvas.height = Math.round(img.height * scaleCanvas);

  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);

  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  ctx.textBaseline = "top";

  // ✅ ONLY DRAW MATCHED FACES
  for (const fr of faceResults) {
    if (!fr.best) continue;

    const detScale = fr.scale || 1;
    const b = fr.det.detection.box;

    const ox = b.x / detScale;
    const oy = b.y / detScale;
    const ow = b.width / detScale;
    const oh = b.height / detScale;

    const x = ox * scaleCanvas;
    const y = oy * scaleCanvas;
    const w = ow * scaleCanvas;
    const h = oh * scaleCanvas;

    const stroke = "#b00020"; // red
    ctx.strokeStyle = stroke;
    ctx.strokeRect(x, y, w, h);

    const label = `MATCH: ${fr.best.name} (${fr.best.confidence}%)`;

    const pad = 4;
    const tw = ctx.measureText(label).width;

    ctx.fillStyle = stroke;
    ctx.fillRect(x, y, tw + pad * 2, 18 + pad);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + pad, y + pad);
  }
}

// -------------------- Export / Import --------------------
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

// -------------------- Help dialog --------------------
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

// -------------------- Init --------------------
syncSelection();
renderAll();
