(() => {
  "use strict";

  const LANGS = {
    vi: { name: "Tiếng Việt", code: "VI" },
    en: { name: "English", code: "EN" },
    zh: { name: "中文 · Mandarin", code: "ZH" },
    ko: { name: "한국어 · Korean", code: "KO" },
  };
  let PAIRS = [["vi", "en"], ["en", "vi"], ["vi", "zh"], ["zh", "vi"], ["vi", "ko"], ["ko", "vi"]];

  const TARGET_SR = 16000;
  const MIN_SEC = 0.25;
  const MAX_SEC = 30;

  const $ = (s, p = document) => p.querySelector(s);
  const app = $("#app");
  const feed = $("#feed");
  const empty = $("#empty");
  const tpl = $("#turnTpl");
  const srcSel = $("#srcLang");
  const tgtSel = $("#tgtLang");
  const swapBtn = $("#swapBtn");
  const micBtn = $("#micBtn");
  const halo = $("#halo");
  const statusEl = $("#status");
  const stagesEl = $("#stages");
  const fileInput = $("#fileInput");
  const canvas = $("#wave");
  const ctx2d = canvas.getContext("2d");
  const stackInfoEl = $("#stackInfo");
  const textForm = $("#textForm");
  const textInput = $("#textInput");
  const sendTextBtn = $("#sendTextBtn");
  const dropOverlay = $("#dropOverlay");

  // ---------- theme ----------
  const root = document.documentElement;
  $("#themeBtn").addEventListener("click", () => {
    const dark = root.dataset.theme === "dark";
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch {}
  });

  // ---------- autoplay ----------
  let autoplay = true;
  try { autoplay = localStorage.getItem("autoplay") !== "0"; } catch {}
  const autoplayBtn = $("#autoplayBtn");
  const syncAutoplay = () => {
    autoplayBtn.setAttribute("aria-pressed", String(autoplay));
    autoplayBtn.title = "Tự động phát giọng dịch: " + (autoplay ? "Bật" : "Tắt");
  };
  syncAutoplay();
  autoplayBtn.addEventListener("click", () => {
    autoplay = !autoplay;
    try { localStorage.setItem("autoplay", autoplay ? "1" : "0"); } catch {}
    syncAutoplay();
  });

  // ---------- language pairs ----------
  const homeLang = "vi";
  const partnerOf = (s) => PAIRS.filter(([a]) => a === s).map(([, b]) => b);

  function fillSelect(sel, codes, value) {
    sel.innerHTML = "";
    for (const c of codes) {
      const o = new Option(LANGS[c]?.name || c, c);
      sel.add(o);
    }
    sel.value = codes.includes(value) ? value : codes[0];
  }

  function syncQuickPairChips() {
    const [s, t] = pair0();
    document.querySelectorAll(".pair-chip").forEach((b) => {
      const [a, c] = b.dataset.pair.split("-");
      const active = (s === a && t === c) || (s === c && t === a);
      b.classList.toggle("active", active);
    });
  }

  function setPair(src, tgt) {
    fillSelect(srcSel, Object.keys(LANGS), src);
    fillSelect(tgtSel, partnerOf(srcSel.value), tgt);
    try { localStorage.setItem("pair", srcSel.value + "-" + tgtSel.value); } catch {}
    syncQuickPairChips();
  }

  const pair0 = () => [srcSel.value, tgtSel.value];
  {
    let saved = "vi-en";
    try { saved = localStorage.getItem("pair") || saved; } catch {}
    const [a, b] = saved.split("-");
    setPair(LANGS[a] ? a : "vi", b);
  }

  srcSel.addEventListener("change", () => setPair(srcSel.value, tgtSel.value));
  tgtSel.addEventListener("change", () => {
    try { localStorage.setItem("pair", srcSel.value + "-" + tgtSel.value); } catch {}
    syncQuickPairChips();
  });
  swapBtn.addEventListener("click", () => {
    const [s, t] = pair0();
    setPair(t, s);
    swapBtn.classList.toggle("spin");
  });

  document.querySelectorAll(".pair-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const [a, b] = btn.dataset.pair.split("-");
      const [s, t] = pair0();
      if (s === a && t === b) setPair(b, a);
      else setPair(a, b);
    });
  });

  // ---------- health check ----------
  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (Array.isArray(h.pairs) && h.pairs.length) {
      PAIRS = h.pairs;
      setPair(srcSel.value, tgtSel.value);
    }
    const pill = $("#enginePill");
    const isMock = (h.engine || "").toLowerCase().includes("mock");
    pill.classList.add(isMock ? "mock" : "ok");
    $("#engineName").textContent = (h.engine || "Ready") + " · Offline";
    const info = h.info || {};
    const parts = [];
    if (info.asr) parts.push(`ASR: ${info.asr}`);
    if (info.mt) parts.push(`MT: ${info.mt}`);
    if (info.tts) parts.push(`TTS: ${info.tts}`);
    if (parts.length && stackInfoEl) {
      stackInfoEl.textContent = parts.join(" · ");
      pill.title = parts.join("\n");
    } else if (stackInfoEl) {
      stackInfoEl.textContent = isMock ? "Chế độ mô phỏng UI (Mock)" : "Pipeline NPU / CPU Offline";
    }
  }).catch(() => {
    $("#engineName").textContent = "Mất kết nối server";
    toast("Không kết nối được server /api/health");
  });

  let toastT;
  function toast(msg) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toastT);
    toastT = setTimeout(() => el.classList.remove("show"), 2600);
  }

  // ---------- state ----------
  let state = "idle"; // idle | recording | busy
  let busyTimerId = null;
  let busyStart = 0;

  function setState(s, hintText) {
    state = s;
    app.classList.toggle("recording", s === "recording");
    app.classList.toggle("busy", s === "busy");
    micBtn.setAttribute("aria-label", s === "recording" ? "Dừng và dịch" : s === "busy" ? "Đang xử lý" : "Bắt đầu ghi âm");
    sendTextBtn.disabled = s !== "idle";
    if (s === "idle") {
      clearInterval(busyTimerId);
      busyTimerId = null;
      statusEl.textContent = "Sẵn sàng";
    } else if (s === "busy" && hintText) {
      statusEl.textContent = hintText;
    }
  }

  function startBusyClock(labelFn) {
    clearInterval(busyTimerId);
    busyStart = performance.now();
    statusEl.textContent = labelFn(0);
    busyTimerId = setInterval(() => {
      const sec = (performance.now() - busyStart) / 1000;
      statusEl.textContent = labelFn(sec);
    }, 250);
  }

  // ---------- recording (raw PCM via AudioWorklet) ----------
  let audioCtx, stream, srcNode, worklet, analyser, chunks = [], recStart = 0, rafId, timerId;
  const WORKLET = `
    class Rec extends AudioWorkletProcessor {
      process(inputs) { const c = inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
    }
    registerProcessor("rec", Rec);`;

  async function ensureCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await audioCtx.audioWorklet.addModule(url);
    }
    if (audioCtx.state === "suspended") await audioCtx.resume();
    return audioCtx;
  }

  async function startRec() {
    if (state !== "idle") return;
    try {
      await ensureCtx();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e) {
      toast(e && e.name === "NotAllowedError" ? "Cần cấp quyền micro để ghi âm" : "Không mở được micro");
      return;
    }
    chunks = [];
    srcNode = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = 0.6;
    worklet = new AudioWorkletNode(audioCtx, "rec");
    worklet.port.onmessage = (e) => chunks.push(e.data);
    srcNode.connect(analyser); srcNode.connect(worklet);
    const mute = audioCtx.createGain(); mute.gain.value = 0; worklet.connect(mute).connect(audioCtx.destination);
    recStart = performance.now();
    levels.fill(0);
    setState("recording");
    sizeCanvas();
    hideStages();
    tick();
    timerId = setInterval(() => {
      const s = (performance.now() - recStart) / 1000;
      statusEl.textContent = "Đang thu âm… " + fmt(s) + " (nhấn Space hoặc nút đỏ để dừng)";
      if (s >= MAX_SEC) stopRec();
    }, 100);
    statusEl.textContent = "Đang thu âm… 0:00";
  }

  function teardown() {
    cancelAnimationFrame(rafId); clearInterval(timerId);
    halo.style.transform = "";
    try { srcNode && srcNode.disconnect(); worklet && worklet.disconnect(); } catch {}
    stream && stream.getTracks().forEach((t) => t.stop());
    stream = srcNode = worklet = null;
  }

  async function stopRec(cancel = false) {
    if (state !== "recording") return;
    const rate = audioCtx.sampleRate;
    teardown();
    const len = chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(len); let o = 0;
    for (const c of chunks) { pcm.set(c, o); o += c.length; }
    chunks = [];
    if (cancel) { setState("idle"); toast("Đã huỷ ghi âm"); return; }
    if (len / rate < MIN_SEC) { setState("idle"); toast("Đoạn ghi quá ngắn — hãy nói lâu hơn"); return; }
    const wav = encodeWav(await resample(pcm, rate), TARGET_SR);
    run(wav);
  }

  async function resample(pcm, rate) {
    if (rate === TARGET_SR) return pcm;
    const off = new OfflineAudioContext(1, Math.ceil(pcm.length * TARGET_SR / rate), TARGET_SR);
    const buf = off.createBuffer(1, pcm.length, rate); buf.copyToChannel(pcm, 0);
    const s = off.createBufferSource(); s.buffer = buf; s.connect(off.destination); s.start();
    return (await off.startRendering()).getChannelData(0);
  }

  function encodeWav(f32, sr) {
    const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, "data"); v.setUint32(40, f32.length * 2, true);
    for (let i = 0, o = 44; i < f32.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, f32[i])); v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  // ---------- visualizer ----------
  const BARS = 44;
  const levels = new Float32Array(BARS);
  const td = new Float32Array(1024);
  function sizeCanvas() {
    const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    if (!r.width || !r.height) return;
    canvas.width = Math.round(r.width * d);
    canvas.height = Math.round(r.height * d);
    ctx2d.setTransform(d, 0, 0, d, 0, 0);
  }
  sizeCanvas(); window.addEventListener("resize", sizeCanvas);
  if (window.matchMedia) {
    try { window.matchMedia("(resolution: 1dppx)").addEventListener("change", sizeCanvas); } catch {}
  }

  let frame = 0;
  function tick() {
    analyser.getFloatTimeDomainData(td);
    let sum = 0; for (let i = 0; i < td.length; i++) sum += td[i] * td[i];
    const rms = Math.min(1, Math.sqrt(sum / td.length) * 6);
    halo.style.transform = `scale(${1 + rms * 0.52})`;
    if ((frame++ & 1) === 0) { levels.copyWithin(0, 1); levels[BARS - 1] = rms; }
    draw();
    rafId = requestAnimationFrame(tick);
  }
  function draw() {
    const W = canvas.clientWidth, H = canvas.clientHeight, mid = H / 2;
    ctx2d.clearRect(0, 0, W, H);
    const color = getComputedStyle(document.documentElement).getPropertyValue("--rec").trim();
    const half = BARS / 2, gap = 50, bw = 3, step = (W / 2 - gap) / half;
    for (let i = 0; i < half; i++) {
      const lv = levels[BARS - 1 - i * 2] || 0;
      const h = Math.max(3, lv * (H - 10));
      ctx2d.globalAlpha = 0.88 - (i / half) * 0.65;
      ctx2d.fillStyle = color;
      for (const x of [W / 2 + gap + i * step, W / 2 - gap - i * step - bw]) {
        roundRect(x, mid - h / 2, bw, h, 1.5);
      }
    }
    ctx2d.globalAlpha = 1;
  }
  function roundRect(x, y, w, h, r) {
    ctx2d.beginPath();
    ctx2d.roundRect ? ctx2d.roundRect(x, y, w, h, r) : ctx2d.rect(x, y, w, h);
    ctx2d.fill();
  }

  // ---------- stages ----------
  const stageLis = {};
  stagesEl.querySelectorAll("li").forEach((li) => (stageLis[li.dataset.stage] = li));
  function resetStages() {
    for (const li of Object.values(stageLis)) { li.className = ""; li.querySelector(".st-ms").textContent = ""; }
    stagesEl.classList.add("show");
  }
  function hideStages() { stagesEl.classList.remove("show"); }
  function stageActive(k) { stageLis[k].className = "active"; }
  function stageDone(k, ms) { stageLis[k].className = "done"; stageLis[k].querySelector(".st-ms").textContent = typeof ms === "number" ? ms + " ms" : ms; }

  // ---------- turns ----------
  let turnCount = 0;
  function createTurn(src, tgt, inputBlob, modeLabel) {
    empty.hidden = true;
    const el = tpl.content.firstElementChild.cloneNode(true);
    el.classList.add(src === homeLang ? "right" : "left");
    el.querySelector(".m-src").textContent = LANGS[src]?.code || src.toUpperCase();
    el.querySelector(".m-tgt").textContent = LANGS[tgt]?.code || tgt.toUpperCase();
    el.querySelector(".m-time").textContent = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    if (modeLabel) {
      const m = el.querySelector(".m-mode");
      m.textContent = modeLabel;
      m.hidden = false;
    }
    if (inputBlob) {
      const origUrl = URL.createObjectURL(inputBlob);
      const inBtn = el.querySelector(".play-in");
      inBtn.hidden = false;
      setDuration(inBtn, origUrl);
      inBtn.addEventListener("click", () => player.toggle(inBtn, origUrl));
    }
    el.dataset.src = src; el.dataset.tgt = tgt;
    el._stageMs = {};
    wireEdit(el);
    feed.appendChild(el);
    turnCount++;
    requestAnimationFrame(() => el.scrollIntoView({ behavior: "smooth", block: "end" }));
    return el;
  }

  function renderTurnStagePills(el) {
    const box = el.querySelector(".turn-stage-pills");
    const entries = Object.entries(el._stageMs || {});
    if (!entries.length) return;
    box.innerHTML = "";
    for (const [k, v] of entries) {
      const s = document.createElement("span");
      s.className = "stage-mini";
      s.textContent = `${k.toUpperCase()} ${v}ms`;
      box.appendChild(s);
    }
    box.hidden = false;
  }

  function setText(el, sel, text, lang) {
    const p = el.querySelector(sel);
    p.textContent = text; p.lang = lang; p.style.color = ""; p.style.fontSize = ""; p.classList.add("reveal");
  }

  function wireCopy(turn, getTranslated) {
    turn.querySelector(".a-copy").onclick = async (ev) => {
      try {
        await navigator.clipboard.writeText(turn._translated || getTranslated());
        const b = ev.currentTarget; b.classList.add("done");
        setTimeout(() => b.classList.remove("done"), 1200);
        toast("Đã sao chép bản dịch");
      } catch {
        toast("Không sao chép được");
      }
    };
  }

  // ---------- playback (one at a time) ----------
  const player = {
    audio: new Audio(), btn: null,
    toggle(btn, url) {
      if (this.btn === btn && this.audio.src === url) {
        if (this.audio.paused) this.audio.play(); else this.audio.pause();
        return;
      }
      this.stop();
      this.btn = btn; this.audio.src = url;
      this.audio.play().catch(() => toast("Trình duyệt chặn tự phát — bấm ▶ để nghe"));
    },
    stop() { this.audio.pause(); this.mark(false); if (this.btn) this.btn.querySelector(".bar-fill").style.width = "0"; },
    mark(on) { if (this.btn) this.btn.classList.toggle("playing", on); },
  };
  player.audio.addEventListener("play", () => player.mark(true));
  player.audio.addEventListener("pause", () => player.mark(false));
  player.audio.addEventListener("ended", () => { player.mark(false); if (player.btn) player.btn.querySelector(".bar-fill").style.width = "0"; });
  player.audio.addEventListener("timeupdate", () => {
    const a = player.audio; if (!player.btn || !a.duration) return;
    player.btn.querySelector(".bar-fill").style.width = (a.currentTime / a.duration) * 100 + "%";
  });

  // ---------- edit the recognised text, translate again (POST /api/text) ----------
  function wireEdit(turn) {
    const btn = turn.querySelector(".a-edit"), p = turn.querySelector(".src-text");
    const start = () => {
      turn.classList.add("editing"); p.contentEditable = "plaintext-only";
      if (p.contentEditable !== "plaintext-only") p.contentEditable = "true";
      p.focus();
      const r = document.createRange(); r.selectNodeContents(p); r.collapse(false);
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      statusEl.textContent = "Sửa chữ rồi bấm Enter hoặc ✓ để dịch lại (Esc: huỷ)";
    };
    const stop = () => { turn.classList.remove("editing"); p.contentEditable = "false"; };
    btn.addEventListener("click", () => (turn.classList.contains("editing") ? retranslate(turn) : (p.dataset.orig = p.textContent, start())));
    p.addEventListener("keydown", (e) => {
      if (!turn.classList.contains("editing")) return;
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); retranslate(turn); }
      else if (e.key === "Escape") { e.preventDefault(); p.textContent = p.dataset.orig || p.textContent; stop(); statusEl.textContent = "Sẵn sàng"; }
      e.stopPropagation();
    });
    p.addEventListener("keyup", (e) => { if (turn.classList.contains("editing")) e.stopPropagation(); });
    turn._stopEdit = stop;
  }

  async function readEvents(res, onEvent) {
    if (!res.ok || !res.body) {
      let m = "Server trả lỗi " + res.status;
      try { m += ": " + (await res.json()).error; } catch {}
      throw new Error(m);
    }
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true }); let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (line) { const ev = JSON.parse(line); if (ev.stage === "error") throw new Error(ev.message); onEvent(ev); }
      }
    }
  }

  async function retranslate(turn) {
    if (state !== "idle") return toast("Đang bận, chờ lượt hiện tại xong");
    const p = turn.querySelector(".src-text"), text = p.textContent.trim(), src = turn.dataset.src, tgt = turn.dataset.tgt;
    turn._stopEdit();
    if (!text) return toast("Chữ trống");
    setState("busy"); resetStages(); stageDone("asr", "sửa tay"); stageActive("mt");
    startBusyClock((s) => `Đang dịch lại… ${s.toFixed(1)}s`);
    const out = turn.querySelector(".tgt-text"); out.innerHTML = '<span class="skel"></span>'; turn.classList.remove("error");
    const pb = turn.querySelector(".play-out"); pb.disabled = true;
    turn._stageMs = {};
    try {
      const res = await fetch(`/api/text?src=${src}&tgt=${tgt}`, { method: "POST", body: JSON.stringify({ text }), headers: { "Content-Type": "application/json" } });
      await readEvents(res, (ev) => {
        if (ev.stage === "mt") {
          setText(turn, ".tgt-text", ev.text, tgt);
          turn._translated = ev.text; turn._stageMs.mt = ev.ms;
          stageDone("mt", ev.ms); stageActive("tts");
          startBusyClock((s) => `Đang tạo giọng nói (TTS)… ${s.toFixed(1)}s`);
          turn.querySelector(".a-copy").disabled = false;
        } else if (ev.stage === "tts") {
          turn._stageMs.tts = ev.ms; stageDone("tts", ev.ms); attachAudio(turn, ev.audio);
        } else if (ev.stage === "done") {
          turn.querySelector(".m-lat").textContent = (ev.total_ms / 1000).toFixed(2) + " s (sửa)";
          renderTurnStagePills(turn);
        }
      });
      if (!turn.querySelector(".edited-note")) { const n = document.createElement("small"); n.className = "edited-note"; n.textContent = "Đã sửa tay & dịch lại"; p.after(n); }
    } catch (e) {
      out.textContent = "Không dịch được: " + (e.message || ""); out.style.color = "var(--rec)"; out.style.fontSize = "14px";
      turn.classList.add("error"); toast("Lỗi: " + (e.message || "không rõ"));
      for (const li of Object.values(stageLis)) if (li.className === "active") li.className = "";
    }
    setState("idle");
    setTimeout(() => { if (state === "idle") hideStages(); }, 4000);
  }

  // ---------- quick text translation (POST /api/text) ----------
  async function runText(rawText) {
    const text = (rawText || "").trim();
    if (!text) return;
    if (state !== "idle") return toast("Đang bận, chờ lượt hiện tại xong");
    const [src, tgt] = pair0();
    setState("busy"); resetStages(); stageDone("asr", "nhập tay"); stageActive("mt");
    startBusyClock((s) => `Đang dịch máy (${s.toFixed(1)}s)…`);
    const turn = createTurn(src, tgt, null, "Văn bản");
    setText(turn, ".src-text", text, src);
    turn.querySelector(".a-edit").disabled = false;
    let translated = "";
    wireCopy(turn, () => translated);

    try {
      const res = await fetch(`/api/text?src=${src}&tgt=${tgt}`, { method: "POST", body: JSON.stringify({ text }), headers: { "Content-Type": "application/json" } });
      await readEvents(res, (ev) => {
        if (ev.stage === "mt") {
          translated = ev.text; turn._translated = ev.text; turn._stageMs.mt = ev.ms;
          setText(turn, ".tgt-text", ev.text, tgt); stageDone("mt", ev.ms); stageActive("tts");
          startBusyClock((s) => `Đang tổng hợp giọng nói (TTS)… ${s.toFixed(1)}s`);
          turn.querySelector(".a-copy").disabled = false;
        } else if (ev.stage === "tts") {
          turn._stageMs.tts = ev.ms; stageDone("tts", ev.ms); attachAudio(turn, ev.audio);
        } else if (ev.stage === "done") {
          turn.querySelector(".m-lat").textContent = (ev.total_ms / 1000).toFixed(2) + " s";
          renderTurnStagePills(turn);
        }
      });
    } catch (e) {
      turn.classList.add("error");
      turn.querySelectorAll(".skel").forEach((s) => s.remove());
      const p = turn.querySelector(".tgt-text");
      if (!translated) { p.textContent = "Không dịch được: " + (e.message || ""); p.style.color = "var(--rec)"; p.style.fontSize = "14px"; }
      for (const li of Object.values(stageLis)) if (li.className === "active") li.className = "";
      toast("Lỗi: " + (e.message || "không rõ"));
    }
    setState("idle");
    setTimeout(() => { if (state === "idle") hideStages(); }, 4000);
  }

  // ---------- speech pipeline call (POST /api/translate) ----------
  async function run(wavBlob) {
    const [src, tgt] = pair0();
    setState("busy"); resetStages(); stageActive("asr");
    startBusyClock((s) => `Đang nhận dạng giọng nói (ASR)… ${s.toFixed(1)}s`);
    const turn = createTurn(src, tgt, wavBlob, null);
    let translated = "";
    wireCopy(turn, () => translated);

    try {
      const res = await fetch(`/api/translate?src=${src}&tgt=${tgt}`, { method: "POST", body: wavBlob, headers: { "Content-Type": "audio/wav" } });
      await readEvents(res, (ev) => {
        if (ev.stage === "asr") {
          turn._stageMs.asr = ev.ms;
          setText(turn, ".src-text", ev.text, src); stageDone("asr", ev.ms); stageActive("mt");
          startBusyClock((s) => s > 4 ? `Đang dịch máy NLLB (lần đầu nạp model chờ ~20s)… ${s.toFixed(0)}s` : `Đang dịch máy… ${s.toFixed(1)}s`);
          turn.querySelector(".a-edit").disabled = false;
        } else if (ev.stage === "mt") {
          translated = ev.text; turn._translated = ev.text; turn._stageMs.mt = ev.ms;
          setText(turn, ".tgt-text", ev.text, tgt); stageDone("mt", ev.ms); stageActive("tts");
          startBusyClock((s) => `Đang tổng hợp giọng nói (TTS)… ${s.toFixed(1)}s`);
          turn.querySelector(".a-copy").disabled = false;
        } else if (ev.stage === "tts") {
          turn._stageMs.tts = ev.ms; stageDone("tts", ev.ms); attachAudio(turn, ev.audio);
        } else if (ev.stage === "done") {
          turn.querySelector(".m-lat").textContent = (ev.total_ms / 1000).toFixed(2) + " s";
          renderTurnStagePills(turn);
        }
      });
    } catch (e) {
      turn.classList.add("error");
      turn.querySelectorAll(".skel").forEach((s) => s.remove());
      const p = turn.querySelector(".tgt-text");
      if (!translated) { p.textContent = "Không dịch được: " + (e.message || ""); p.style.color = "var(--rec)"; p.style.fontSize = "14px"; }
      for (const li of Object.values(stageLis)) if (li.className === "active") li.className = "";
      toast("Lỗi: " + (e.message || "không rõ"));
    }
    setState("idle");
    setTimeout(() => { if (state === "idle") hideStages(); }, 4000);
  }

  function attachAudio(turn, b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    const btn = turn.querySelector(".play-out");
    btn.disabled = false;
    setDuration(btn, url);
    btn.addEventListener("click", () => player.toggle(btn, url));
    if (autoplay) player.toggle(btn, url);
  }

  function setDuration(btn, url) {
    const probe = new Audio(url);
    probe.addEventListener("loadedmetadata", () => (btn.querySelector(".dur").textContent = fmt(probe.duration)));
  }

  const fmt = (s) => { s = Math.max(0, s || 0); return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0"); };

  // ---------- inputs ----------
  textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const val = textInput.value.trim();
    if (!val || state !== "idle") return;
    textInput.value = "";
    runText(val);
  });

  micBtn.addEventListener("click", () => {
    if (state === "idle") startRec(); else if (state === "recording") stopRec();
  });

  let spaceHeld = false;
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !isTyping(e)) {
      e.preventDefault();
      if (state === "idle") { spaceHeld = true; startRec(); }
    } else if (e.key === "Escape" && state === "recording") stopRec(true);
  });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space" && spaceHeld) { e.preventDefault(); spaceHeld = false; stopRec(); }
  });
  const isTyping = (e) => /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable;

  $("#uploadBtn").addEventListener("click", () => state === "idle" && fileInput.click());
  fileInput.addEventListener("change", async () => {
    const f = fileInput.files[0]; fileInput.value = "";
    if (!f) return;
    try {
      await ensureCtx();
      const ab = await audioCtx.decodeAudioData(await f.arrayBuffer());
      const ch = ab.numberOfChannels, mono = new Float32Array(ab.length);
      for (let c = 0; c < ch; c++) { const d = ab.getChannelData(c); for (let i = 0; i < d.length; i++) mono[i] += d[i] / ch; }
      run(encodeWav(await resample(mono, ab.sampleRate), TARGET_SR));
    } catch { toast("Không đọc được file âm thanh này"); }
  });

  // drag & drop an audio file anywhere with visual overlay
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    if (state === "idle") { dragDepth++; dropOverlay.classList.add("show"); }
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropOverlay.classList.remove("show");
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0; dropOverlay.classList.remove("show");
    const f = e.dataTransfer.files[0];
    if (f && (f.type.startsWith("audio/") || /\.(wav|mp3|flac|m4a|ogg)$/i.test(f.name)) && state === "idle") {
      const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files; fileInput.dispatchEvent(new Event("change"));
    } else if (f) {
      toast("Vui lòng chọn file âm thanh (.wav, .mp3, .flac)");
    }
  });

  $("#clearBtn").addEventListener("click", () => {
    if (state !== "idle" || !turnCount) return;
    player.stop();
    feed.querySelectorAll(".turn").forEach((t) => t.remove());
    turnCount = 0; empty.hidden = false;
  });

  setState("idle");
})();
