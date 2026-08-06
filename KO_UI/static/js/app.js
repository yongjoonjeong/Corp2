const state = {
  users: [],
  currentUser: null,
  sessions: [],
  database: { ok: false, users: 0, sessions: 0, path: "instance/ko.sqlite3" },
  currentScreen: "home",
  stream: null,
  poseLandmarker: null,
  poseLoadFailed: false,
  poseLoopToken: 0,
  lastPoseResult: null,
  measurement: {
    stage: "wingspan",
    collecting: false,
    samples: [],
    values: {},
  },
  trainingConfig: {
    type: "straight",
    hand: "right",
    durationSec: 60,
    difficulty: "normal",
  },
  training: {
    running: false,
    paused: false,
    remainingSec: 60,
    punches: 0,
    successful: 0,
    prompts: 0,
    reactions: [],
    promptStartedAt: null,
    timerId: null,
    promptId: null,
    armReady: true,
    lastPunchAt: 0,
  },
  vision: {
    connected: false,
    previewAvailable: false,
    evidenceAvailable: false,
    previewVersion: 0,
    evidenceVersion: 0,
    lastEventId: 0,
    statusTimer: null,
    previewTimer: null,
    eventTimer: null,
    statusBusy: false,
    previewBusy: false,
    eventBusy: false,
    liveStatus: {},
  },
  sttConfigured: false,
  wake: {
    available: false,
    running: false,
    enabled: true,
    state: "starting",
    message: "음성 기능 준비 중",
    display_name: "웨이크 업 케이오",
    lastEventId: 0,
    pollTimer: null,
    pollBusy: false,
    session_active: false,
    session_remaining_sec: 0,
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function showToast(message, type = "normal") {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show${type === "error" ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.className = "toast", 2600);
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const estimatedMs = Math.max(2200, String(text).length * 105);
  const playSpeech = () => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    utterance.rate = 1.03;
    utterance.pitch = 0.92;
    window.speechSynthesis.speak(utterance);
  };
  // Tell the local listener to ignore the speaker output before TTS begins.
  api("/api/wakeword/suppress", {
    method: "POST",
    body: JSON.stringify({ duration_ms: estimatedMs + 900 }),
  }).catch(() => {}).finally(playSpeech);
}

function showScreen(name) {
  state.currentScreen = name;
  $$(".screen").forEach((screen) => screen.classList.toggle("active", screen.id === `screen-${name}`));
  $$(".sport-nav-item").forEach((button) => button.classList.toggle("active", button.dataset.screen === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (!new Set(["measure", "ready", "training"]).has(name)) stopPoseLoop();

  // A wake word opens one conversation. Measurement and training keep that
  // conversation alive longer so gloves-on users do not need to call KO again.
  if (state.wake.session_active) {
    if (name === "measure" || name === "measure-result") extendVoiceSession(300);
    else if (name === "ready") extendVoiceSession(180);
    else if (name === "training") extendVoiceSession(Math.max(180, state.training.remainingSec + 120));
  }
}

async function api(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const config = { ...options, headers };
  const response = await fetch(url, config);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했습니다.");
  return data;
}

async function extendVoiceSession(durationSec = 30) {
  if (!state.wake.session_active) return null;
  try {
    const result = await api("/api/wakeword/session", {
      method: "POST",
      body: JSON.stringify({ action: "extend", duration_sec: durationSec }),
    });
    Object.assign(state.wake, result);
    updateWakeUi();
    return result;
  } catch (error) {
    console.warn("Voice session extension failed", error);
    return null;
  }
}

async function ensureVoiceSession(durationSec = 30) {
  try {
    const result = await api("/api/wakeword/session", {
      method: "POST",
      body: JSON.stringify({
        action: state.wake.session_active ? "extend" : "start",
        duration_sec: durationSec,
      }),
    });
    Object.assign(state.wake, result);
    updateWakeUi();
    return result;
  } catch (error) {
    console.warn("Voice session start failed", error);
    return null;
  }
}

async function endVoiceSession(reason = "ui") {
  try {
    const result = await api("/api/wakeword/session", {
      method: "POST",
      body: JSON.stringify({ action: "end", reason }),
    });
    Object.assign(state.wake, result);
    updateWakeUi();
    return result;
  } catch (error) {
    console.warn("Voice session end failed", error);
    return null;
  }
}

async function loadUsers() {
  try {
    state.users = await api("/api/users");
    renderProfiles();
    renderSidebarUser();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderSidebarUser() {
  const name = $("#sidebarUserName");
  const meta = $("#sidebarUserMeta");
  const avatar = $("#sidebarAvatar");
  if (!name || !meta || !avatar) return;
  if (!state.currentUser) {
    name.textContent = "사용자 미선택";
    meta.textContent = "프로필을 선택하세요";
    avatar.textContent = "K";
    return;
  }
  name.textContent = state.currentUser.name;
  const hand = state.currentUser.dominant_hand === "right" ? "오른손잡이" : "왼손잡이";
  meta.textContent = `${Math.round(state.currentUser.height_cm)}cm · ${hand}`;
  avatar.textContent = [...state.currentUser.name][0] || "K";
}

function renderProfiles() {
  const grid = $("#profileGrid");
  const empty = $("#profileEmpty");
  grid.innerHTML = "";
  empty.classList.toggle("hidden", state.users.length > 0);
  grid.classList.toggle("hidden", state.users.length === 0);

  state.users.forEach((user) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "profile-card";
    const hand = user.dominant_hand === "right" ? "오른손" : "왼손";
    const reach = user.wingspan_cm ? `${Math.round(user.wingspan_cm)}cm` : "미측정";
    const initial = [...user.name][0] || "R";
    card.innerHTML = `
      <span class="profile-arrow">→</span>
      <div class="profile-avatar">${escapeHtml(initial)}</div>
      <h3>${escapeHtml(user.name)}</h3>
      <p>${Math.round(user.height_cm)}cm · ${hand}</p>
      <div class="profile-specs">
        <span>양팔 리치<strong>${reach}</strong></span>
        <span>최근 훈련<strong>${formatDate(user.last_training_at)}</strong></span>
      </div>`;
    card.addEventListener("click", () => selectUser(user));
    grid.appendChild(card);
  });
  renderHomeRecentUser();
}

function renderHomeRecentUser() {
  const card = $("#homeRecentCard");
  const empty = $("#homeRecentEmpty");
  const recentButton = $("#homeRecentLoadButton");
  const primaryButton = $("#homePrimaryAction");
  if (!card || !empty || !primaryButton) return;
  const recent = state.users[0] || null;
  card.classList.toggle("hidden", !recent);
  empty.classList.toggle("hidden", Boolean(recent));
  recentButton?.classList.toggle("hidden", !recent);

  if (!recent) {
    $("#homePrimaryTitle").textContent = "첫 사용자 등록하기";
    $("#homePrimarySubtext").textContent = "프로필을 만들고 맞춤 훈련을 시작하세요.";
    primaryButton.onclick = () => showScreen("register");
    setHomeMetric("#homeMetricAccuracy", "—");
    setHomeMetric("#homeMetricReaction", "—");
    setHomeMetric("#homeMetricPunches", "—");
    return;
  }

  const hand = recent.dominant_hand === "right" ? "오른손잡이" : "왼손잡이";
  $("#homeRecentAvatar").textContent = [...recent.name][0] || "K";
  $("#homeRecentName").textContent = recent.name;
  $("#homeRecentMeta").textContent = `${Math.round(recent.height_cm)}cm · ${hand}`;
  $("#homeRecentDate").textContent = recent.last_training_at
    ? `최근 훈련 ${new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(recent.last_training_at))}`
    : "아직 훈련 기록이 없습니다";
  $("#homePrimaryTitle").textContent = `${recent.name}님으로 계속하기`;
  $("#homePrimarySubtext").textContent = recent.wingspan_cm
    ? "저장된 리치와 최근 기록을 불러와 바로 준비합니다."
    : "프로필을 불러온 뒤 리치 측정을 진행합니다.";
  primaryButton.onclick = () => selectUser(recent);
  if (recentButton) recentButton.onclick = () => selectUser(recent);
  loadHomeRecentSession(recent.id);
}

function trainingTypeLabel(type) {
  return ({ jab: "잽", straight: "스트레이트", hook: "훅", uppercut: "어퍼컷" })[type] || "스트레이트";
}

function setHomeMetric(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

async function loadHomeRecentSession(userId) {
  const requestedUserId = Number(userId);
  setHomeMetric("#homeMetricAccuracy", "—");
  setHomeMetric("#homeMetricReaction", "—");
  setHomeMetric("#homeMetricPunches", "—");
  try {
    const sessions = await api(`/api/users/${requestedUserId}/sessions`);
    if (Number(state.users[0]?.id) !== requestedUserId || !sessions.length) return;
    const latest = sessions[0];
    setHomeMetric("#homeMetricAccuracy", Number(latest.success_rate || 0).toFixed(0));
    setHomeMetric("#homeMetricReaction", latest.avg_reaction_ms ? (Number(latest.avg_reaction_ms) / 1000).toFixed(2) : "—");
    setHomeMetric("#homeMetricPunches", String(latest.punch_count ?? "—"));
  } catch (error) {
    console.warn("최근 훈련 지표를 불러오지 못했습니다.", error);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function formatDate(value) {
  if (!value) return "첫 훈련";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "기록 있음";
  return new Intl.DateTimeFormat("ko-KR", { month: "short", day: "numeric" }).format(date);
}

function selectUser(user) {
  state.currentUser = user;
  renderSidebarUser();
  $("#readyUserName").textContent = user.name;
  $("#readyDistance").textContent = user.recommended_distance_cm ? `${Math.round(user.recommended_distance_cm)} cm` : "측정 필요";
  state.trainingConfig.hand = user.dominant_hand;
  updateTrainingLabels();
  renderDashboard();
  showToast(`${user.name}님의 프로필을 불러왔습니다.`);
  speak(`${user.name}님, 다시 오셨군요. 어떤 훈련을 할까요?`);
  showScreen("dashboard");
}

function valueOrPending(value, suffix = "cm") {
  return value === null || value === undefined || value === "" ? "미측정" : `${Math.round(Number(value))}${suffix}`;
}

function renderDashboard() {
  const user = state.currentUser;
  if (!user) return;
  const hand = user.dominant_hand === "right" ? "오른손잡이" : "왼손잡이";
  const initial = [...user.name][0] || "K";
  $("#dashboardUserName").textContent = user.name;
  $("#dashboardAvatar").textContent = initial;
  $("#dashboardName").textContent = user.name;
  $("#dashboardBasic").textContent = `${Math.round(user.height_cm)}cm · ${hand}`;
  $("#dashboardWingspan").textContent = valueOrPending(user.wingspan_cm);
  $("#dashboardRightReach").textContent = valueOrPending(user.right_punch_reach_cm);
  $("#dashboardLeftReach").textContent = valueOrPending(user.left_punch_reach_cm);
  $("#dashboardDistance").textContent = valueOrPending(user.recommended_distance_cm);
  const dashboardDbText = $("#dashboardDbText");
  if (dashboardDbText) dashboardDbText.textContent = state.database.ok ? "정상" : "확인 필요";
}

function openTrainingSetup() {
  if (!state.currentUser) {
    showToast("먼저 사용자를 선택해 주세요.", "error");
    showScreen("profiles");
    return;
  }
  const form = $("#trainingSetupForm");
  const handInput = form.querySelector(`input[name="hand"][value="${state.trainingConfig.hand}"]`) || form.querySelector('input[name="hand"][value="right"]');
  const typeInput = form.querySelector(`input[name="type"][value="${state.trainingConfig.type}"]`);
  const durationInput = form.querySelector(`input[name="duration"][value="${state.trainingConfig.durationSec}"]`) || form.querySelector('input[name="duration"][value="60"]');
  const difficultyInput = form.querySelector(`input[name="difficulty"][value="${state.trainingConfig.difficulty}"]`);
  if (handInput) handInput.checked = true;
  if (typeInput) typeInput.checked = true;
  if (durationInput) durationInput.checked = true;
  if (difficultyInput) difficultyInput.checked = true;
  showScreen("setup");
}

async function submitTrainingSetup(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  configureTraining({
    hand: String(form.get("hand") || "right"),
    type: String(form.get("type") || "straight"),
    durationSec: Number(form.get("duration") || 60),
  });
  state.trainingConfig.difficulty = String(form.get("difficulty") || "normal");
  await sendRobotCommand("training_start");
  goToTrainingReady();
}

async function loadHistory() {
  if (!state.currentUser) return;
  try {
    state.sessions = await api(`/api/users/${state.currentUser.id}/sessions`);
    renderHistory();
    showScreen("history");
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderHistory() {
  const user = state.currentUser;
  const list = $("#historyList");
  const empty = $("#historyEmpty");
  const summary = $("#historySummary");
  if (!user) return;
  $("#historySubtitle").textContent = `${user.name}님의 최근 훈련 결과입니다.`;
  list.innerHTML = "";
  const sessions = state.sessions || [];
  empty.classList.toggle("hidden", sessions.length > 0);
  list.classList.toggle("hidden", sessions.length === 0);
  summary.classList.toggle("hidden", sessions.length === 0);
  if (!sessions.length) return;

  const totalPunches = sessions.reduce((sum, item) => sum + Number(item.punch_count || 0), 0);
  const averageSuccess = sessions.reduce((sum, item) => sum + Number(item.success_rate || 0), 0) / sessions.length;
  const reactionValues = sessions.map((item) => Number(item.avg_reaction_ms)).filter(Number.isFinite);
  const averageReaction = reactionValues.length ? reactionValues.reduce((a,b) => a+b, 0) / reactionValues.length : null;
  summary.innerHTML = `
    <div><span>저장된 훈련</span><strong>${sessions.length}</strong><small>회</small></div>
    <div><span>누적 펀치</span><strong>${totalPunches}</strong><small>회</small></div>
    <div><span>평균 성공률</span><strong>${averageSuccess.toFixed(1)}</strong><small>%</small></div>
    <div><span>평균 반응</span><strong>${averageReaction ? (averageReaction/1000).toFixed(2) : "—"}</strong><small>초</small></div>`;

  sessions.forEach((session) => {
    const item = document.createElement("article");
    item.className = "history-item";
    const hand = session.hand === "left" ? "왼손" : session.hand === "both" ? "양손" : "오른손";
    const type = trainingTypeLabel(session.training_type);
    const date = new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(session.created_at));
    item.innerHTML = `
      <div class="history-date"><span>${escapeHtml(date)}</span><small>SESSION #${session.id}</small></div>
      <div class="history-main"><strong>${hand} ${type}</strong><span>${session.duration_sec}초 훈련</span></div>
      <div class="history-stat"><span>펀치</span><strong>${session.punch_count}회</strong></div>
      <div class="history-stat"><span>성공률</span><strong>${Number(session.success_rate).toFixed(1)}%</strong></div>
      <div class="history-stat"><span>평균 반응</span><strong>${session.avg_reaction_ms ? `${(Number(session.avg_reaction_ms)/1000).toFixed(2)}초` : "—"}</strong></div>
      <p>${escapeHtml(session.feedback || "저장된 코칭 피드백이 없습니다.")}</p>`;
    list.appendChild(item);
  });
}

function updateTrainingLabels() {
  const handText = state.trainingConfig.hand === "right" ? "오른손" : state.trainingConfig.hand === "left" ? "왼손" : "양손";
  const typeText = trainingTypeLabel(state.trainingConfig.type);
  const label = `${handText} ${typeText}`;
  $("#readyModeChip").textContent = `${label} · ${state.trainingConfig.durationSec}초`;
  $("#countdownMode").textContent = label;
  $("#trainingTitle").textContent = label;
  $("#resultSubtitle").textContent = `${label} · ${state.trainingConfig.durationSec}초`;
}

async function registerUser(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const user = await api("/api/users", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        height_cm: Number(form.get("height_cm")),
        dominant_hand: form.get("dominant_hand"),
      }),
    });
    state.currentUser = user;
  renderSidebarUser();
    state.users.unshift(user);
    renderProfiles();
    await checkDatabaseStatus();
    selectUser(user);
    resetMeasurement();
    showScreen("measure");
    speak(`${user.name}님, 이제 양팔 리치를 측정할게요. 전신이 보이도록 서서 손가락을 펴고 양팔을 수평으로 벌려주세요.`);
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function ensureCamera(videoElement) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("이 브라우저는 카메라 접근을 지원하지 않습니다.");
  if (!state.stream) {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  }
  videoElement.srcObject = state.stream;
  await videoElement.play();
  return state.stream;
}

async function ensurePoseLandmarker() {
  if (state.poseLandmarker) return state.poseLandmarker;
  if (state.poseLoadFailed) throw new Error("자세 인식 모듈을 불러오지 못했습니다. 직접 입력을 이용해 주세요.");
  try {
    const visionModule = await import("/static/vendor/mediapipe/tasks-vision-0.10.14/vision_bundle.mjs");
    const vision = await visionModule.FilesetResolver.forVisionTasks(
      "/static/vendor/mediapipe/tasks-vision-0.10.14/wasm"
    );
    const commonOptions = {
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    };
    const modelAssetPath = "/static/vendor/mediapipe/models/pose_landmarker_lite.task";
    try {
      state.poseLandmarker = await visionModule.PoseLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { modelAssetPath, delegate: "GPU" },
      });
    } catch (gpuError) {
      console.warn("GPU delegate unavailable; falling back to CPU", gpuError);
      state.poseLandmarker = await visionModule.PoseLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { modelAssetPath, delegate: "CPU" },
      });
    }
    return state.poseLandmarker;
  } catch (error) {
    console.error("PoseLandmarker load failed", error);
    state.poseLoadFailed = true;
    throw new Error("로컬 자세 인식 모델을 준비하지 못했습니다. UI 서버를 다시 시작하거나 직접 입력을 이용해 주세요.");
  }
}

function stopPoseLoop() {
  state.poseLoopToken += 1;
}

async function startPoseLoop(video, canvas, onResult) {
  stopPoseLoop();
  const token = state.poseLoopToken;
  const poseLandmarker = await ensurePoseLandmarker();
  const ctx = canvas.getContext("2d");
  let lastVideoTime = -1;
  let lastInferenceAt = 0;

  const render = async () => {
    if (token !== state.poseLoopToken) return;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime && performance.now() - lastInferenceAt > 50) {
      lastVideoTime = video.currentTime;
      lastInferenceAt = performance.now();
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      try {
        const result = poseLandmarker.detectForVideo(video, performance.now());
        state.lastPoseResult = result;
        drawPose(ctx, canvas, result?.landmarks?.[0]);
        onResult?.(result);
      } catch (error) {
        console.warn("Pose inference error", error);
      }
    }
    requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
}

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
  [15,17],[15,19],[16,18],[16,20],
  [23,25],[25,27],[27,29],[27,31],[24,26],[26,28],[28,30],[28,32]
];

function drawPose(ctx, canvas, landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.lineWidth = Math.max(3, canvas.width / 350);
  ctx.strokeStyle = "rgba(39, 119, 199, .88)";
  ctx.fillStyle = "rgba(255, 255, 255, .95)";
  for (const [a, b] of POSE_CONNECTIONS) {
    const p1 = landmarks[a];
    const p2 = landmarks[b];
    if (!p1 || !p2 || Math.min(p1.visibility ?? 1, p2.visibility ?? 1) < .45) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
    ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
    ctx.stroke();
  }
  [11,12,13,14,15,16,17,18,19,20].forEach((index) => {
    const p = landmarks[index];
    if (!p || (p.visibility ?? 1) < .45) return;
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, Math.max(5, canvas.width / 180), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function dist3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function avgPoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 };
}

function angleDeg(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const mag = Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z);
  if (!mag) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
}

function calculateBodyMetrics(result) {
  const n = result?.landmarks?.[0];
  const w = result?.worldLandmarks?.[0];
  if (!n || !w || !state.currentUser) return null;
  const required = [0,11,12,13,14,15,16,29,30];
  const visibility = required.map((i) => n[i]?.visibility ?? 0);
  if (Math.min(...visibility) < .55) return null;

  const rawNoseHeel = dist3(w[0], avgPoint(w[29], w[30]));
  if (rawNoseHeel < .4) return null;
  const userHeightM = state.currentUser.height_cm / 100;
  const scale = userHeightM * .94 / rawNoseHeel;

  const leftArm = dist3(w[11], w[13]) + dist3(w[13], w[15]);
  const rightArm = dist3(w[12], w[14]) + dist3(w[14], w[16]);
  const shoulderWidth = dist3(w[11], w[12]);
  const leftHandCandidates = [17,19].map((i) => ({
    distance: w[i] ? dist3(w[15], w[i]) : 0,
    visibility: n[i]?.visibility ?? 0,
  }));
  const rightHandCandidates = [18,20].map((i) => ({
    distance: w[i] ? dist3(w[16], w[i]) : 0,
    visibility: n[i]?.visibility ?? 0,
  }));
  const leftOpenHand = Math.max(...leftHandCandidates.filter((item) => item.visibility >= .55).map((item) => item.distance), 0);
  const rightOpenHand = Math.max(...rightHandCandidates.filter((item) => item.visibility >= .55).map((item) => item.distance), 0);
  const openHandConfidence = Math.min(
    Math.max(...leftHandCandidates.map((item) => item.visibility)),
    Math.max(...rightHandCandidates.map((item) => item.visibility)),
  );
  const openHandsVisible = leftOpenHand > 0 && rightOpenHand > 0;

  // Wingspan uses the observed open-hand fingertip landmarks. Punch reach uses
  // a separate wrist-to-front-of-fist allowance because YOLO/MediaPipe wrists
  // do not represent the actual contact point of a closed fist.
  const fistFrontAllowanceCm = 9;
  const wingspanCm = (leftArm + shoulderWidth + rightArm + leftOpenHand + rightOpenHand) * scale * 100;
  const leftReachCm = leftArm * scale * 100 + fistFrontAllowanceCm;
  const rightReachCm = rightArm * scale * 100 + fistFrontAllowanceCm;

  const leftAngle = angleDeg(w[11], w[13], w[15]);
  const rightAngle = angleDeg(w[12], w[14], w[16]);
  const armsHorizontal = [11,12,13,14,15,16].every((i) => Math.abs(n[i].y - (n[11].y + n[12].y) / 2) < .105);
  const centerX = (n[11].x + n[12].x) / 2;
  const centered = centerX > .28 && centerX < .72;
  const confidence = Math.min(1, visibility.reduce((a, b) => a + b, 0) / visibility.length);

  return {
    wingspanCm, leftReachCm, rightReachCm,
    leftAngle, rightAngle, armsHorizontal, centered, confidence,
    openHandsVisible, openHandConfidence,
  };
}

function resetMeasurement() {
  state.measurement = { stage: "wingspan", collecting: false, samples: [], values: {} };
  updateMeasurementUI();
}

function updateMeasurementUI() {
  const stage = state.measurement.stage;
  const labels = {
    wingspan: "전신이 보이도록 서서 손가락을 펴고 양팔을 수평으로 벌려주세요.",
    right: "가드 자세에서 오른손을 천천히 끝까지 뻗어주세요.",
    left: "가드 자세에서 왼손을 천천히 끝까지 뻗어주세요.",
  };
  $("#measureInstruction").textContent = labels[stage] || "측정 완료";
  $$(".measure-stage").forEach((el) => {
    const elStage = el.dataset.measureStage;
    el.classList.toggle("active", elStage === stage);
    const order = ["wingspan", "right", "left"];
    el.classList.toggle("done", order.indexOf(elStage) < order.indexOf(stage) || stage === "done");
  });
  $("#measureProgress").textContent = state.measurement.samples.length;
  $("#wingspanValue").textContent = state.measurement.values.wingspan ? `${Math.round(state.measurement.values.wingspan)} cm` : "—";
  $("#rightReachValue").textContent = state.measurement.values.right ? `${Math.round(state.measurement.values.right)} cm` : "—";
  $("#leftReachValue").textContent = state.measurement.values.left ? `${Math.round(state.measurement.values.left)} cm` : "—";
}

async function startMeasurementCamera() {
  if (!state.currentUser) {
    showToast("먼저 사용자를 등록하거나 선택해 주세요.", "error");
    showScreen("profiles");
    return;
  }
  const video = $("#cameraVideo");
  const canvas = $("#poseCanvas");
  const button = $("#cameraStartButton");
  button.disabled = true;
  button.textContent = "카메라 준비 중…";
  try {
    await ensureCamera(video);
    await startPoseLoop(video, canvas, processMeasurementFrame);
    $("#captureMeasureButton").disabled = false;
    button.textContent = "카메라 연결됨";
    $("#cameraMessage").textContent = "자세를 맞춘 뒤 현재 단계 측정을 누르세요";
  } catch (error) {
    button.disabled = false;
    button.textContent = "카메라 다시 시작";
    showToast(error.message, "error");
    $("#cameraMessage").textContent = "자동 측정 불가 · 직접 입력을 이용해 주세요";
  }
}

function processMeasurementFrame(result) {
  const metrics = calculateBodyMetrics(result);
  if (!metrics) {
    $("#cameraMessage").textContent = "전신과 양손이 모두 보이게 이동해 주세요";
    return;
  }
  const stage = state.measurement.stage;
  let valid = metrics.centered;
  if (stage === "wingspan") valid = valid && metrics.openHandsVisible && metrics.armsHorizontal && metrics.leftAngle > 145 && metrics.rightAngle > 145;
  if (stage === "right") valid = valid && metrics.rightAngle > 145;
  if (stage === "left") valid = valid && metrics.leftAngle > 145;

  $("#cameraMessage").textContent = valid
    ? state.measurement.collecting ? "움직이지 말고 자세를 유지하세요" : "좋습니다. 현재 단계 측정을 눌러주세요"
    : stage === "wingspan"
      ? metrics.openHandsVisible ? "양팔을 어깨 높이로 곧게 펴주세요" : "손가락을 펴고 양손 끝이 모두 보이게 해주세요"
      : `${stage === "right" ? "오른팔" : "왼팔"}을 끝까지 펴주세요`;

  if (!state.measurement.collecting || !valid) return;
  const value = stage === "wingspan" ? metrics.wingspanCm : stage === "right" ? metrics.rightReachCm : metrics.leftReachCm;
  const confidence = stage === "wingspan"
    ? Math.min(metrics.confidence, metrics.openHandConfidence)
    : metrics.confidence;
  state.measurement.samples.push({ value, confidence });
  if (state.measurement.samples.length > 30) state.measurement.samples.shift();
  updateMeasurementUI();
  if (state.measurement.samples.length >= 30) finishMeasurementStage();
}

function beginMeasurementStage() {
  if (state.measurement.stage === "done") return;
  state.measurement.collecting = true;
  state.measurement.samples = [];
  $("#captureMeasureButton").disabled = true;
  $("#captureMeasureButton").textContent = "측정 중…";
  $("#cameraMessage").textContent = "자세를 유지하세요";
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function finishMeasurementStage() {
  const stage = state.measurement.stage;
  const samples = state.measurement.samples;
  state.measurement.values[stage] = median(samples.map((sample) => sample.value));
  state.measurement.values[`${stage}Confidence`] = samples.reduce((sum, sample) => sum + sample.confidence, 0) / samples.length;
  state.measurement.collecting = false;
  state.measurement.samples = [];
  const order = ["wingspan", "right", "left"];
  const nextIndex = order.indexOf(stage) + 1;
  state.measurement.stage = order[nextIndex] || "done";
  $("#captureMeasureButton").disabled = false;
  $("#captureMeasureButton").textContent = "현재 단계 측정";
  updateMeasurementUI();

  if (state.measurement.stage === "done") {
    stopPoseLoop();
    prepareMeasurementResult();
  } else {
    const nextText = state.measurement.stage === "right" ? "이제 오른손을 끝까지 뻗어주세요." : "마지막으로 왼손을 끝까지 뻗어주세요.";
    speak(nextText);
  }
}

function applyManualMeasurement(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  state.measurement.values = {
    wingspan: Number(form.get("wingspan")),
    right: Number(form.get("right")),
    left: Number(form.get("left")),
    wingspanConfidence: .65,
    rightConfidence: .65,
    leftConfidence: .65,
  };
  state.measurement.stage = "done";
  closeModal("manualMeasureModal");
  prepareMeasurementResult();
}

function prepareMeasurementResult() {
  const v = state.measurement.values;
  const maxReach = Math.max(v.right, v.left);
  v.recommended = Math.round(maxReach + 35);
  v.confidence = Math.min(v.wingspanConfidence || .6, v.rightConfidence || .6, v.leftConfidence || .6);
  $("#resultWingspan").textContent = `${Math.round(v.wingspan)} cm`;
  $("#resultRightReach").textContent = `${Math.round(v.right)} cm`;
  $("#resultLeftReach").textContent = `${Math.round(v.left)} cm`;
  $("#resultDistance").textContent = `${Math.round(v.recommended)} cm`;
  $("#confidenceText").textContent = `측정 결과 ${Math.round(v.confidence * 100)}% 안정적으로 확인됨`;
  showScreen("measure-result");
  speak("리치 측정이 완료됐습니다. 측정 결과를 확인해 주세요.");
}

async function saveMeasurement() {
  if (!state.currentUser) return;
  const v = state.measurement.values;
  try {
    const user = await api(`/api/users/${state.currentUser.id}/measurement`, {
      method: "PATCH",
      body: JSON.stringify({
        wingspan_cm: v.wingspan,
        left_punch_reach_cm: v.left,
        right_punch_reach_cm: v.right,
        recommended_distance_cm: v.recommended,
        measurement_confidence: v.confidence,
      }),
    });
    state.currentUser = user;
  renderSidebarUser();
    const index = state.users.findIndex((item) => item.id === user.id);
    if (index >= 0) state.users[index] = user;
    renderProfiles();
    selectUser(user);
    showToast("프로필과 리치 측정값을 저장했습니다.");
    showScreen("dashboard");
    if (state.wake.session_active) extendVoiceSession(30);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function openModal(id) { $(`#${id}`).classList.add("open"); }
function closeModal(id) { $(`#${id}`).classList.remove("open"); }

const VISION_FEEDBACK_TEXT = {
  elbow_not_extended: "팔꿈치를 조금 더 충분히 펴세요.",
  elbow_flared: "팔꿈치가 바깥으로 벌어지지 않게 유지하세요.",
  guard_dropped: "반대 손 가드를 얼굴 가까이 올리세요.",
  torso_overlean: "상체가 너무 많이 기울어졌습니다.",
  wrist_height_off: "주먹 높이를 기준 자세에 맞추세요.",
  hook_elbow_angle_off: "훅의 팔꿈치 각도를 조정하세요.",
  hook_elbow_path_off: "훅 팔꿈치 궤적을 더 둥글게 유지하세요.",
  hook_wrist_elbow_misaligned: "훅에서 손목과 팔꿈치 높이를 맞추세요.",
  uppercut_elbow_angle_off: "어퍼컷 팔꿈치 각도를 조정하세요.",
  uppercut_wrist_path_off: "어퍼컷 손목을 위쪽으로 움직이세요.",
  uppercut_height_off: "어퍼컷 주먹 높이를 조정하세요.",
};

function startVisionPolling() {
  clearInterval(state.vision.statusTimer);
  clearInterval(state.vision.previewTimer);
  clearInterval(state.vision.eventTimer);
  state.vision.statusTimer = setInterval(checkVisionStatus, 900);
  state.vision.previewTimer = setInterval(refreshVisionPreview, 100);
  state.vision.eventTimer = setInterval(pollVisionEvents, 220);
  checkVisionStatus();
  refreshVisionPreview();
  pollVisionEvents();
}

function currentVisionPreview() {
  if (state.currentScreen === "ready") return $("#readyVisionPreview");
  if (state.currentScreen === "training") return $("#liveVisionPreview");
  return null;
}

function refreshVisionPreview() {
  if (!state.vision.connected || !state.vision.previewAvailable || state.vision.previewBusy) return;
  const image = currentVisionPreview();
  if (!image || image.classList.contains("hidden")) return;

  state.vision.previewBusy = true;
  const release = () => {
    image.removeEventListener("load", release);
    image.removeEventListener("error", release);
    state.vision.previewBusy = false;
  };
  image.addEventListener("load", release);
  image.addEventListener("error", release);
  image.src = `/api/vision/preview.jpg?t=${Date.now()}`;
}

async function checkVisionStatus() {
  if (state.vision.statusBusy) return state.vision;
  state.vision.statusBusy = true;
  try {
    const status = await api("/api/vision/status");
    state.vision.connected = Boolean(status.connected);
    state.vision.previewAvailable = Boolean(status.preview_available);
    state.vision.evidenceAvailable = Boolean(status.evidence_available);
    state.vision.liveStatus = status.live_status || {};
    const dot = $("#visionStatusDot");
    dot?.classList.toggle("ready", state.vision.connected);
    const readyBadge = $(".vision-source-badge");
    readyBadge?.classList.toggle("connected", state.vision.connected);
    if ($("#readyVisionSource")) {
      $("#readyVisionSource").textContent = state.vision.connected ? "자세 인식 연결됨" : "카메라 연결 대기";
    }
    if (status.preview_version && status.preview_version !== state.vision.previewVersion) {
      state.vision.previewVersion = Number(status.preview_version);
    }
    if (status.evidence_version && status.evidence_version !== state.vision.evidenceVersion) {
      state.vision.evidenceVersion = Number(status.evidence_version);
      const url = `/api/vision/evidence.jpg?v=${state.vision.evidenceVersion}`;
      const live = $("#liveEvidenceImage");
      const result = $("#resultEvidenceImage");
      if (live) { live.src = url; live.classList.remove("hidden"); }
      if (result) result.src = url;
    }
    return status;
  } catch (error) {
    state.vision.connected = false;
    $("#visionStatusDot")?.classList.remove("ready");
    return state.vision;
  } finally {
    state.vision.statusBusy = false;
  }
}

function useVisionPreview(screenName) {
  const isReady = screenName === "ready";
  const image = isReady ? $("#readyVisionPreview") : $("#liveVisionPreview");
  const video = isReady ? $("#trainingVideo") : $("#liveVideo");
  const canvas = isReady ? $("#trainingCanvas") : $("#liveCanvas");
  image?.classList.remove("hidden");
  video?.classList.add("vision-hidden");
  canvas?.classList.add("vision-hidden");
  if (state.vision.previewVersion && image) image.src = `/api/vision/preview.jpg?v=${state.vision.previewVersion}`;
}

function useBrowserCamera(screenName) {
  const isReady = screenName === "ready";
  const image = isReady ? $("#readyVisionPreview") : $("#liveVisionPreview");
  const video = isReady ? $("#trainingVideo") : $("#liveVideo");
  const canvas = isReady ? $("#trainingCanvas") : $("#liveCanvas");
  image?.classList.add("hidden");
  video?.classList.remove("vision-hidden");
  canvas?.classList.remove("vision-hidden");
}

async function pollVisionEvents() {
  if (state.vision.eventBusy) return;
  state.vision.eventBusy = true;
  try {
    const result = await api(`/api/vision/events?after=${state.vision.lastEventId}`);
    for (const event of result.events || []) {
      state.vision.lastEventId = Math.max(state.vision.lastEventId, Number(event.id) || 0);
      if (event.type === "punch") handleVisionPunch(event.payload || {});
    }
  } catch (error) {
    // Vision node may simply be offline during UI-only development.
  } finally {
    state.vision.eventBusy = false;
  }
}

function handleVisionPunch(payload) {
  if (!state.training.running || state.training.paused || !state.training.useVision) return;
  const configuredHand = state.trainingConfig.hand;
  if (configuredHand !== "both" && payload.punch_side && payload.punch_side !== configuredHand) {
    $("#visionFeedbackTitle").textContent = "반대 손 펀치가 감지됐습니다";
    $("#visionFeedbackText").textContent = "현재 선택한 훈련 손으로 다시 시도하세요.";
    return;
  }

  const t = state.training;
  const score = Number(payload.total_score || 0);
  const passed = Boolean(payload.passed);
  t.lastPunchAt = performance.now();
  t.punches += 1;
  if (passed) t.successful += 1;
  t.scores.push(score);
  t.punchEvents.push(payload);

  for (const violation of payload.violations || []) {
    const code = String(violation.code || "unknown");
    t.violationCounts[code] = (t.violationCounts[code] || 0) + 1;
  }

  if (t.promptStartedAt) {
    const reaction = performance.now() - t.promptStartedAt;
    t.reactions.push(reaction);
    t.promptStartedAt = null;
  }

  const firstViolation = (payload.violations || [])[0];
  const feedback = firstViolation
    ? (VISION_FEEDBACK_TEXT[firstViolation.code] || "자세를 조금 더 안정적으로 유지하세요.")
    : "좋은 자세입니다. 펀치 후 가드로 빠르게 복귀하세요.";
  t.lastFeedback = feedback;

  const side = payload.punch_side === "left" ? "왼손" : "오른손";
  const typeMap = { straight: "스트레이트", hook: "훅", uppercut: "어퍼컷" };
  const type = typeMap[payload.punch_type] || String(payload.punch_type || "펀치");
  $("#liveVisionScore").textContent = score.toFixed(0);
  $("#visionPunchType").textContent = `${side} ${type}`;
  $("#visionFeedbackTitle").textContent = passed ? "GOOD · 자세 기준 통과" : "자세를 조정해 보세요";
  $("#visionFeedbackText").textContent = feedback;
  $("#liveStatus").textContent = `${type} 감지 · ${score.toFixed(1)}점`;
  $("#liveCommand").textContent = passed ? "좋아요! 가드로 복귀" : feedback;
  pulseTarget(passed);
  updateMetrics();
}

function dominantViolationText() {
  const entries = Object.entries(state.training.violationCounts || {});
  if (!entries.length) return state.training.lastFeedback || "안정적인 자세를 유지했습니다.";
  entries.sort((a, b) => b[1] - a[1]);
  return VISION_FEEDBACK_TEXT[entries[0][0]] || state.training.lastFeedback || "자세를 조금 더 안정적으로 유지하세요.";
}

function configureTraining({ hand, type, durationSec } = {}) {
  if (hand) state.trainingConfig.hand = hand;
  if (type) state.trainingConfig.type = type;
  if (durationSec) state.trainingConfig.durationSec = durationSec;
  updateTrainingLabels();
}

function goToTrainingReady() {
  if (!state.currentUser) {
    showToast("먼저 사용자 프로필을 선택해 주세요.");
    showScreen("profiles");
    return;
  }
  $("#readyUserName").textContent = state.currentUser.name;
  $("#readyDistance").textContent = state.currentUser.recommended_distance_cm ? `${Math.round(state.currentUser.recommended_distance_cm)} cm` : "측정 필요";
  updateTrainingLabels();
  showScreen("ready");
  speak(`${state.currentUser.name}님, 화면의 훈련 영역에 맞춰 서주세요.`);
}

async function startAlignment() {
  const button = $("#prepareTrainingButton");
  button.disabled = true;
  button.textContent = "비전 확인 중…";
  await checkVisionStatus();

  if (state.vision.connected && state.vision.previewAvailable) {
    useVisionPreview("ready");
    $("#checkVision").classList.add("ok");
    let stableFrames = 0;
    for (let index = 0; index < 80; index += 1) {
      if (state.currentScreen !== "ready") return;
      const live = state.vision.liveStatus || {};
      const poseOk = Boolean(live.pose_detected);
      const centered = Boolean(live.centered);
      $("#checkPose").classList.toggle("ok", poseOk);
      $("#checkDistance").classList.toggle("ok", centered);
      $("#alignmentFeedback").textContent = !poseOk
        ? "상체와 양손이 카메라에 보이게 이동하세요"
        : !centered
          ? "화면 중앙으로 이동하세요"
          : live.detector_state === "READY"
            ? "가드 자세 확인 완료"
            : "가드 자세를 잠시 유지하세요";
      stableFrames = poseOk && centered ? stableFrames + 1 : Math.max(0, stableFrames - 2);
      if (stableFrames >= 10) {
        button.textContent = "정렬 완료";
        speak("위치가 확인됐습니다. 3초 후 시작합니다.");
        await runCountdown();
        return;
      }
      await sleep(150);
    }
    button.disabled = false;
    button.textContent = "다시 확인";
    showToast("위치 확인 시간이 초과됐습니다. 자세를 맞춘 뒤 다시 시도하세요.");
    return;
  }

  useBrowserCamera("ready");
  $("#checkVision").classList.remove("ok");
  $("#readyVisionSource").textContent = "카메라 직접 연결";
  const video = $("#trainingVideo");
  const canvas = $("#trainingCanvas");
  button.textContent = "정렬 확인 중…";
  try {
    await ensureCamera(video);
    let stableFrames = 0;
    await startPoseLoop(video, canvas, (result) => {
      const metrics = calculateBodyMetrics(result);
      const poseOk = Boolean(metrics && metrics.leftAngle > 70 && metrics.rightAngle > 70);
      const centered = Boolean(metrics?.centered);
      $("#checkPose").classList.toggle("ok", poseOk);
      $("#checkDistance").classList.toggle("ok", centered);
      $("#alignmentFeedback").textContent = !poseOk ? "상체와 양손이 보이게 이동하세요" : !centered ? "화면 중앙으로 이동하세요" : "위치가 확인됐습니다";
      stableFrames = poseOk && centered ? stableFrames + 1 : Math.max(0, stableFrames - 2);
      if (stableFrames >= 22) {
        stopPoseLoop();
        button.textContent = "정렬 완료";
        speak("위치가 확인됐습니다. 3초 후 시작합니다.");
        runCountdown();
      }
    });
  } catch (error) {
    button.disabled = false;
    button.textContent = "다시 시도";
    showToast(error.message, "error");
  }
}

async function runCountdown() {
  showScreen("countdown");
  for (const number of [3, 2, 1]) {
    $("#countdownNumber").textContent = number;
    speak(String(number));
    await sleep(900);
  }
  $("#countdownNumber").textContent = "GO";
  speak("시작");
  await sleep(500);
  startTraining();
}

function resetTrainingState() {
  clearInterval(state.training.timerId);
  clearTimeout(state.training.promptId);
  state.training = {
    running: false,
    paused: false,
    remainingSec: state.trainingConfig.durationSec,
    punches: 0,
    successful: 0,
    prompts: 0,
    reactions: [],
    promptStartedAt: null,
    timerId: null,
    promptId: null,
    armReady: true,
    lastPunchAt: 0,
    useVision: false,
    scores: [],
    punchEvents: [],
    violationCounts: {},
    lastFeedback: "",
  };
}

async function startTraining() {
  resetTrainingState();
  state.training.running = true;
  await ensureVoiceSession(Math.max(180, state.trainingConfig.durationSec + 120));
  showScreen("training");
  await checkVisionStatus();
  state.training.useVision = Boolean(state.vision.connected && state.vision.previewAvailable);
  updateMetrics();

  if (state.training.useVision) {
    useVisionPreview("training");
    $("#liveStatus").textContent = "실시간 자세 코칭 연결됨";
    $("#visionFeedbackTitle").textContent = "가드 자세를 잡아주세요";
    $("#visionFeedbackText").textContent = "펀치가 확정될 때마다 종류·자세 점수·오류가 표시됩니다.";
  } else {
    useBrowserCamera("training");
    const video = $("#liveVideo");
    const canvas = $("#liveCanvas");
    try {
      await ensureCamera(video);
      await startPoseLoop(video, canvas, processTrainingFrame);
    } catch (error) {
      $("#liveStatus").textContent = "카메라 없음 · Space 키 데모";
      showToast("실시간 자세 분석이 연결되지 않아 기본 카메라 모드로 실행합니다.");
    }
  }

  // STT 또는 수동 훈련 확정 시 이미 위빙 정지/준비 자세 복귀 명령을 보냈다.
  schedulePrompt(800);
  state.training.timerId = setInterval(() => {
    if (!state.training.running || state.training.paused) return;
    state.training.remainingSec -= 1;
    updateMetrics();
    if (state.training.remainingSec <= 0) finishTraining();
  }, 1000);
}

function processTrainingFrame(result) {
  if (!state.training.running || state.training.paused) return;
  const w = result?.worldLandmarks?.[0];
  const n = result?.landmarks?.[0];
  if (!w || !n) {
    $("#liveStatus").textContent = "사용자를 찾는 중";
    return;
  }
  const isRight = state.trainingConfig.hand !== "left";
  const s = isRight ? 12 : 11;
  const e = isRight ? 14 : 13;
  const wrist = isRight ? 16 : 15;
  const visibility = Math.min(n[s]?.visibility ?? 0, n[e]?.visibility ?? 0, n[wrist]?.visibility ?? 0);
  if (visibility < .5) {
    $("#liveStatus").textContent = "훈련 손을 화면에 보여주세요";
    return;
  }
  const angle = angleDeg(w[s], w[e], w[wrist]);
  $("#liveStatus").textContent = angle > 150 ? "팔 뻗기 감지" : "가드 자세";
  if (angle < 125) state.training.armReady = true;
  if (angle > 157 && state.training.armReady && performance.now() - state.training.lastPunchAt > 420) {
    registerPunch();
    state.training.armReady = false;
  }
}

function schedulePrompt(delay = 1800) {
  clearTimeout(state.training.promptId);
  if (!state.training.running) return;
  state.training.promptId = setTimeout(() => {
    if (state.training.paused) return schedulePrompt(500);
    state.training.prompts += 1;
    state.training.promptStartedAt = performance.now();
    const handText = state.trainingConfig.hand === "left" ? "왼손" : "오른손";
    const typeText = trainingTypeLabel(state.trainingConfig.type);
    $("#liveCommand").textContent = `${handText} ${typeText}!`;
    if (state.training.prompts % 3 === 1) speak(`${handText} ${typeText}`);
    const nextDelay = state.trainingConfig.difficulty === "high" ? 1300 : 1800 + Math.random() * 900;
    schedulePrompt(nextDelay);
  }, delay);
}

function registerPunch() {
  if (!state.training.running || state.training.paused) return;
  state.training.lastPunchAt = performance.now();
  state.training.punches += 1;
  let success = false;
  if (state.training.promptStartedAt) {
    const reaction = performance.now() - state.training.promptStartedAt;
    if (reaction <= 1500) {
      success = true;
      state.training.successful += 1;
      state.training.reactions.push(reaction);
      $("#liveCommand").textContent = reaction < 550 ? "좋아요! 빠릅니다" : "좋습니다!";
    } else {
      $("#liveCommand").textContent = "조금 더 빠르게!";
    }
    state.training.promptStartedAt = null;
  } else {
    $("#liveCommand").textContent = "가드로 복귀하세요";
  }
  pulseTarget(success);
  updateMetrics();
}

function pulseTarget(success) {
  const target = $(".target-zone");
  target.animate([
    { transform: "scale(1)", opacity: 1 },
    { transform: "scale(1.25)", opacity: .75 },
    { transform: "scale(1)", opacity: 1 },
  ], { duration: 260, easing: "ease-out" });
}

function updateMetrics() {
  const t = state.training;
  const minutes = String(Math.floor(Math.max(0, t.remainingSec) / 60)).padStart(2, "0");
  const seconds = String(Math.max(0, t.remainingSec) % 60).padStart(2, "0");
  const successBase = t.useVision ? t.punches : t.prompts;
  const successRate = successBase ? Math.min(100, Math.round(t.successful / successBase * 100)) : 0;
  const avgReaction = t.reactions.length ? t.reactions.reduce((a, b) => a + b, 0) / t.reactions.length : null;
  $("#trainingTimer").textContent = `${minutes}:${seconds}`;
  $("#metricPunches").textContent = t.punches;
  $("#metricSuccess").textContent = successRate;
  $("#metricReaction").textContent = avgReaction ? (avgReaction / 1000).toFixed(2) : "—";
}

async function togglePause() {
  if (!state.training.running) return;
  state.training.paused = !state.training.paused;
  $("#pauseButton").textContent = state.training.paused ? "다시 시작" : "일시정지";
  $("#liveCommand").textContent = state.training.paused ? "일시정지" : "훈련 재개";
  await sendRobotCommand(state.training.paused ? "pause" : "resume");
  speak(state.training.paused ? "훈련을 잠시 멈춥니다." : "훈련을 다시 시작합니다.");
}

async function finishTraining() {
  if (!state.training.running) return;
  state.training.running = false;
  clearInterval(state.training.timerId);
  clearTimeout(state.training.promptId);
  stopPoseLoop();
  await sendRobotCommand("stop");

  const t = state.training;
  const elapsed = state.trainingConfig.durationSec - Math.max(0, t.remainingSec);
  const successBase = t.useVision ? t.punches : t.prompts;
  const successRate = successBase ? Math.min(100, Math.round(t.successful / successBase * 100)) : 0;
  const avgReaction = t.reactions.length ? t.reactions.reduce((a, b) => a + b, 0) / t.reactions.length : null;
  const score = t.scores.length
    ? Math.round(t.scores.reduce((a, b) => a + b, 0) / t.scores.length)
    : Math.round(Math.min(100, successRate * .65 + Math.min(35, t.punches * 1.5)));
  let feedback = t.useVision ? dominantViolationText() : "리듬을 유지하면서 펀치 후 가드로 빠르게 복귀해 보세요.";
  if (!t.useVision) {
    if (successRate >= 85 && avgReaction && avgReaction < 650) feedback = "반응속도와 타격 타이밍이 매우 좋았습니다. 다음 훈련에서는 콤비네이션 난이도를 높여도 좋습니다.";
    else if (successRate >= 70) feedback = "정확한 타이밍이 안정적입니다. 펀치 후 손을 턱 옆 가드 위치로 더 빠르게 복귀해 보세요.";
    else if (t.punches < 5) feedback = "카메라에 훈련 손이 잘 보이도록 자세를 조정하고, 음성 지시에 맞춰 천천히 다시 시작해 보세요.";
  }

  $("#resultScore").textContent = score;
  $("#resultPunches").textContent = `${t.punches}회`;
  $("#resultSuccess").textContent = `${successRate}%`;
  $("#resultReaction").textContent = avgReaction ? `${(avgReaction / 1000).toFixed(2)}초` : "측정 없음";
  $("#resultFeedback").textContent = feedback;
  const evidenceCard = $("#resultEvidenceCard");
  evidenceCard?.classList.toggle("hidden", !state.vision.evidenceAvailable || !t.useVision);
  if (state.vision.evidenceAvailable) $("#resultEvidenceImage").src = `/api/vision/evidence.jpg?v=${state.vision.evidenceVersion}`;
  $("#resultEvidenceText").textContent = feedback;
  showScreen("result");
  if (state.wake.session_active) extendVoiceSession(30);
  speak(`훈련이 끝났습니다. 총 ${t.punches}회 펀치를 기록했습니다.`);

  if (state.currentUser) {
    try {
      const savedSession = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          user_id: state.currentUser.id,
          training_type: state.trainingConfig.type,
          hand: state.trainingConfig.hand,
          duration_sec: Math.max(1, elapsed),
          punch_count: t.punches,
          success_rate: successRate,
          avg_reaction_ms: avgReaction,
          posture_score: score,
          feedback,
        }),
      });

      if (t.useVision && savedSession?.id) {
        await api("/api/vision/results", {
          method: "POST",
          body: JSON.stringify({
            session_id: savedSession.id,
            total_punches: t.punches,
            successful_punches: t.successful,
            accuracy_percent: successRate,
            average_reaction_sec: avgReaction ? avgReaction / 1000 : null,
            guard_drop_count: Number(t.violationCounts.guard_dropped || 0),
            slow_guard_return_count: null,
            arm_extension_score: null,
            guard_score: null,
            torso_balance_score: score,
            representative_images: state.vision.evidenceAvailable ? ["/api/vision/evidence.jpg"] : [],
            punch_events: t.punchEvents,
            violation_counts: t.violationCounts,
          }),
        });
      }
      await loadUsers();
      await checkDatabaseStatus();
    } catch (error) {
      showToast("훈련 기록 저장에 실패했습니다.", "error");
    }
  }
}

async function sendRobotCommand(command) {
  try {
    return await api("/api/robot/command", { method: "POST", body: JSON.stringify({ command }) });
  } catch (error) {
    console.warn("Robot command failed", command, error);
    return null;
  }
}

async function emergencyStop() {
  state.training.running = false;
  clearInterval(state.training.timerId);
  clearTimeout(state.training.promptId);
  stopPoseLoop();
  await sendRobotCommand("emergency_stop");
  showToast("비상정지 명령을 보냈습니다.", "error");
  speak("비상정지했습니다. 안전을 확인해 주세요.");
  showScreen("home");
}

async function checkDatabaseStatus() {
  try {
    const status = await api("/api/database/status");
    state.database = status;
    $("#databaseStatusDot")?.classList.toggle("ready", Boolean(status.ok));
    renderDashboard();
  } catch (error) {
    state.database.ok = false;
    $("#databaseStatusDot")?.classList.remove("ready");
  }
}

async function checkSttStatus() {
  try {
    const status = await api("/api/stt/status");
    state.sttConfigured = Boolean(status.configured);
  } catch (error) {
    state.sttConfigured = false;
  }
}

async function checkWakeWordStatus() {
  try {
    const status = await api("/api/wakeword/status");
    Object.assign(state.wake, status);
    updateWakeUi();
  } catch (error) {
    Object.assign(state.wake, {
      available: false,
      running: false,
      state: "error",
      message: "웨이크업 서버 상태를 확인하지 못했습니다",
    });
    updateWakeUi();
  }
}

function updateWakeUi(overrideText = "") {
  const wake = state.wake;
  const dot = $("#micStatusDot");
  const topDot = $("#topWakeDot");
  const openAiDot = $("#openAiStatusDot");
  const topLabel = $("#topWakeLabel");
  const stateText = $("#wakeStateText");
  const hint = $("#voiceHintText");
  const toggle = $("#wakeToggleButton");
  if (!dot || !stateText || !hint || !toggle) return;

  const wakeReady = Boolean(wake.available && wake.running && wake.enabled);
  const activelyListening = ["wake_detected", "command_listening", "transcribing"].includes(wake.state);
  const inVoiceSession = Boolean(wake.session_active);
  dot.classList.toggle("ready", wakeReady);
  dot.classList.toggle("listening", activelyListening);
  topDot?.classList.toggle("ready", wakeReady);
  topDot?.classList.toggle("listening", activelyListening);
  openAiDot?.classList.toggle("ready", state.sttConfigured);
  toggle.textContent = wake.enabled ? "음소거" : "마이크 켜기";
  toggle.setAttribute("aria-pressed", String(!wake.enabled));

  let label = overrideText || wake.message || "음성 상태 확인 중";
  if (!state.sttConfigured) label = "음성 기능 확인 필요";
  else if (!wake.enabled) label = "음성 대기 꺼짐";
  else if (inVoiceSession && wake.state === "session_waiting") label = "명령 대기 중";
  stateText.textContent = label;

  if (topLabel) {
    if (!state.sttConfigured) topLabel.textContent = "음성 기능 확인";
    else if (!wake.enabled) topLabel.textContent = "음성 대기 꺼짐";
    else if (wake.state === "wake_detected" || wake.state === "command_listening") topLabel.textContent = "명령 청취 중";
    else if (wake.state === "transcribing") topLabel.textContent = "명령 분석 중";
    else if (inVoiceSession) topLabel.textContent = "명령 대기 중";
    else if (wake.state === "error") topLabel.textContent = "연결 확인 필요";
    else topLabel.textContent = "호출어 대기";
  }

  if (!state.sttConfigured) hint.textContent = "음성 기능을 사용할 수 없습니다";
  else if (wake.state === "error") hint.textContent = wake.last_error || wake.message;
  else if (!wake.enabled) hint.textContent = "마이크 켜기를 누르면 호출어 감지를 재개합니다";
  else if (wake.state === "wake_detected" || wake.state === "command_listening") hint.textContent = "지금 명령을 말해 주세요";
  else if (wake.state === "transcribing") hint.textContent = "음성 명령을 확인하고 있습니다";
  else if (inVoiceSession) hint.textContent = "명령을 하나 말씀해 주세요 · 처리 후 호출 대기로 돌아갑니다";
  else hint.textContent = `대기 중 · “${wake.display_name || "웨이크 업 케이오"}”라고 부른 뒤 명령을 말해 주세요`;
}

function startWakeEventPolling() {
  clearInterval(state.wake.pollTimer);
  state.wake.pollTimer = setInterval(pollWakeEvents, 450);
  pollWakeEvents();
}

async function pollWakeEvents() {
  if (state.wake.pollBusy) return;
  state.wake.pollBusy = true;
  try {
    const result = await api(`/api/wakeword/events?after=${state.wake.lastEventId}`);
    for (const event of result.events || []) {
      state.wake.lastEventId = Math.max(state.wake.lastEventId, Number(event.id) || 0);
      handleWakeEvent(event);
    }
  } catch (error) {
    console.warn("Wake event polling failed", error);
  } finally {
    state.wake.pollBusy = false;
  }
}

function handleWakeEvent(event) {
  const payload = event.payload || {};
  if (event.type === "ready" || event.type === "status") {
    Object.assign(state.wake, payload);
    updateWakeUi();
    return;
  }
  if (event.type === "session_started" || event.type === "session_extended") {
    Object.assign(state.wake, payload, { session_active: true });
    state.wake.state = payload.state || "session_waiting";
    state.wake.message = payload.message || "명령 대기 중 · 명령을 하나 말씀하세요";
    updateWakeUi();
    return;
  }
  if (event.type === "session_ended") {
    Object.assign(state.wake, payload, { session_active: false, session_remaining_sec: 0 });
    state.wake.state = "waiting_wakeword";
    state.wake.message = `‘${state.wake.display_name || "웨이크 업 케이오"}’ 호출어 대기 중`;
    updateWakeUi();
    return;
  }
  if (event.type === "wake_detected") {
    Object.assign(state.wake, payload, { session_active: true });
    state.wake.state = "wake_detected";
    state.wake.message = "호출어 감지 · 명령을 듣는 중";
    $("#voicePanel").classList.add("open");
    $("#recognizedText").textContent = "KO가 듣고 있습니다…";
    updateWakeUi();
    return;
  }
  if (event.type === "listening") {
    Object.assign(state.wake, payload);
    state.wake.state = "command_listening";
    state.wake.message = "명령을 듣는 중";
    updateWakeUi();
    return;
  }
  if (event.type === "transcribing") {
    Object.assign(state.wake, payload);
    state.wake.state = "transcribing";
    state.wake.message = "음성 명령을 확인하는 중";
    $("#recognizedText").textContent = "음성을 확인하고 있습니다…";
    updateWakeUi();
    return;
  }
  if (event.type === "transcript") {
    const sessionActive = payload.session_active !== false && Boolean(state.wake.session_active);
    state.wake.session_active = sessionActive;
    state.wake.state = sessionActive ? "session_waiting" : "waiting_wakeword";
    state.wake.message = sessionActive
      ? "명령 처리 중 · 완료 후 호출 대기로 돌아갑니다"
      : `‘${state.wake.display_name || "웨이크 업 케이오"}’ 호출어 대기 중`;
    const transcript = String(payload.text || "").trim();
    $("#voicePanel").classList.add("open");
    $("#recognizedText").textContent = transcript ? `“${transcript}”` : "음성을 인식하지 못했습니다.";
    const extracted = extractWakeCommand(transcript);
    const command = extracted === null ? transcript : extracted;
    if (command) handleVoiceCommand(command);
    else {
      speak("네. 계속 말씀해 주세요.");
      closeVoicePanelSoon();
    }
    updateWakeUi();
    return;
  }
  if (event.type === "command_error") {
    state.wake.state = state.wake.session_active ? "session_waiting" : "waiting_wakeword";
    showToast(payload.message || "음성을 확인하지 못했습니다. 다시 말씀해 주세요.", "error");
    updateWakeUi();
    return;
  }
  if (event.type === "error") {
    state.wake.state = "error";
    state.wake.last_error = payload.message || "음성 처리 오류";
    state.wake.message = state.wake.last_error;
    showToast(state.wake.last_error, "error");
    updateWakeUi();
  }
}

function extractWakeCommand(rawText) {
  const normalized = String(rawText || "")
    .toLowerCase()
    .replace(/[.,!?~·:;\-]/g, "")
    .replace(/\s+/g, "");
  const variants = ["웨이크업케이오", "웨이크업코", "웨이크업ko", "wakeupko", "헤이케이오", "heyko", "케이오야", "케이오", "케이요", "ko", "록키야", "로키야", "록키", "로키", "rocky"];
  let best = null;
  for (const variant of variants) {
    const index = normalized.indexOf(variant);
    if (index >= 0 && (!best || index < best.index || (index === best.index && variant.length > best.variant.length))) {
      best = { index, variant };
    }
  }
  if (!best) return null;
  return normalized.slice(best.index + best.variant.length);
}

async function toggleWakeMute() {
  try {
    const result = await api("/api/wakeword/control", {
      method: "POST",
      body: JSON.stringify({ enabled: !state.wake.enabled }),
    });
    Object.assign(state.wake, result);
    updateWakeUi();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function parseDuration(text) {
  const secMatch = text.match(/(\d+)\s*초/);
  if (secMatch) return Math.max(10, Math.min(600, Number(secMatch[1])));
  const minMatch = text.match(/(\d+)\s*분/);
  if (minMatch) return Math.max(10, Math.min(600, Number(minMatch[1]) * 60));
  return null;
}

function normalizeVoiceText(rawText) {
  return String(rawText || "").toLowerCase().replace(/[.,!?~·:;\-]/g, "").replace(/\s+/g, "");
}

function findUserByVoice(text) {
  return state.users.find((user) => text.includes(String(user.name).replace(/\s+/g, ""))) || null;
}

function parseCentimeters(text) {
  const match = text.match(/(\d+(?:\.\d+)?)\s*(?:센티|cm|센티미터)/i);
  return match ? Number(match[1]) : null;
}

async function voiceStartReachMeasurement() {
  if (!state.currentUser) {
    speak("먼저 사용자를 선택하거나 등록해 주세요.");
    showScreen(state.users.length ? "profiles" : "register");
    return;
  }
  resetMeasurement();
  showScreen("measure");
  speak("리치 측정을 시작합니다. 카메라 중앙에 서서 손가락을 펴고 양팔을 벌려 주세요.");
  await sleep(350);
  try {
    await startMeasurementCamera();
  } catch (error) {
    speak("처음 사용하는 카메라는 화면에서 권한 허용을 한 번 눌러 주세요.");
  }
}

function handleVoiceCommand(rawText) {
  const text = normalizeVoiceText(rawText);
  const namedUser = findUserByVoice(text);

  if (text.includes("대화종료") || text.includes("음성종료") || text.includes("음성대기모드") || text.includes("그만들을게") || text.includes("호출어대기로")) {
    endVoiceSession("voice_command");
    speak("음성 대기 모드로 돌아갑니다.");
    closeVoicePanelSoon();
    return;
  }

  if (text.includes("홈으로") || text === "홈" || text.includes("메인화면")) {
    showScreen("home"); speak("홈으로 이동합니다."); closeVoicePanelSoon(); return;
  }
  if (text.includes("뒤로가") || text === "뒤로") {
    const previous = ({ register: "profiles", measure: "dashboard", "measure-result": "measure", setup: "dashboard", history: "dashboard", ready: "setup", settings: "home" })[state.currentScreen] || "home";
    showScreen(previous); speak("이전 화면으로 이동합니다."); closeVoicePanelSoon(); return;
  }
  if (text.includes("시스템설정") || text.includes("연결상태")) {
    showScreen("settings"); speak("연결 상태를 보여드릴게요."); closeVoicePanelSoon(); return;
  }
  if (text.includes("기록보여") || text.includes("훈련기록") || text.includes("최근기록") || text.includes("분석리포트")) {
    if (state.currentUser) loadHistory(); else showScreen("profiles");
    speak(state.currentUser ? "최근 훈련 기록을 보여드릴게요." : "먼저 사용자를 선택해 주세요.");
    closeVoicePanelSoon(); return;
  }
  if (text.includes("사용자등록") || text.includes("프로필등록") || text.includes("새사용자")) {
    speak("새 사용자를 등록할게요. 이름과 키를 말하거나 입력해 주세요.");
    showScreen("register"); closeVoicePanelSoon(); return;
  }
  if (text.includes("사용자불러오기") || text.includes("프로필불러오기") || text.includes("기존사용자") || text.includes("사용자목록")) {
    speak("등록된 사용자 목록을 보여드릴게요."); showScreen("profiles"); closeVoicePanelSoon(); return;
  }
  if (namedUser && (text.includes("불러") || text.includes("선택") || text.includes("로그인") || text.includes("계속"))) {
    selectUser(namedUser); closeVoicePanelSoon(); return;
  }

  if (state.currentScreen === "register") {
    const nameMatch = rawText.match(/(?:이름(?:은|이)?|나는)\s*([가-힣A-Za-z0-9]{1,20})/);
    const cm = parseCentimeters(rawText);
    if (nameMatch) { $("#userName").value = nameMatch[1]; speak(`이름을 ${nameMatch[1]}으로 입력했습니다.`); closeVoicePanelSoon(); return; }
    if ((text.includes("키") || text.includes("신장")) && cm) { $("#userHeight").value = cm; speak(`키를 ${cm}센티미터로 입력했습니다.`); closeVoicePanelSoon(); return; }
    if (text.includes("오른손잡이") || text === "오른손") { $('#registerForm input[name="dominant_hand"][value="right"]').checked = true; speak("오른손잡이로 설정했습니다."); closeVoicePanelSoon(); return; }
    if (text.includes("왼손잡이") || text === "왼손") { $('#registerForm input[name="dominant_hand"][value="left"]').checked = true; speak("왼손잡이로 설정했습니다."); closeVoicePanelSoon(); return; }
    if (text.includes("등록완료") || text.includes("저장하고측정") || text === "저장해") { $("#registerForm").requestSubmit(); closeVoicePanelSoon(); return; }
  }

  if (text.includes("리치측정시작") || text.includes("카메라시작") || text.includes("리치재측정")) {
    voiceStartReachMeasurement(); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "measure") {
    if (text.includes("측정시작") || text.includes("현재단계측정") || text === "측정해") { beginMeasurementStage(); speak("자세를 유지해 주세요."); closeVoicePanelSoon(); return; }
    if (text.includes("직접입력")) { openModal("manualMeasureModal"); speak("리치 값을 직접 입력해 주세요."); closeVoicePanelSoon(); return; }
  }
  if (state.currentScreen === "measure-result") {
    if (text.includes("다시측정") || text.includes("다시해")) { resetMeasurement(); showScreen("measure"); voiceStartReachMeasurement(); closeVoicePanelSoon(); return; }
    if (text.includes("저장") || text.includes("완료")) { saveMeasurement(); closeVoicePanelSoon(); return; }
  }

  if (text.includes("비상정지") || text.includes("긴급정지")) { emergencyStop(); closeVoicePanelSoon(); return; }
  if (text.includes("훈련종료") || text.includes("운동종료") || text === "그만") {
    if (state.training.running) finishTraining(); else showScreen("home"); closeVoicePanelSoon(); return;
  }
  if (text.includes("잠깐멈춰") || text.includes("일시정지") || text === "멈춰") {
    if (state.training.running && !state.training.paused) togglePause(); closeVoicePanelSoon(); return;
  }
  if (text.includes("다시시작") || text.includes("계속해")) {
    if (state.training.running && state.training.paused) togglePause(); closeVoicePanelSoon(); return;
  }
  if (text.includes("현재기록")) { speak(`현재 펀치 ${state.training.punches}회입니다.`); closeVoicePanelSoon(); return; }
  if (text.includes("초더") || text.includes("시간추가")) {
    const extra = parseDuration(rawText) || 30; state.training.remainingSec += extra; updateMetrics(); speak(`${extra}초를 추가했습니다.`); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "ready" && (text.includes("정렬시작") || text.includes("카메라켜") || text.includes("준비시작"))) {
    startAlignment(); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "ready" && (text.includes("바로시작") || text.includes("건너뛰기"))) {
    runCountdown(); closeVoicePanelSoon(); return;
  }

  const wantsTraining = text.includes("운동시작") || text.includes("훈련시작") || text.includes("연습할게") || text.includes("스트레이트") || text.includes("잽") || text.includes("훅") || text.includes("어퍼");
  if (wantsTraining) {
    if (!state.currentUser) { speak("먼저 사용자를 선택해 주세요."); showScreen("profiles"); closeVoicePanelSoon(); return; }
    const hand = text.includes("왼손") ? "left" : text.includes("양손") ? "both" : text.includes("오른손") ? "right" : state.currentUser.dominant_hand || "right";
    const type = text.includes("어퍼") ? "uppercut" : text.includes("훅") ? "hook" : text.includes("잽") ? "jab" : "straight";
    const durationSec = parseDuration(rawText) || 60;
    configureTraining({ hand, type, durationSec });
    const handText = hand === "right" ? "오른손" : hand === "left" ? "왼손" : "양손";
    speak(`${handText} ${trainingTypeLabel(type)} ${durationSec}초 훈련을 준비합니다.`);
    closeVoicePanelSoon(); goToTrainingReady(); return;
  }

  speak("명령을 이해하지 못했어요. 사용자 등록, 리치 측정 시작, 기록 보여줘, 또는 오른손 스트레이트 1분 훈련처럼 말해 주세요.");
}
function closeVoicePanelSoon() {
  setTimeout(() => $("#voicePanel").classList.remove("open"), 650);
}

function wireEvents() {
  $$('[data-screen]').forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screen)));
  $("#registerForm").addEventListener("submit", registerUser);
  $("#trainingSetupForm").addEventListener("submit", submitTrainingSetup);
  $("#openTrainingSetupButton").addEventListener("click", openTrainingSetup);
  $("#openHistoryButton").addEventListener("click", loadHistory);
  $("#historyStartButton").addEventListener("click", openTrainingSetup);
  $("#remeasureButton").addEventListener("click", () => { resetMeasurement(); showScreen("measure"); });
  $("#cameraStartButton").addEventListener("click", startMeasurementCamera);
  $("#captureMeasureButton").addEventListener("click", beginMeasurementStage);
  $("#manualMeasureButton").addEventListener("click", () => openModal("manualMeasureModal"));
  $("#manualMeasureForm").addEventListener("submit", applyManualMeasurement);
  $("#finishProfileButton").addEventListener("click", saveMeasurement);
  $$('[data-close-modal]').forEach((button) => button.addEventListener("click", () => closeModal(button.dataset.closeModal)));
  $("#prepareTrainingButton").addEventListener("click", startAlignment);
  $("#skipAlignmentButton").addEventListener("click", runCountdown);
  $("#pauseButton").addEventListener("click", togglePause);
  $("#endTrainingButton").addEventListener("click", finishTraining);
  $("#retryTrainingButton").addEventListener("click", goToTrainingReady);
  $("#backHomeButton").addEventListener("click", () => showScreen("home"));
  $("#emergencyButton").addEventListener("click", emergencyStop);
  $("#wakeToggleButton").addEventListener("click", toggleWakeMute);
  $("#closeVoicePanel").addEventListener("click", () => $("#voicePanel").classList.remove("open"));
  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && state.currentScreen === "training" && !state.training.useVision) {
      event.preventDefault();
      registerPunch();
    }
    if (event.key.toLowerCase() === "m" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) toggleWakeMute();
  });
  $(".target-zone").addEventListener("click", () => { if (!state.training.useVision) registerPunch(); });
}

async function init() {
  wireEvents();
  await checkSttStatus();
  await checkDatabaseStatus();
  await checkWakeWordStatus();
  startWakeEventPolling();
  startVisionPolling();
  await loadUsers();
  renderProfiles();
  updateTrainingLabels();
}

init();
