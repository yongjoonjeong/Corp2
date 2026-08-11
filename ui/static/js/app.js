const state = {
  appMode: "user",
  appConfig: {},
  adminStatusTimer: null,
  users: [],
  currentUser: null,
  sessions: [],
  savedReport: null,
  database: { ok: false, users: 0, sessions: 0, path: "instance/ko.sqlite3" },
  currentScreen: "home",
  stream: null,
  poseLandmarker: null,
  poseLoadFailed: false,
  poseLoopToken: 0,
  lastPoseResult: null,
  impactAudioContext: null,
  measurement: {
    stage: "wingspan",
    collecting: false,
    transitioning: false,
    samples: [],
    values: {},
  },
  trainingConfig: {
    type: "straight",
    hand: "right",
    durationSec: 60,
    difficulty: "normal",
    mode: "single",
    combinationId: null,
    sequence: [],
    clientSessionId: null,
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
    evidenceStartVersion: 0,
  },
  vision: {
    connected: false,
    previewAvailable: false,
    frontAvailable: false,
    evidenceAvailable: false,
    previewVersion: 0,
    frontVersion: 0,
    evidenceVersion: 0,
    lastEventId: 0,
    statusTimer: null,
    previewTimer: null,
    eventTimer: null,
    statusBusy: false,
    previewBusy: false,
    eventBusy: false,
    liveStatus: {},
    lastImpactAt: 0,
    healthHistory: [],
    recognitionStatus: "checking",
  },
  force: { available: false, version: 0, lastSeenAt: null, lastHit: null },
  robot: { connected: false, state: "WAITING_BRIDGE", message: "ROS 브리지 연결 대기", lastAnnouncedCalibrationState: "" },
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

const COMBINATIONS = {
  1: { name: "원투", sequence: [
    { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" },
  ] },
  2: { name: "잽잽 스트레이트", sequence: [
    { punch: "jab", role: "lead" }, { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" },
  ] },
  3: { name: "원투 훅", sequence: [
    { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" }, { punch: "hook", role: "lead" },
  ] },
  4: { name: "원투 원투", sequence: [
    { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" }, { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" },
  ] },
  5: { name: "원투 어퍼", sequence: [
    { punch: "jab", role: "lead" }, { punch: "straight", role: "rear" }, { punch: "uppercut", role: "lead" },
  ] },
};

function resolveCombinationSequence(id) {
  const combo = COMBINATIONS[Number(id)];
  if (!combo) return [];
  const dominant = state.currentUser?.dominant_hand === "left" ? "left" : "right";
  const lead = dominant === "right" ? "left" : "right";
  const rear = dominant;
  return combo.sequence.map((step, index) => ({
    index,
    punch: step.punch,
    role: step.role,
    hand: step.role === "lead" ? lead : rear,
  }));
}

function combinationPayload() {
  const id = Number(state.trainingConfig.combinationId || 0);
  const combo = COMBINATIONS[id];
  if (!combo) return null;
  return {
    mode: "combination",
    combination_id: id,
    name: combo.name,
    sequence: resolveCombinationSequence(id),
    duration_sec: state.trainingConfig.durationSec,
    difficulty: state.trainingConfig.difficulty,
  };
}

function createClientSessionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `ko-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function currentTrainingPayload() {
  const user = state.currentUser;
  const common = {
    client_session_id: state.trainingConfig.clientSessionId || "",
    user_id: user?.id ?? null,
    dominant_hand: user?.dominant_hand || "",
    height_cm: user?.height_cm ?? null,
    left_punch_reach_cm: user?.left_punch_reach_cm ?? null,
    right_punch_reach_cm: user?.right_punch_reach_cm ?? null,
    duration_sec: state.trainingConfig.durationSec,
    difficulty: state.trainingConfig.difficulty,
  };
  const combo = combinationPayload();
  if (combo) return { ...common, ...combo };
  const calibrationRoles = ["jab", "straight"].includes(String(state.trainingConfig.type))
    ? [String(state.trainingConfig.type)]
    : [];
  return {
    ...common,
    mode: "single",
    training_type: state.trainingConfig.type,
    punch_type: state.trainingConfig.type,
    hand: state.trainingConfig.hand,
    calibration_roles: calibrationRoles,
  };
}

function validateRobotTrainingProfile() {
  const user = state.currentUser;
  if (!user) throw new Error("먼저 사용자를 선택해 주세요.");
  if (!Number.isFinite(Number(user.height_cm))) throw new Error("사용자 키 정보가 없습니다.");
  const configuredPunches = Array.isArray(state.appConfig.robot_supported_punches)
    ? state.appConfig.robot_supported_punches
    : (state.robot.connected ? ["jab", "straight"] : ["jab", "straight", "hook", "uppercut"]);
  const requestedPunches = state.trainingConfig.mode === "combination"
    ? resolveCombinationSequence(state.trainingConfig.combinationId).map((step) => step.punch)
    : [String(state.trainingConfig.type || "straight")];
  const unsupported = [...new Set(requestedPunches.filter((punch) => !configuredPunches.includes(punch)))];
  if (unsupported.length) {
    const names = unsupported.map((punch) => trainingTypeLabel(punch)).join(", ");
    throw new Error(`현재 실물 로봇 검증 범위는 잽과 스트레이트입니다. ${names} 미트 자세는 실물 경로 검증 후 사용할 수 있습니다.`);
  }
  const left = Number(user.left_punch_reach_cm);
  const right = Number(user.right_punch_reach_cm);
  if (state.trainingConfig.mode === "combination") {
    if (!Number.isFinite(left) || !Number.isFinite(right)) {
      throw new Error("콤비네이션 훈련에는 좌우 펀치 리치 측정값이 모두 필요합니다.");
    }
  } else {
    const selected = state.trainingConfig.hand === "left" ? left : right;
    if (!Number.isFinite(selected)) throw new Error("선택한 손의 펀치 리치 측정값이 필요합니다.");
  }
}

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

const VOICE_HELP_BY_SCREEN = {
  home: ["새 사용자 등록", "등록 사용자 불러오기"],
  profiles: ["사용자 목록", "정우 사용자 선택해"],
  dashboard: ["오른손 스트레이트 1분 훈련", "콤비네이션 1 시작해줘", "최근 훈련 기록 보여줘"],
  setup: ["오른손 스트레이트 1분 훈련", "콤비네이션 3 1분 훈련", "왼손 잽 30초 훈련"],
  history: ["최근 훈련 기록 보여줘", "홈으로 가"],
  register: ["이름은 정우, 키는 175센티, 주 사용 손은 오른손", "저장하고 측정"],
  measure: ["리치 측정 시작", "직접 입력", "카메라 다시 연결"],
  "measure-result": ["측정값 저장", "다시 측정"],
  ready: ["정렬 시작", "바로 시작", "비상정지"],
  countdown: ["비상정지"],
  training: ["현재 기록", "일시정지", "30초 더", "훈련 종료"],
  result: ["결과 읽어줘", "다시 훈련", "홈으로 가"],
  settings: ["홈으로 가"],
};

function currentWakePhrase() {
  return state.wake.display_name || (state.wake.initial_wake_completed ? "케이오" : "웨이크 업 케이오");
}

function updateContextVoiceHelp(screenName = state.currentScreen) {
  const title = $("#contextVoiceTitle");
  const examples = $("#contextVoiceExamples");
  if (!title || !examples) return;
  const commands = VOICE_HELP_BY_SCREEN[screenName] || VOICE_HELP_BY_SCREEN.home;
  const wake = currentWakePhrase();
  title.textContent = "호출어를 말한 뒤 명령 1개를 말씀하세요";
  examples.textContent = commands.slice(0, 3).map((command) => `“${wake}” → “${command}”`).join("  ·  ");
}

function applyAppMode(mode) {
  state.appMode = mode === "admin" ? "admin" : "user";
  document.body.classList.toggle("admin-mode", state.appMode === "admin");
  document.body.classList.toggle("user-mode", state.appMode !== "admin");
  if (state.appMode !== "admin" && state.currentScreen === "settings") {
    state.currentScreen = "home";
  }
  updateContextVoiceHelp(state.currentScreen);
}

async function loadAppConfig() {
  try {
    state.appConfig = await api("/api/app/config");
    applyAppMode(state.appConfig.mode);
  } catch (error) {
    applyAppMode("user");
  }
}

function showScreen(name) {
  if (name === "settings" && state.appMode !== "admin") {
    showToast("시스템 설정은 관리자 모드에서만 사용할 수 있습니다.");
    name = "home";
  }
  state.currentScreen = name;
  $$(".screen").forEach((screen) => screen.classList.toggle("active", screen.id === `screen-${name}`));
  $$(".sport-nav-item").forEach((button) => button.classList.toggle("active", button.dataset.screen === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (!new Set(["measure", "ready", "training"]).has(name)) stopPoseLoop();

  // Noise-test policy: every command requires a fresh wake activation.
  // Keep the UI hint synchronized with the current screen instead of extending sessions.
  updateContextVoiceHelp(name);
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
  const comboMatch = String(type || "").match(/^combination_(\d)$/);
  if (comboMatch && COMBINATIONS[Number(comboMatch[1])]) return `콤비네이션 ${comboMatch[1]} · ${COMBINATIONS[Number(comboMatch[1])].name}`;
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
  const typeInput = form.querySelector(`input[name="type"][value="${state.trainingConfig.type}"]`) || form.querySelector('input[name="type"][value="straight"]');
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
  const selectedType = String(form.get("type") || "straight");
  const comboMatch = selectedType.match(/^combination_(\d)$/);
  configureTraining({
    hand: comboMatch ? "both" : String(form.get("hand") || "right"),
    type: selectedType,
    durationSec: Number(form.get("duration") || 60),
    combinationId: comboMatch ? Number(comboMatch[1]) : null,
  });
  state.trainingConfig.difficulty = String(form.get("difficulty") || "normal");
  await goToTrainingReady();
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
      <p>${escapeHtml(session.feedback || "저장된 코칭 피드백이 없습니다.")}</p>
      <button class="button ghost history-report-button" type="button">보고서 열기 →</button>`;
    item.querySelector(".history-report-button")?.addEventListener("click", () => openSavedReport(session.id));
    list.appendChild(item);
  });
}

function parsedObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch (_) {
    return null;
  }
}

function storedForceSummary(summary) {
  const validHits = Number(summary?.valid_hit_count || 0);
  if (!validHits) return "";
  const pieces = [`유효 힘 타격 ${validHits}회`];
  if (Number.isFinite(Number(summary.average_peak_force_n))) pieces.push(`평균 최대 힘 ${Number(summary.average_peak_force_n).toFixed(1)}N`);
  if (Number.isFinite(Number(summary.average_center_error_mm))) pieces.push(`평균 중심 오차 ${Number(summary.average_center_error_mm).toFixed(1)}mm`);
  const directions = Object.entries(summary.direction_counts || {}).sort((first, second) => Number(second[1]) - Number(first[1]));
  if (directions.length) pieces.push(`주요 방향 ${directions[0][0]}`);
  return `${pieces.join(" · ")}입니다.`;
}

function storedReportModel(details) {
  const session = details?.session || {};
  const report = details?.ai_report || {};
  const raw = report.raw || {};
  const nextObject = parsedObject(report.next_training) || parsedObject(raw.next_training);
  const nextText = !nextObject && typeof report.next_training === "string" ? report.next_training.trim() : "";
  return {
    session,
    headline: raw.headline || "이번 훈련 코칭 요약",
    coach_message: report.coach_message || report.summary || session.feedback || "저장된 코칭 피드백이 없습니다.",
    strengths: report.strengths || raw.strengths || [],
    improvements: report.improvements || raw.improvements || [],
    force_analysis: raw.force_analysis || storedForceSummary(details?.force_summary),
    next_training: nextObject || (nextText ? { title: nextText, duration_sec: 60, goal: "저장된 다음 훈련 목표입니다." } : null),
    progress: details.progress || raw.metrics?.progress || null,
    progress_message: raw.progress_message || "",
    best_comment: raw.best_punch_comment || "가장 높은 이벤트 점수의 타격 순간입니다.",
    worst_comment: raw.check_point_comment || "가장 낮은 이벤트 점수의 타격 순간입니다.",
    best: details.best_punch || null,
    worst: details.check_point || null,
    model: report.model || raw.model || "",
  };
}

function renderSavedProgress(progress, message = "") {
  const card = $("#savedReportProgressCard");
  const tracked = progress?.tracked_result;
  const visible = Boolean(progress?.has_previous || message);
  card?.classList.toggle("hidden", !visible);
  if (!visible) return;
  if (tracked) {
    const status = tracked.status === "improved" ? "좋아졌어요" : tracked.status === "declined" ? "조금 더 신경 써야 해요" : "비슷하게 유지됐어요";
    $("#savedReportProgressTitle").textContent = `${tracked.metric_label || "이전 훈련 비교"} · ${status}`;
  } else {
    $("#savedReportProgressTitle").textContent = progress?.has_previous ? "이전 같은 훈련과 비교" : "이번 훈련부터 발전 추적 시작";
  }
  $("#savedReportProgressText").textContent = message || "이전 훈련의 측정값과 현재 결과를 비교했습니다.";
}

function renderSavedReport(details) {
  const model = storedReportModel(details);
  const session = model.session;
  state.savedReport = model;
  const hand = session.hand === "left" ? "왼손" : session.hand === "both" ? "양손" : "오른손";
  const date = session.created_at ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "long", timeStyle: "short" }).format(new Date(session.created_at)) : "저장 일시 없음";
  $("#savedReportMeta").textContent = `${date} · SESSION #${session.id}`;
  $("#savedReportSubtitle").textContent = `${hand} ${trainingTypeLabel(session.training_type)} · ${session.duration_sec}초`;
  $("#savedReportScore").textContent = Math.round(Number(session.posture_score || 0));
  $("#savedReportPunches").textContent = `${Number(session.punch_count || 0)}회`;
  $("#savedReportSuccess").textContent = `${Number(session.success_rate || 0).toFixed(1)}%`;
  $("#savedReportReaction").textContent = session.avg_reaction_ms ? `${(Number(session.avg_reaction_ms) / 1000).toFixed(2)}초` : "측정 없음";
  $("#savedReportCoachLabel").textContent = model.model ? `KO COACH · ${model.model}` : "KO COACH · SAVED";
  $("#savedReportFeedback").textContent = model.coach_message;
  $("#savedReportHeadline").textContent = model.headline;
  setListContent("#savedReportStrengths", model.strengths, "저장된 강점 항목이 없습니다.");
  setListContent("#savedReportImprovements", model.improvements, "저장된 개선 항목이 없습니다.");
  $("#savedReportForceCard")?.classList.toggle("hidden", !model.force_analysis);
  $("#savedReportForceText").textContent = model.force_analysis;
  $("#savedReportNextCard")?.classList.toggle("hidden", !model.next_training);
  if (model.next_training) {
    $("#savedReportNextTitle").textContent = model.next_training.title || "다음 훈련";
    $("#savedReportNextDuration").textContent = `${Math.round(Number(model.next_training.duration_sec || 60))}초`;
    $("#savedReportNextGoal").textContent = model.next_training.goal || "저장된 개선 목표를 이어가세요.";
  }
  renderSavedProgress(model.progress, model.progress_message);
  const hasEvidence = Boolean(model.best?.evidence_url || model.worst?.evidence_url);
  $("#savedReportEvidenceDuo")?.classList.toggle("hidden", !hasEvidence);
  const bestReview = $("#savedReportBestImage")?.closest("article");
  const worstReview = $("#savedReportWorstImage")?.closest("article");
  bestReview?.classList.toggle("hidden", !model.best?.evidence_url);
  worstReview?.classList.toggle("hidden", !model.worst?.evidence_url);
  if (model.best?.evidence_url) $("#savedReportBestImage").src = model.best.evidence_url;
  if (model.worst?.evidence_url) $("#savedReportWorstImage").src = model.worst.evidence_url;
  $("#savedReportBestText").textContent = model.best_comment;
  $("#savedReportWorstText").textContent = model.worst_comment;
}

async function openSavedReport(sessionId) {
  try {
    const details = await api(`/api/sessions/${Number(sessionId)}/details`);
    renderSavedReport(details);
    showScreen("report-detail");
  } catch (error) {
    showToast(`보고서를 열지 못했습니다: ${error.message}`, "error");
  }
}

function readSavedReport() {
  const report = state.savedReport;
  if (!report) return speak("열린 보고서가 없습니다.");
  const pieces = [report.coach_message];
  if (report.strengths?.length) pieces.push(`잘한 점은 ${report.strengths.join(". ")}`);
  if (report.improvements?.length) pieces.push(`더 신경 쓸 점은 ${report.improvements.join(". ")}`);
  if (report.force_analysis) pieces.push(report.force_analysis);
  if (report.next_training?.title) pieces.push(`다음 추천 훈련은 ${report.next_training.title}, ${Number(report.next_training.duration_sec || 60)}초입니다. ${report.next_training.goal || ""}`);
  speak(pieces.filter(Boolean).join(" "));
}

function updateTrainingLabels() {
  const isCombo = state.trainingConfig.mode === "combination" && state.trainingConfig.combinationId;
  const handText = state.trainingConfig.hand === "right" ? "오른손" : state.trainingConfig.hand === "left" ? "왼손" : "양손";
  const typeText = trainingTypeLabel(state.trainingConfig.type);
  const label = isCombo ? typeText : `${handText} ${typeText}`;
  $("#readyModeChip").textContent = `${label} · ${state.trainingConfig.durationSec}초`;
  $("#countdownMode").textContent = label;
  $("#trainingTitle").textContent = label;
  $("#resultSubtitle").textContent = `${label} · ${state.trainingConfig.durationSec}초`;
  renderCombinationStrip();
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
    await sleep(350);
    await startMeasurementCamera();
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

async function startSharedFrontPoseLoop(image, canvas, onResult) {
  stopPoseLoop();
  const token = state.poseLoopToken;
  const poseLandmarker = await ensurePoseLandmarker();
  const ctx = canvas.getContext("2d");

  const loadNextFrame = () => {
    if (token !== state.poseLoopToken) return;
    image.onload = () => {
      if (token !== state.poseLoopToken) return;
      canvas.width = image.naturalWidth || 640;
      canvas.height = image.naturalHeight || 480;
      try {
        const result = poseLandmarker.detectForVideo(image, performance.now());
        state.lastPoseResult = result;
        drawPose(ctx, canvas, result?.landmarks?.[0]);
        onResult?.(result);
      } catch (error) {
        console.warn("Shared front pose inference error", error);
      }
      window.setTimeout(loadNextFrame, 75);
    };
    image.onerror = () => {
      if (token !== state.poseLoopToken) return;
      $("#cameraMessage").textContent = "전면 RealSense 공유 프레임을 기다리는 중입니다";
      window.setTimeout(loadNextFrame, 250);
    };
    image.src = `/api/vision/front.jpg?t=${Date.now()}`;
  };
  loadNextFrame();
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

const FRONT_TRAINING_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24];

function frontTrainingPoseDetected(result, minimumConfidence = .55) {
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) return false;
  return FRONT_TRAINING_JOINTS.every((index) => {
    const point = landmarks[index];
    return Boolean(
      point
      && (point.visibility ?? 0) >= minimumConfidence
      && (point.presence ?? 1) >= minimumConfidence
    );
  });
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
  state.measurement = { stage: "wingspan", collecting: false, transitioning: false, samples: [], values: {} };
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
  const sharedImage = $("#measurementVisionPreview");
  const canvas = $("#poseCanvas");
  const button = $("#cameraStartButton");
  button.classList.remove("hidden");
  button.disabled = true;
  button.textContent = "카메라 준비 중…";
  try {
    for (let attempt = 0; attempt < 12 && !state.vision.frontAvailable; attempt += 1) {
      await checkVisionStatus();
      if (!state.vision.frontAvailable) await sleep(150);
    }
    if (state.vision.connected) {
      if (!state.vision.frontAvailable) {
        throw new Error("전면 RealSense 공유 프레임이 아직 없습니다. 통합 실행을 다시 시작해 주세요.");
      }
      sharedImage.classList.remove("hidden");
      video.classList.add("hidden");
      await startSharedFrontPoseLoop(sharedImage, canvas, processMeasurementFrame);
      button.textContent = "전면 RealSense 공유 중";
    } else {
      sharedImage.classList.add("hidden");
      video.classList.remove("hidden");
      await ensureCamera(video);
      await startPoseLoop(video, canvas, processMeasurementFrame);
      button.textContent = "브라우저 카메라 연결됨";
    }
    $("#captureMeasureButton").disabled = true;
    $("#captureMeasureButton").classList.add("hidden");
    button.classList.add("hidden");
    $("#cameraMessage").textContent = "자세가 30프레임 연속으로 안정되면 자동 측정됩니다";
    beginMeasurementStage(true);
  } catch (error) {
    button.classList.remove("hidden");
    button.disabled = false;
    button.textContent = "카메라 다시 연결";
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
  if (stage === "done" || state.measurement.transitioning) return;
  let valid = metrics.centered;
  if (stage === "wingspan") {
    valid = valid && metrics.openHandsVisible && metrics.armsHorizontal && metrics.leftAngle > 150 && metrics.rightAngle > 150;
  }
  // 한 팔 리치는 반대쪽 팔이 가드처럼 굽혀져 있어야 인정한다.
  // 양팔을 계속 벌린 상태가 오른팔/왼팔 단계에 연속으로 통과하는 것을 방지한다.
  if (stage === "right") valid = valid && metrics.rightAngle > 150 && metrics.leftAngle < 135;
  if (stage === "left") valid = valid && metrics.leftAngle > 150 && metrics.rightAngle < 135;

  if (!valid) {
    if (state.measurement.collecting && state.measurement.samples.length) {
      state.measurement.samples = [];
      updateMeasurementUI();
    }
    $("#cameraMessage").textContent = stage === "wingspan"
      ? metrics.openHandsVisible ? "양팔을 어깨 높이로 곧게 펴주세요" : "손가락을 펴고 양손 끝이 모두 보이게 해주세요"
      : stage === "right"
        ? "오른팔을 끝까지 뻗고 왼팔은 가드에 두세요"
        : "왼팔을 끝까지 뻗고 오른팔은 가드에 두세요";
    return;
  }

  if (!state.measurement.collecting) beginMeasurementStage(true);
  $("#cameraMessage").textContent = `좋습니다. 자세 유지 ${state.measurement.samples.length}/30`;
  const value = stage === "wingspan" ? metrics.wingspanCm : stage === "right" ? metrics.rightReachCm : metrics.leftReachCm;
  const confidence = stage === "wingspan"
    ? Math.min(metrics.confidence, metrics.openHandConfidence)
    : metrics.confidence;
  state.measurement.samples.push({ value, confidence });
  if (state.measurement.samples.length > 30) state.measurement.samples.shift();
  updateMeasurementUI();
  if (state.measurement.samples.length >= 30) finishMeasurementStage();
}

function beginMeasurementStage(auto = false) {
  if (state.measurement.stage === "done" || state.measurement.transitioning) return;
  state.measurement.collecting = true;
  state.measurement.samples = [];
  const capture = $("#captureMeasureButton");
  if (capture) {
    capture.disabled = true;
    capture.textContent = auto ? "자동 측정 중…" : "측정 중…";
  }
  $("#cameraMessage").textContent = "자세를 유지하세요 · 0/30";
  updateMeasurementUI();
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function finishMeasurementStage() {
  const stage = state.measurement.stage;
  const samples = state.measurement.samples;
  if (!samples.length || state.measurement.transitioning) return;
  state.measurement.transitioning = true;
  state.measurement.values[stage] = median(samples.map((sample) => sample.value));
  state.measurement.values[`${stage}Confidence`] = samples.reduce((sum, sample) => sum + sample.confidence, 0) / samples.length;
  state.measurement.collecting = false;
  state.measurement.samples = [];
  const order = ["wingspan", "right", "left"];
  const nextIndex = order.indexOf(stage) + 1;
  state.measurement.stage = order[nextIndex] || "done";
  updateMeasurementUI();

  if (state.measurement.stage === "done") {
    stopPoseLoop();
    state.measurement.transitioning = false;
    speak("왼팔 리치 측정이 완료되었습니다. 전체 리치 측정이 완료되었습니다. 결과를 확인해 주세요.");
    prepareMeasurementResult({ speakResult: false });
    return;
  }

  const announcement = stage === "wingspan"
    ? "양팔 리치 측정이 완료되었습니다. 오른팔 측정을 시작합니다."
    : "오른팔 리치 측정이 완료되었습니다. 왼팔 측정을 시작합니다.";
  speak(announcement);
  window.setTimeout(() => {
    state.measurement.transitioning = false;
    beginMeasurementStage(true);
  }, 1200);
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

function prepareMeasurementResult({ speakResult = true } = {}) {
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
  if (speakResult) speak("리치 측정이 완료됐습니다. 측정 결과를 확인해 주세요.");
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
  straight_forward_path_off: "스트레이트를 목표 방향으로 더 직선적으로 뻗으세요.",
  straight_path_not_linear: "손목이 좌우나 상하로 흔들리지 않게 직선 궤적을 유지하세요.",
  hook_elbow_angle_off: "훅의 팔꿈치 각도를 조정하세요.",
  hook_elbow_path_off: "훅 팔꿈치 궤적을 더 둥글게 유지하세요.",
  hook_lateral_path_off: "주먹을 바깥쪽 챔버에서 안쪽으로 수평으로 휘두르세요.",
  hook_curve_off: "훅의 방향 전환이 드러나도록 바깥쪽에서 안쪽으로 아크를 그리세요.",
  hook_wrist_elbow_misaligned: "훅에서 손목과 팔꿈치 높이를 맞추세요.",
  uppercut_elbow_angle_off: "어퍼컷 팔꿈치 각도를 조정하세요.",
  uppercut_wrist_path_off: "어퍼컷 손목을 위쪽으로 움직이세요.",
  uppercut_upward_path_off: "아래쪽 챔버에서 시작해 주먹을 위쪽으로 올리세요.",
  uppercut_height_off: "어퍼컷 주먹 높이를 조정하세요.",
};

function setVisionPunchBadge(mode, text) {
  const chip = $("#visionPunchType");
  if (!chip) return;
  chip.textContent = text;
  for (const stateName of ["ready", "active", "impact", "cooldown"]) {
    chip.classList.toggle(stateName, mode === stateName);
  }
}

function resetTargetZonePosition() {
  const target = $("#visionTargetZone");
  if (!target) return;
  target.classList.remove("mitt-target", "mitt-lost");
  for (const property of ["left", "top", "width", "height"]) {
    target.style.removeProperty(property);
  }
}

function updateVisionTargetZone(live = {}) {
  const target = $("#visionTargetZone");
  const stage = target?.parentElement;
  const preview = $("#liveVisionPreview");
  if (!target || !stage || !preview) return;
  if (!state.training.useVision) {
    resetTargetZonePosition();
    return;
  }

  const roi = live.mitt_tracker?.roi_normalized;
  const layout = live.preview_layout || {};
  const tile = layout.front_tile_xywh;
  const frontOnly = preview.dataset.visionSource === "front";
  const validRoi = Array.isArray(roi)
    && roi.length === 4
    && roi.every((value) => Number.isFinite(Number(value)))
    && Number(roi[2]) > Number(roi[0])
    && Number(roi[3]) > Number(roi[1]);
  if (!validRoi || (!frontOnly && (!Array.isArray(tile) || tile.length !== 4))) {
    target.classList.add("mitt-target", "mitt-lost");
    return;
  }

  const frontSize = Array.isArray(layout.front_frame_size) ? layout.front_frame_size : [640, 480];
  const canvasWidth = frontOnly
    ? Number(frontSize[0] || preview.naturalWidth || 640)
    : Number(layout.canvas_width || preview.naturalWidth || 1440);
  const canvasHeight = frontOnly
    ? Number(frontSize[1] || preview.naturalHeight || 480)
    : Number(layout.canvas_height || preview.naturalHeight || 360);
  const scale = Math.min(stage.clientWidth / canvasWidth, stage.clientHeight / canvasHeight);
  if (!Number.isFinite(scale) || scale <= 0) return;
  const imageLeft = (stage.clientWidth - canvasWidth * scale) / 2;
  const imageTop = (stage.clientHeight - canvasHeight * scale) / 2;
  const [tileX, tileY, tileWidth, tileHeight] = frontOnly
    ? [0, 0, canvasWidth, canvasHeight]
    : tile.map(Number);
  const x1 = imageLeft + (tileX + Number(roi[0]) * tileWidth) * scale;
  const y1 = imageTop + (tileY + Number(roi[1]) * tileHeight) * scale;
  const x2 = imageLeft + (tileX + Number(roi[2]) * tileWidth) * scale;
  const y2 = imageTop + (tileY + Number(roi[3]) * tileHeight) * scale;
  const diameter = Math.max(48, Math.min(116, Math.min(x2 - x1, y2 - y1) * 0.92));

  target.style.left = `${(x1 + x2 - diameter) / 2}px`;
  target.style.top = `${(y1 + y2 - diameter) / 2}px`;
  target.style.width = `${diameter}px`;
  target.style.height = `${diameter}px`;
  target.classList.add("mitt-target");
  target.classList.remove("mitt-lost");
}

function formatVector(values) {
  if (!Array.isArray(values) || values.length < 3) return "—";
  return values.slice(0, 3).map((value) => Number(value).toFixed(0)).join(", ");
}

function renderVisionTelemetry(live) {
  for (const side of ["left", "right"]) {
    const fist = live.fists?.[side];
    const position = $(`#${side}FistPosition`);
    const velocity = $(`#${side}FistVelocity`);
    if (position) position.textContent = fist ? `P ${formatVector(fist.position_base_mm)}` : "WAITING";
    if (velocity) velocity.textContent = fist ? `V ${formatVector(fist.velocity_base_mm_s)} mm/s` : "V —";
  }
  const guard = Math.max(0, Number(live.guard_count || 0));
  const goal = Math.max(1, Number(live.guard_goal || 4));
  if ($("#webGuardText")) $("#webGuardText").textContent = `${Math.min(guard, goal)}/${goal}`;
  if ($("#webGuardFill")) $("#webGuardFill").style.width = `${Math.min(100, guard / goal * 100)}%`;

  const detectorState = String(live.detector_state || "").toUpperCase();
  const impactAgeMs = Date.now() - state.vision.lastImpactAt;
  if (impactAgeMs >= 0 && impactAgeMs < 300) {
    setVisionPunchBadge("impact", "IMPACT");
  } else if (impactAgeMs >= 300 && impactAgeMs < 750) {
    setVisionPunchBadge("cooldown", "COOLDOWN");
  } else if (detectorState === "READY") {
    setVisionPunchBadge("ready", "READY");
  } else if (detectorState === "ACTIVE") {
    setVisionPunchBadge("active", "ACTIVE");
  } else if (detectorState === "IMPACT") {
    setVisionPunchBadge("impact", "IMPACT");
  } else if (detectorState === "COOLDOWN") {
    setVisionPunchBadge("cooldown", "COOLDOWN");
  } else {
    setVisionPunchBadge("neutral", detectorState === "WAIT_GUARD" ? "GUARD" : detectorState || "WAITING");
  }
  updateVisionTargetZone(live);
}

function startVisionPolling() {
  clearInterval(state.vision.statusTimer);
  clearInterval(state.vision.previewTimer);
  clearInterval(state.vision.eventTimer);
  state.vision.statusTimer = setInterval(checkVisionStatus, 120);
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

function preferredVisionFeed() {
  // ADMIN MODE: use the annotated LEFT / FRONT / RIGHT composite produced by
  // the current three-camera vision runtime. Fall back to the front feed only
  // while the composite is not ready.
  if (state.appMode === "admin") {
    if (state.vision.previewAvailable) {
      return { source: "triptych", path: "/api/vision/preview.jpg", version: state.vision.previewVersion };
    }
    if (state.vision.frontAvailable) {
      return { source: "front", path: "/api/vision/front.jpg", version: state.vision.frontVersion };
    }
    return null;
  }

  // USER MODE: keep the clean front-camera view. Detailed multi-camera
  // overlays stay exclusive to ADMIN MODE even though the same vision runtime
  // continues to operate in the background.
  if (state.vision.frontAvailable) {
    return { source: "front", path: "/api/vision/front.jpg", version: state.vision.frontVersion };
  }
  return null;
}

function refreshVisionPreview() {
  const feed = preferredVisionFeed();
  if (!state.vision.connected || !feed || state.vision.previewBusy) return;
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
  image.dataset.visionSource = feed.source;
  image.src = `${feed.path}?t=${Date.now()}`;
}

function setDiagnosticDot(id, level) {
  const dot = $(`#${id}`);
  if (!dot) return;
  dot.classList.remove("ready", "warning", "error");
  if (level === "ready") dot.classList.add("ready");
  else if (level === "warning") dot.classList.add("warning");
  else if (level === "error") dot.classList.add("error");
}

function cameraHealth(live, name) {
  const camera = live?.cameras?.[name];
  if (!camera) return { ok: false, text: "상태 없음" };
  const frames = Number(camera.frames || 0);
  const error = String(camera.error || "").trim();
  return { ok: frames > 0 && !error, text: error || (frames > 0 ? `${frames} frames` : "프레임 대기") };
}

function computeVisionFrameHealth(status) {
  const live = status?.live_status || {};
  const cameraStates = ["front", "left", "right"].map((name) => cameraHealth(live, name));
  const camerasOk = cameraStates.every((camera) => camera.ok);
  const targetOk = Boolean(live.target_locked || String(live.target_state || "").toUpperCase() === "LOCKED");
  const mittState = String(live.mitt_tracker?.state || "").toUpperCase();
  const mittOk = mittState === "TRACKED" || mittState === "PREDICTED";
  const fistStates = ["left", "right"].map((side) => live.fists?.[side]).filter(Boolean);
  const validFists = fistStates.filter((fist) => Boolean(fist.valid) && Number(fist.camera_count || 0) >= 2).length;
  const poseEntries = Object.values(live.pose || {});
  const poseOk = poseEntries.length === 0 || poseEntries.every((entry) => !String(entry?.error || "").trim());
  const connected = Boolean(status?.connected);
  const allOk = connected && camerasOk && targetOk && mittOk && validFists >= 2 && poseOk;
  const presence = connected && camerasOk && (targetOk || validFists > 0);
  let reason = "사용자·양손·미트가 정상적으로 인식되고 있습니다.";
  if (!connected) reason = "비전 시스템 연결을 확인해 주세요.";
  else if (!camerasOk) reason = "카메라 연결 상태를 확인해 주세요.";
  else if (!targetOk) reason = "카메라 중앙에서 자세를 잡아 주세요.";
  else if (!mittOk) reason = "미트를 인식할 수 없습니다. 미트가 카메라에 보이게 해 주세요.";
  else if (validFists < 2) reason = "양손이 카메라에 보이도록 자세를 유지해 주세요.";
  else if (!poseOk) reason = "자세 인식이 불안정합니다. 잠시 자세를 유지해 주세요.";
  return { allOk, presence, reason, camerasOk, targetOk, mittOk, validFists, poseOk };
}

function updateUserVisionRecognition(status) {
  const health = computeVisionFrameHealth(status);
  const history = state.vision.healthHistory;
  history.push({ ok: health.allOk, presence: health.presence });
  while (history.length > 15) history.shift();

  let mode = "checking";
  if (!status?.connected) {
    mode = "unavailable";
  } else if (history.length >= 5) {
    const okCount = history.filter((sample) => sample.ok).length;
    const presenceCount = history.filter((sample) => sample.presence).length;
    const normalThreshold = Math.ceil(history.length * 0.8);
    const presenceThreshold = Math.ceil(history.length * 0.4);
    if (okCount >= normalThreshold) mode = "normal";
    else if (presenceCount >= presenceThreshold) mode = "unstable";
    else mode = "unavailable";
  }
  state.vision.recognitionStatus = mode;
  const indicator = $("#userVisionIndicator");
  const title = $("#userVisionStatusTitle");
  const message = $("#userVisionStatusMessage");
  if (indicator) indicator.className = `user-vision-indicator ${mode}`;
  if (title) title.textContent = mode === "normal" ? "정상 인식" : mode === "unstable" ? "인식 불안정" : mode === "unavailable" ? "인식 불가" : "인식 확인 중";
  if (message) message.textContent = mode === "normal" ? "사용자·양손·미트가 정상적으로 인식되고 있습니다." : health.reason;
  return health;
}

function updateAdminVisionDiagnostics(status, health) {
  if (state.appMode !== "admin") return;
  const live = status?.live_status || {};
  for (const [name, prefix] of [["front", "front"], ["left", "left"], ["right", "right"]]) {
    const camera = cameraHealth(live, name);
    setDiagnosticDot(`${prefix}CameraStatusDot`, camera.ok ? "ready" : "error");
    const text = $(`#${prefix}CameraStatusText`);
    if (text) text.textContent = camera.text;
  }
  setDiagnosticDot("targetStatusDot", health.targetOk ? "ready" : status?.connected ? "warning" : "error");
  if ($("#targetStatusText")) $("#targetStatusText").textContent = String(live.target_state || "WAITING");
  setDiagnosticDot("mittStatusDot", health.mittOk ? "ready" : status?.connected ? "warning" : "error");
  if ($("#mittStatusText")) $("#mittStatusText").textContent = String(live.mitt_tracker?.state || "WAITING");
  setDiagnosticDot("fist3dStatusDot", health.validFists >= 2 ? "ready" : health.validFists === 1 ? "warning" : "error");
  if ($("#fist3dStatusText")) $("#fist3dStatusText").textContent = `${health.validFists}/2 valid`;

  if ($("#adminTargetState")) $("#adminTargetState").textContent = String(live.target_state || "—");
  if ($("#adminMittState")) $("#adminMittState").textContent = String(live.mitt_tracker?.state || "—");
  if ($("#adminImpactState")) $("#adminImpactState").textContent = String(live.detector_state || live.impact_state || "—");
  if ($("#adminGuardState")) $("#adminGuardState").textContent = `${Number(live.guard_count || 0)}/${Number(live.guard_goal || 4)}`;
  const left = live.fists?.left;
  const right = live.fists?.right;
  if ($("#adminLeftP")) $("#adminLeftP").textContent = left ? `P ${formatVector(left.position_base_mm)}` : "WAITING";
  if ($("#adminLeftV")) $("#adminLeftV").textContent = left ? `V ${formatVector(left.velocity_base_mm_s)} mm/s` : "V —";
  if ($("#adminRightP")) $("#adminRightP").textContent = right ? `P ${formatVector(right.position_base_mm)}` : "WAITING";
  if ($("#adminRightV")) $("#adminRightV").textContent = right ? `V ${formatVector(right.velocity_base_mm_s)} mm/s` : "V —";
}

async function checkRobotStatus() {
  try {
    const previousAnnouncement = String(state.robot?.lastAnnouncedCalibrationState || "");
    const status = await api("/api/robot/status");
    state.robot = { ...status, lastAnnouncedCalibrationState: previousAnnouncement };
    const connected = Boolean(state.robot.connected);
    setDiagnosticDot("robotStatusDot", connected ? "ready" : "warning");
    if ($("#robotStatusText")) $("#robotStatusText").textContent = state.robot.state || state.robot.message || (connected ? "ONLINE" : "WAITING");

    const robotState = String(state.robot.state || "");
    const calibrationFallback = {
      REACH_CALIBRATION_POSE: "1차 리치 보정 · 비주손을 앞으로 끝까지 뻗고 움직이지 마세요.",
      REACH_CALIBRATION_BASELINE: "1차 리치 보정 · 팔을 유지하세요. 힘 기준을 측정하고 있습니다.",
      REACH_CALIBRATION_APPROACH: "1차 리치 보정 · 미트가 Tool +Z 방향으로 천천히 접근합니다.",
      REACH_CALIBRATION_CONTACT_SAVED: "1차 리치 보정 접촉 위치 저장 완료 · 팔을 내려주세요.",
      REACH_CALIBRATION_COMPLETE: "1차 리치 보정 완료 · 2차 타격 위치 보정을 준비합니다.",
      MITT_CALIBRATION_ZEROING: "2차 미트 보정 · 힘 센서 영점 조정 중입니다.",
      MITT_CALIBRATION_PUNCH_READY: "2차 미트 보정 · 미트를 5회 펀치해 주세요.",
      MITT_CALIBRATION_ADJUSTING: "2차 미트 보정 · 타격 방향에 따라 미트 위치를 조정하고 있습니다.",
      MITT_CALIBRATION_COMPLETE: "2차 미트 보정 완료 · 실제 훈련을 시작합니다.",
    };
    const calibrationMessage = calibrationFallback[robotState]
      ? String(state.robot.message || calibrationFallback[robotState])
      : "";
    if (calibrationMessage) {
      if ($("#alignmentFeedback")) $("#alignmentFeedback").textContent = calibrationMessage;
      if ($("#liveStatus")) $("#liveStatus").textContent = robotState.startsWith("REACH_")
        ? "1차 리치 보정 진행 중"
        : "2차 5회 펀치 보정 진행 중";
      if ($("#liveCommand")) $("#liveCommand").textContent = calibrationMessage;
      const button = $("#prepareTrainingButton");
      if (button && state.currentScreen === "ready") button.textContent = "캘리브레이션 진행 중";
      if (previousAnnouncement !== robotState) {
        state.robot.lastAnnouncedCalibrationState = robotState;
        if ([
          "REACH_CALIBRATION_POSE",
          "REACH_CALIBRATION_APPROACH",
          "REACH_CALIBRATION_CONTACT_SAVED",
          "MITT_CALIBRATION_ZEROING",
          "MITT_CALIBRATION_PUNCH_READY",
        ].includes(robotState)) speak(calibrationMessage);
      }
    } else {
      state.robot.lastAnnouncedCalibrationState = "";
    }
  } catch (error) {
    state.robot.connected = false;
    setDiagnosticDot("robotStatusDot", "error");
    if ($("#robotStatusText")) $("#robotStatusText").textContent = "상태 확인 실패";
  }
}

async function checkForceStatus() {
  try {
    const status = await api("/api/force/status");
    state.force.available = Boolean(status.available);
    state.force.version = Number(status.version || 0);
    state.force.lastSeenAt = status.last_seen_at || null;
    state.force.lastHit = status.last_hit || null;
    const target = $("#adminForceStatusText");
    if (target) {
      const hit = state.force.lastHit?.payload || state.force.lastHit || null;
      const peak = Number(hit?.peak_force_n);
      const errorMm = Number(hit?.center_error_mm);
      target.textContent = state.force.available
        ? `${Number.isFinite(peak) ? `${peak.toFixed(1)} N` : `DATA · v${state.force.version}`}${Number.isFinite(errorMm) ? ` · ${errorMm.toFixed(0)} mm` : ""}`
        : "WAITING FOR DATA";
    }
    setDiagnosticDot("forceStatusDot", state.force.available ? "ready" : "warning");
    return status;
  } catch (error) {
    state.force.available = false;
    return state.force;
  }
}

function renderEvidencePair(summary) {
  const best = summary?.best || null;
  const worst = summary?.worst || null;
  const duo = $("#resultEvidenceDuo");
  const hasEvidence = Boolean(best?.evidence_url || worst?.evidence_url);
  duo?.classList.toggle("hidden", !hasEvidence);
  if (best) {
    $("#resultBestTitle").textContent = `#${best.punch_index || "—"} · ${Math.round(Number(best.event_score || 0))}점`;
    $("#resultBestText").textContent = "이번 훈련에서 가장 높은 이벤트 점수를 기록한 타격입니다.";
    if (best.evidence_url) $("#resultBestImage").src = `${best.evidence_url}?v=${Date.now()}`;
  }
  if (worst) {
    $("#resultWorstTitle").textContent = `#${worst.punch_index || "—"} · ${Math.round(Number(worst.event_score || 0))}점`;
    const tags = (worst.issue_tags || []).join(", ");
    $("#resultWorstText").textContent = tags ? `확인 항목: ${tags}` : "이번 훈련에서 가장 낮은 이벤트 점수를 기록한 타격입니다.";
    if (worst.evidence_url) $("#resultWorstImage").src = `${worst.evidence_url}?v=${Date.now()}`;
  }
}

function renderProgress(progress, message = "") {
  const card = $("#resultProgressCard");
  const tracked = progress?.tracked_result;
  const hasPrevious = Boolean(progress?.has_previous);
  card?.classList.toggle("hidden", !hasPrevious && !message);
  if (!hasPrevious && message) {
    $("#resultProgressTitle").textContent = "이번 훈련부터 발전 추적 시작";
    $("#resultProgressText").textContent = message;
    return;
  }
  if (tracked) {
    const statusText = tracked.status === "improved" ? "좋아졌어요" : tracked.status === "declined" ? "조금 더 신경 써야 해요" : "비슷하게 유지됐어요";
    $("#resultProgressTitle").textContent = `${tracked.metric_label} · ${statusText}`;
  } else {
    $("#resultProgressTitle").textContent = "이전 같은 훈련과 비교";
  }
  $("#resultProgressText").textContent = message || "이전 훈련의 측정값과 현재 결과를 비교했습니다.";
}


function setListContent(id, items, fallback = "측정 가능한 항목을 분석하고 있습니다.") {
  const list = $(id);
  if (!list) return;
  const values = Array.isArray(items) ? items.filter(Boolean).slice(0, 3) : [];
  list.innerHTML = "";
  if (!values.length) {
    const li = document.createElement("li");
    li.textContent = fallback;
    list.appendChild(li);
    return;
  }
  values.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = String(value);
    list.appendChild(li);
  });
}

function renderReportDetails(coaching = {}) {
  const details = $("#resultReportDetails");
  if (!details) return;
  const strengths = Array.isArray(coaching.strengths) ? coaching.strengths : [];
  const improvements = Array.isArray(coaching.improvements) ? coaching.improvements : [];
  const next = coaching.next_training && typeof coaching.next_training === "object" ? coaching.next_training : null;
  const forceText = String(coaching.force_analysis || "").trim();
  const hasContent = Boolean(strengths.length || improvements.length || next || forceText || coaching.headline);
  details.classList.toggle("hidden", !hasContent);
  if (!hasContent) return;
  if ($("#resultHeadline")) $("#resultHeadline").textContent = coaching.headline || "이번 훈련 코칭 요약";
  setListContent("#resultStrengths", strengths, "이번 훈련의 강점을 확인하고 있습니다.");
  setListContent("#resultImprovements", improvements, "다음에 집중할 항목을 확인하고 있습니다.");
  const forceCard = $("#resultForceCard");
  if (forceCard) forceCard.classList.toggle("hidden", !forceText);
  if ($("#resultForceText")) $("#resultForceText").textContent = forceText || "";
  const nextCard = $("#resultNextTrainingCard");
  if (nextCard) nextCard.classList.toggle("hidden", !next);
  if (next) {
    $("#resultNextTrainingTitle").textContent = next.title || "다음 훈련";
    $("#resultNextTrainingGoal").textContent = next.goal || coaching.next_focus || "이번 개선 목표를 이어가세요.";
    $("#resultNextTrainingDuration").textContent = `${Math.round(Number(next.duration_sec || 60))}초`;
  }
}

function readCurrentReport() {
  const report = state.training.report || {};
  const pieces = [];
  const progress = String($("#resultProgressText")?.textContent || "").trim();
  const feedback = String($("#resultFeedback")?.textContent || "").trim();
  if (feedback) pieces.push(feedback);
  if (!$("#resultProgressCard")?.classList.contains("hidden") && progress) pieces.push(progress);
  const strengths = Array.isArray(report.strengths) ? report.strengths : [];
  const improvements = Array.isArray(report.improvements) ? report.improvements : [];
  if (strengths.length) pieces.push(`잘한 점은 ${strengths.join(". ")}`);
  if (improvements.length) pieces.push(`더 신경 쓸 점은 ${improvements.join(". ")}`);
  if (report.force_analysis) pieces.push(String(report.force_analysis));
  const next = report.next_training;
  if (next?.title) pieces.push(`다음 추천 훈련은 ${next.title}, ${Number(next.duration_sec || 60)}초입니다. ${next.goal || ""}`);
  speak(pieces.filter(Boolean).join(" ") || "아직 읽을 훈련 결과가 없습니다.");
}

function startAdminStatusPolling() {
  clearInterval(state.adminStatusTimer);
  checkRobotStatus();
  checkForceStatus();
  state.adminStatusTimer = setInterval(() => { checkRobotStatus(); checkForceStatus(); }, 1000);
}

async function checkVisionStatus() {
  if (state.vision.statusBusy) return state.vision;
  state.vision.statusBusy = true;
  try {
    const status = await api("/api/vision/status");
    state.vision.connected = Boolean(status.connected);
    state.vision.previewAvailable = Boolean(status.preview_available);
    state.vision.frontAvailable = Boolean(status.front_available);
    state.vision.evidenceAvailable = Boolean(status.evidence_available);
    state.vision.liveStatus = status.live_status || {};
    renderVisionTelemetry(state.vision.liveStatus);
    const visionHealth = updateUserVisionRecognition(status);
    updateAdminVisionDiagnostics(status, visionHealth);
    const dot = $("#visionStatusDot");
    dot?.classList.toggle("ready", state.vision.connected);
    const readyBadge = $(".vision-source-badge");
    readyBadge?.classList.toggle("connected", state.vision.connected);
    if ($("#readyVisionSource")) {
      let sourceLabel = "카메라 연결 대기";
      if (state.vision.connected) {
        if (state.appMode === "admin" && state.vision.previewAvailable) {
          sourceLabel = "LEFT · FRONT · RIGHT 3카메라 인식";
        } else if (state.appMode === "admin") {
          sourceLabel = "전면 카메라 연결됨 · 3카메라 프리뷰 대기";
        } else {
          sourceLabel = "전면 카메라 인식";
        }
      }
      $("#readyVisionSource").textContent = sourceLabel;
    }
    if (status.preview_version && status.preview_version !== state.vision.previewVersion) {
      state.vision.previewVersion = Number(status.preview_version);
    }
    if (status.front_version && status.front_version !== state.vision.frontVersion) {
      state.vision.frontVersion = Number(status.front_version);
    }
    if (status.evidence_version && status.evidence_version !== state.vision.evidenceVersion) {
      state.vision.evidenceVersion = Number(status.evidence_version);
      const url = `/api/vision/evidence.jpg?v=${state.vision.evidenceVersion}`;
      const live = $("#liveEvidenceImage");
      if (live) { live.src = url; live.classList.remove("hidden"); }
    }
    return status;
  } catch (error) {
    state.vision.connected = false;
    state.vision.frontAvailable = false;
    updateUserVisionRecognition({ connected: false, live_status: {} });
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
  const feed = preferredVisionFeed();
  if (!feed) return;
  image?.classList.remove("hidden");
  video?.classList.add("vision-hidden");
  canvas?.classList.add("vision-hidden");
  if (image) {
    image.dataset.visionSource = feed.source;
    image.src = `${feed.path}?v=${feed.version || Date.now()}`;
  }
}

function useBrowserCamera(screenName) {
  const isReady = screenName === "ready";
  const image = isReady ? $("#readyVisionPreview") : $("#liveVisionPreview");
  const video = isReady ? $("#trainingVideo") : $("#liveVideo");
  const canvas = isReady ? $("#trainingCanvas") : $("#liveCanvas");
  image?.classList.add("hidden");
  video?.classList.remove("vision-hidden");
  canvas?.classList.remove("vision-hidden");
  if (!isReady) resetTargetZonePosition();
}

async function pollVisionEvents() {
  if (state.vision.eventBusy) return;
  state.vision.eventBusy = true;
  try {
    const result = await api(`/api/vision/events?after=${state.vision.lastEventId}`);
    for (const event of result.events || []) {
      state.vision.lastEventId = Math.max(state.vision.lastEventId, Number(event.id) || 0);
      if (state.appMode === "admin") {
        const payload = event.payload || {};
        const side = payload.punch_side ? String(payload.punch_side).toUpperCase() : "";
        if ($("#adminRecentEvent")) $("#adminRecentEvent").textContent = event.type === "punch" ? `IMPACT ${side} · #${payload.punch_id || "—"}` : String(event.type || "event");
        if ($("#adminRecentEventTime")) $("#adminRecentEventTime").textContent = new Date().toLocaleTimeString("ko-KR", { hour12: false });
      }
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
  if (state.trainingConfig.mode !== "combination" && configuredHand !== "both" && payload.punch_side && payload.punch_side !== configuredHand) {
    $("#visionFeedbackTitle").textContent = "반대 손 펀치가 감지됐습니다";
    $("#visionFeedbackText").textContent = "현재 선택한 훈련 손으로 다시 시도하세요.";
    return;
  }

  const t = state.training;
  const score = Number(payload.total_score || 0);
  const passed = Boolean(payload.passed);
  t.lastPunchAt = performance.now();
  state.vision.lastImpactAt = Date.now();
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
    : "타격 순간과 좌·정면·우 영상을 저장했습니다. 가드로 빠르게 복귀하세요.";
  t.lastFeedback = feedback;

  const side = payload.punch_side === "left" ? "왼손" : "오른손";
  const typeMap = { straight: "스트레이트", hook: "훅", uppercut: "어퍼컷", impact: "타격" };
  const type = typeMap[payload.punch_type] || String(payload.punch_type || "펀치");
  $("#liveVisionScore").textContent = score.toFixed(0);
  setVisionPunchBadge("impact", `IMPACT · ${side}`);
  $("#visionFeedbackTitle").textContent = passed ? "IMPACT · 타격 순간 포착" : "타격을 다시 확인하세요";
  $("#visionFeedbackText").textContent = feedback;
  $("#liveStatus").textContent = `${type} 감지 · ${score.toFixed(1)}점`;
  $("#liveCommand").textContent = passed ? "좋아요! 가드로 복귀" : feedback;
  pulseTarget(passed);
  if (passed) advanceCombinationStep();
  updateMetrics();
}

function dominantViolationText() {
  const entries = Object.entries(state.training.violationCounts || {});
  if (!entries.length) return state.training.lastFeedback || "안정적인 자세를 유지했습니다.";
  entries.sort((a, b) => b[1] - a[1]);
  return VISION_FEEDBACK_TEXT[entries[0][0]] || state.training.lastFeedback || "자세를 조금 더 안정적으로 유지하세요.";
}

function configureTraining({ hand, type, durationSec, combinationId = undefined } = {}) {
  state.trainingConfig.clientSessionId = null;
  if (hand) state.trainingConfig.hand = hand;
  if (type) state.trainingConfig.type = type;
  if (durationSec) state.trainingConfig.durationSec = durationSec;
  if (combinationId !== undefined) {
    state.trainingConfig.combinationId = combinationId;
    state.trainingConfig.mode = combinationId ? "combination" : "single";
    state.trainingConfig.sequence = combinationId ? resolveCombinationSequence(combinationId) : [];
    if (combinationId) {
      state.trainingConfig.type = `combination_${combinationId}`;
      state.trainingConfig.hand = "both";
    }
  } else if (!String(state.trainingConfig.type).startsWith("combination_")) {
    state.trainingConfig.mode = "single";
    state.trainingConfig.combinationId = null;
    state.trainingConfig.sequence = [];
  }
  updateTrainingLabels();
}

function renderCombinationStrip() {
  const strip = $("#combinationLiveStrip");
  if (!strip) return;
  const isCombo = state.trainingConfig.mode === "combination" && state.trainingConfig.combinationId;
  strip.classList.toggle("hidden", !isCombo);
  if (!isCombo) { strip.innerHTML = ""; return; }
  const sequence = state.trainingConfig.sequence.length ? state.trainingConfig.sequence : resolveCombinationSequence(state.trainingConfig.combinationId);
  const active = Number(state.training.comboStepIndex || 0) % Math.max(1, sequence.length);
  strip.innerHTML = sequence.map((step, index) => {
    const hand = step.hand === "left" ? "L" : "R";
    const label = trainingTypeLabel(step.punch);
    const cls = index === active ? "active" : index < active ? "done" : "";
    return `<span class="combo-step ${cls}"><i>${index + 1}</i><b>${label}</b><small>${hand}</small></span>`;
  }).join('<em>→</em>');
}

function advanceCombinationStep() {
  if (state.trainingConfig.mode !== "combination") return;
  const sequence = state.trainingConfig.sequence || [];
  if (!sequence.length) return;
  state.training.comboStepIndex = (Number(state.training.comboStepIndex || 0) + 1) % sequence.length;
  if (state.training.comboStepIndex === 0) state.training.comboRounds = Number(state.training.comboRounds || 0) + 1;
  renderCombinationStrip();
  const next = sequence[state.training.comboStepIndex];
  if (next) $("#liveCommand").textContent = `${next.hand === "left" ? "왼손" : "오른손"} ${trainingTypeLabel(next.punch)}`;
}

async function goToTrainingReady() {
  if (!state.currentUser) {
    showToast("먼저 사용자 프로필을 선택해 주세요.");
    showScreen("profiles");
    return;
  }
  try {
    validateRobotTrainingProfile();
  } catch (error) {
    showToast(error.message, "error");
    speak(error.message);
    return;
  }
  state.trainingConfig.clientSessionId = createClientSessionId();
  $("#readyUserName").textContent = state.currentUser.name;
  $("#readyDistance").textContent = state.currentUser.recommended_distance_cm ? `${Math.round(state.currentUser.recommended_distance_cm)} cm` : "측정 필요";
  updateTrainingLabels();
  showScreen("ready");
  if ($("#alignmentFeedback")) $("#alignmentFeedback").textContent = "위빙을 유지한 채 정면 MediaPipe 관절을 확인합니다";
  speak(`${state.currentUser.name}님, 정면 카메라에 상체와 양손이 보이게 서주세요. 관절 인식이 완료되면 로봇이 펀칭 대기 위치로 이동합니다.`);
}

async function waitForRobotTrainingReady(timeoutMs = 180000) {
  const startedAt = Date.now();
  const failureTokens = ["FAILED", "REJECTED", "ERROR", "EMERGENCY_STOP"];
  while (Date.now() - startedAt < timeoutMs) {
    await checkRobotStatus();
    const robotState = String(state.robot?.state || "");
    if (robotState === "TRAINING_READY") return true;
    if (failureTokens.some((token) => robotState.includes(token))) {
      throw new Error(state.robot?.error_detail || state.robot?.message || `로봇 준비 실패: ${robotState}`);
    }
    const preparationMessage = state.robot?.message || "로봇 미트/힘 센서 준비 중";
    if ($("#alignmentFeedback")) $("#alignmentFeedback").textContent = preparationMessage;
    if (state.currentScreen === "training") {
      if ($("#liveStatus")) $("#liveStatus").textContent = "로봇 훈련 준비 중";
      if ($("#liveCommand")) $("#liveCommand").textContent = preparationMessage;
    }
    await sleep(250);
  }
  throw new Error("리치/미트 위치 보정이 완료되지 않았습니다. 로봇/힘제어 상태를 확인해 주세요.");
}

async function waitForForceSettle(timeoutMs = 1200, stableMs = 300) {
  const startedAt = Date.now();
  await checkForceStatus();
  let lastVersion = Number(state.force.version || 0);
  let stableSince = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    await sleep(100);
    await checkForceStatus();
    const version = Number(state.force.version || 0);
    if (version !== lastVersion) {
      lastVersion = version;
      stableSince = Date.now();
    }
    if (Date.now() - stableSince >= stableMs) return true;
  }
  return false;
}

async function waitForRobotTrainingStop(timeoutMs = 5000) {
  if (!state.robot?.connected) {
    await waitForForceSettle();
    return false;
  }
  const startedAt = Date.now();
  const completeStates = new Set([
    "WEAVE_RESTART_REQUESTED", "MOVING_WEAVE_READY", "WEAVING", "READY",
  ]);
  const failureTokens = ["SESSION_STOP_FAILED", "ERROR", "EMERGENCY_STOP"];
  while (Date.now() - startedAt < timeoutMs) {
    await checkRobotStatus();
    const robotState = String(state.robot?.state || "");
    if (completeStates.has(robotState)) {
      // Freeze the result only after ForceHub's version has stopped changing.
      await waitForForceSettle();
      return true;
    }
    if (failureTokens.some((token) => robotState.includes(token))) {
      await waitForForceSettle();
      return false;
    }
    await sleep(120);
  }
  await waitForForceSettle();
  return false;
}

async function startAlignment() {
  const button = $("#prepareTrainingButton");
  button.disabled = true;
  button.textContent = "비전 확인 중…";
  await checkVisionStatus();

  if (state.vision.connected && preferredVisionFeed()) {
    useVisionPreview("ready");
    $("#checkVision").classList.add("ok");
    for (let index = 0; index < 80; index += 1) {
      if (state.currentScreen !== "ready") return;
      const live = state.vision.liveStatus || {};
      const poseOk = Boolean(live.front_pose_detected);
      $("#checkPose").classList.toggle("ok", poseOk);
      $("#checkDistance").classList.toggle("ok", poseOk);
      $("#alignmentFeedback").textContent = poseOk
        ? "정면 8개 관절 검출 완료"
        : "정면 카메라에 어깨·양팔·골반이 보이게 서주세요";
      if (poseOk) {
        button.textContent = "관절 검출 완료";
        await runCountdown();
        return;
      }
      await sleep(150);
    }
    button.disabled = false;
    button.textContent = "다시 확인";
    showToast("정면 관절 확인 시간이 초과됐습니다. 상체와 양손을 보인 뒤 다시 시도하세요.");
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
    await startPoseLoop(video, canvas, (result) => {
      const poseOk = frontTrainingPoseDetected(result);
      $("#checkPose").classList.toggle("ok", poseOk);
      $("#checkDistance").classList.toggle("ok", poseOk);
      $("#alignmentFeedback").textContent = poseOk
        ? "정면 8개 관절 검출 완료"
        : "정면 카메라에 어깨·양팔·골반이 보이게 서주세요";
      if (poseOk) {
        stopPoseLoop();
        button.textContent = "관절 검출 완료";
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
  try {
    if ($("#alignmentFeedback")) $("#alignmentFeedback").textContent = "정면 관절 검출 완료 · 위빙 정지 후 펀칭 대기 위치로 이동 중";
    const accepted = await sendRobotCommand("training_start", currentTrainingPayload());
    if (!accepted) throw new Error("로봇 훈련 준비 명령 전송에 실패했습니다.");
    speak("정면 관절 인식이 완료됐습니다. 로봇을 펀칭 대기 위치로 이동합니다.");

    // Leave the alignment page as soon as training_start is accepted. Robot
    // calibration continues on the training page, while countdown/timer still
    // wait for the authoritative TRAINING_READY state.
    resetTrainingState();
    showScreen("training");
    updateMetrics();
    if (state.vision.connected && preferredVisionFeed()) useVisionPreview("training");
    if ($("#liveStatus")) $("#liveStatus").textContent = "로봇 훈련 준비 중";
    if ($("#liveCommand")) $("#liveCommand").textContent = "리치·미트 위치 보정을 준비합니다";
    await waitForRobotTrainingReady();
  } catch (error) {
    showToast(error.message, "error");
    speak(error.message);
    showScreen("ready");
    const button = $("#prepareTrainingButton");
    if (button) {
      button.disabled = false;
      button.textContent = "다시 확인";
    }
    return;
  }
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
    evidenceStartVersion: 0,
    forceStartVersion: state.force.version || 0,
    comboStepIndex: 0,
    comboRounds: 0,
    report: null,
  };
}

async function startTraining() {
  resetTrainingState();
  await checkForceStatus();
  state.training.evidenceStartVersion = state.vision.evidenceVersion;
  state.training.forceStartVersion = state.force.version || 0;
  state.training.running = true;
  await sendRobotCommand("training_go", currentTrainingPayload());
  await ensureVoiceSession(Math.max(180, state.trainingConfig.durationSec + 120));
  showScreen("training");
  await checkVisionStatus();
  state.training.useVision = Boolean(state.vision.connected && preferredVisionFeed());
  updateVisionTargetZone(state.vision.liveStatus);
  updateMetrics();
  renderCombinationStrip();

  if (state.training.useVision) {
    useVisionPreview("training");
    $("#liveStatus").textContent = "실시간 BASE 주먹 추적 연결됨";
    $("#visionFeedbackTitle").textContent = "가드 자세를 잡아주세요";
    $("#visionFeedbackText").textContent = "타격이 확정되면 신뢰도와 BASE 주먹 좌표, 저장 이미지가 표시됩니다.";
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

  // 카메라 정렬 완료 후 training_start에서 위빙 정지/펀칭 대기 자세 복귀를 완료했다.
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
  state.training.promptId = setTimeout(async () => {
    if (state.training.paused) return schedulePrompt(500);
    // Never cue a punch while the mitt is rebounding or moving to the next
    // combination pose. SessionBridge/HitAnalyzer owns the authoritative gate.
    if (state.robot?.connected) {
      await checkRobotStatus();
      if (String(state.robot?.state || "") !== "WAITING_FOR_HIT") {
        return schedulePrompt(180);
      }
    }
    state.training.prompts += 1;
    state.training.promptStartedAt = performance.now();
    let handText = state.trainingConfig.hand === "left" ? "왼손" : "오른손";
    let typeText = trainingTypeLabel(state.trainingConfig.type);
    if (state.trainingConfig.mode === "combination" && state.trainingConfig.sequence.length) {
      const step = state.trainingConfig.sequence[state.training.comboStepIndex % state.trainingConfig.sequence.length];
      handText = step.hand === "left" ? "왼손" : "오른손";
      typeText = trainingTypeLabel(step.punch);
    }
    $("#liveCommand").textContent = `${handText} ${typeText}!`;
    if (state.training.prompts % 3 === 1 || state.trainingConfig.mode === "combination") speak(`${handText} ${typeText}`);
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
  if (success) advanceCombinationStep();
  updateMetrics();
}

function prepareImpactAudio() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  if (!state.impactAudioContext) {
    state.impactAudioContext = new AudioContextClass();
  }
  if (state.impactAudioContext.state === "suspended") {
    state.impactAudioContext.resume().catch(() => {});
  }
  return state.impactAudioContext;
}

function playImpactSound() {
  const context = prepareImpactAudio();
  if (!context || context.state !== "running") return;
  const now = context.currentTime + .004;
  const master = context.createGain();
  master.gain.setValueAtTime(.0001, now);
  master.gain.exponentialRampToValueAtTime(.055, now + .008);
  master.gain.exponentialRampToValueAtTime(.0001, now + .14);
  master.connect(context.destination);

  const thump = context.createOscillator();
  thump.type = "sine";
  thump.frequency.setValueAtTime(165, now);
  thump.frequency.exponentialRampToValueAtTime(105, now + .13);
  thump.connect(master);
  thump.start(now);
  thump.stop(now + .145);

  const click = context.createOscillator();
  const clickGain = context.createGain();
  click.type = "triangle";
  click.frequency.setValueAtTime(520, now);
  click.frequency.exponentialRampToValueAtTime(310, now + .055);
  clickGain.gain.setValueAtTime(.24, now);
  clickGain.gain.exponentialRampToValueAtTime(.0001, now + .06);
  click.connect(clickGain);
  clickGain.connect(master);
  click.start(now);
  click.stop(now + .065);
}

function pulseTarget(success) {
  if (success) playImpactSound();
  const target = $("#visionTargetZone");
  if (!target || typeof target.animate !== "function") return;
  target.getAnimations().forEach((animation) => animation.cancel());
  const frames = success ? [
    { transform: "scale(1)", opacity: 1, filter: "brightness(1)", boxShadow: "0 0 0 15px rgba(105,205,239,.16), 0 0 24px rgba(105,205,239,.28)" },
    { transform: "scale(1.52)", opacity: 1, filter: "brightness(2.15)", boxShadow: "0 0 0 34px rgba(72,255,140,.31), 0 0 60px rgba(72,255,140,.82)", offset: .42 },
    { transform: "scale(1.52)", opacity: 1, filter: "brightness(1.9)", boxShadow: "0 0 0 32px rgba(72,255,140,.28), 0 0 56px rgba(72,255,140,.74)", offset: .58 },
    { transform: "scale(.96)", opacity: 1, filter: "brightness(1.2)", boxShadow: "0 0 0 21px rgba(72,255,140,.21), 0 0 36px rgba(72,255,140,.46)", offset: .80 },
    { transform: "scale(1)", opacity: 1, filter: "brightness(1)", boxShadow: "0 0 0 15px rgba(105,205,239,.16), 0 0 24px rgba(105,205,239,.28)" },
  ] : [
    { transform: "scale(1)", opacity: 1 },
    { transform: "scale(1.16)", opacity: .9 },
    { transform: "scale(1)", opacity: 1 },
  ];
  target.animate(frames, {
    duration: success ? 620 : 260,
    easing: success ? "cubic-bezier(.18,.8,.25,1)" : "ease-out",
  });
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

async function waitForRobotTrainingPaused(timeoutMs = 6000) {
  if (!state.robot?.connected) return true;
  const startedAt = Date.now();
  const failureTokens = ["PAUSE_FAILED", "PAUSE_REJECTED", "ERROR", "EMERGENCY_STOP"];
  while (Date.now() - startedAt < timeoutMs) {
    await checkRobotStatus();
    const robotState = String(state.robot?.state || "");
    if (robotState === "TRAINING_PAUSED") return true;
    if (failureTokens.some((token) => robotState.includes(token))) {
      throw new Error(state.robot?.error_detail || state.robot?.message || `일시정지 실패: ${robotState}`);
    }
    await sleep(120);
  }
  throw new Error("로봇 타격 세션 일시정지를 확인하지 못했습니다.");
}

async function togglePause() {
  if (!state.training.running) return;
  const button = $("#pauseButton");
  if (!state.training.paused) {
    // Freeze the UI timer immediately, then wait for SessionBridge to release
    // compliance/StopHitTest so no punches are counted during the pause.
    state.training.paused = true;
    button.textContent = "일시정지 중…";
    $("#liveCommand").textContent = "일시정지 준비 중";
    const accepted = await sendRobotCommand("pause");
    try {
      if (!accepted) throw new Error("로봇 일시정지 명령 전송에 실패했습니다.");
      await waitForRobotTrainingPaused();
      button.textContent = "다시 시작";
      $("#liveCommand").textContent = "일시정지";
      speak("훈련을 잠시 멈춥니다.");
    } catch (error) {
      state.training.paused = false;
      button.textContent = "일시정지";
      $("#liveCommand").textContent = "훈련 계속";
      showToast(error.message, "error");
      speak(error.message);
    }
    return;
  }

  // Keep the local timer paused until compliance stabilization has completed
  // and the robot reports WAITING_FOR_HIT again.
  button.textContent = "재개 준비 중…";
  $("#liveCommand").textContent = "로봇 타격 준비 중";
  const accepted = await sendRobotCommand("resume");
  try {
    if (!accepted) throw new Error("로봇 재개 명령 전송에 실패했습니다.");
    await waitForRobotTrainingReady();
    state.training.paused = false;
    button.textContent = "일시정지";
    $("#liveCommand").textContent = "훈련 재개";
    speak("훈련을 다시 시작합니다.");
  } catch (error) {
    state.training.paused = true;
    button.textContent = "다시 시작";
    $("#liveCommand").textContent = "일시정지";
    showToast(error.message, "error");
    speak(error.message);
  }
}

async function finishTraining() {
  if (!state.training.running) return;
  state.training.running = false;
  clearInterval(state.training.timerId);
  clearTimeout(state.training.promptId);
  stopPoseLoop();
  await sendRobotCommand("training_end", currentTrainingPayload());
  // Wait until SessionBridge has stopped the force session before freezing the
  // result window; otherwise the last HitResult can arrive just after sessionize.
  await waitForRobotTrainingStop();

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
  $("#resultCoachLabel").textContent = "KO COACH";
  $("#resultFeedback").textContent = feedback;
  $("#resultEvidenceDuo")?.classList.add("hidden");
  $("#resultProgressCard")?.classList.add("hidden");
  $("#resultReportDetails")?.classList.add("hidden");
  state.training.report = { coach_message: feedback, strengths: [], improvements: [], force_analysis: "", next_training: null };
  showScreen("result");
  speak(`훈련이 끝났습니다. 총 ${t.punches}회 펀치를 기록했습니다.`);

  if (state.currentUser) {
    try {
      let reportAlreadySaved = false;
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
          client_session_id: state.trainingConfig.clientSessionId,
        }),
      });

      if (t.useVision && savedSession?.id) {
        try {
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
              representative_images: [],
              punch_events: t.punchEvents,
              violation_counts: t.violationCounts,
            }),
          });
        } catch (visionSaveError) {
          console.warn("vision result save failed", visionSaveError);
        }

        let phase2Summary = null;
        try {
          phase2Summary = await api("/api/punch-events/sessionize", {
            method: "POST",
            body: JSON.stringify({
              session_id: savedSession.id,
              after_evidence_version: t.evidenceStartVersion,
              after_force_version: t.forceStartVersion || 0,
              punch_events: t.punchEvents,
            }),
          });
          renderEvidencePair(phase2Summary);
        } catch (eventError) {
          console.warn("punch event sessionize failed", eventError);
        }

        $("#resultCoachLabel").textContent = "KO COACH · OPENAI 분석 중";
        $("#resultFeedback").textContent = "이번 훈련에서 저장한 대표 타격 사진을 분석하고 있습니다…";
        try {
          const coaching = await api("/api/ai/vision-coach", {
            method: "POST",
            body: JSON.stringify({
              session_id: savedSession.id,
              after_evidence_version: t.evidenceStartVersion,
              expected_image_count: t.punches,
              fallback_feedback: feedback,
              metrics: {
                configured_training_type: state.trainingConfig.type,
                configured_hand: state.trainingConfig.hand,
                training_payload: currentTrainingPayload(),
                score,
                punch_count: t.punches,
                successful_punches: t.successful,
                success_rate: successRate,
                average_reaction_ms: avgReaction,
              },
            }),
          });
          feedback = coaching.coach_message || feedback;
          $("#resultFeedback").textContent = feedback;
          renderProgress(coaching.progress, coaching.progress_message || "");
          if (coaching.best || coaching.worst) renderEvidencePair({ best: coaching.best, worst: coaching.worst });
          if (coaching.best_punch_comment) $("#resultBestText").textContent = coaching.best_punch_comment;
          if (coaching.check_point_comment) $("#resultWorstText").textContent = coaching.check_point_comment;
          state.training.report = coaching;
          renderReportDetails(coaching);
          if (coaching.used_ai) {
            reportAlreadySaved = true;
            $("#resultCoachLabel").textContent = `KO COACH · OPENAI VISION · ${coaching.image_count}장`;
          } else if (coaching.reason === "api_key_missing") {
            $("#resultCoachLabel").textContent = "KO COACH · API KEY 필요";
          } else if (coaching.reason === "no_session_images") {
            $("#resultCoachLabel").textContent = "KO COACH · 분석 이미지 없음";
          } else {
            $("#resultCoachLabel").textContent = "KO COACH · 로컬 피드백";
          }
        } catch (coachError) {
          console.warn("OpenAI vision coaching failed", coachError);
          $("#resultCoachLabel").textContent = "KO COACH · 로컬 피드백";
          $("#resultFeedback").textContent = feedback;
          state.training.report = { coach_message: feedback, strengths: [], improvements: [], force_analysis: "", next_training: null };
        }
      } else if (savedSession?.id) {
        // Force results are useful even if the vision node is temporarily unavailable.
        // Persist them against the same client_session_id instead of dropping them.
        try {
          await api("/api/punch-events/sessionize", {
            method: "POST",
            body: JSON.stringify({
              session_id: savedSession.id,
              after_evidence_version: t.evidenceStartVersion,
              after_force_version: t.forceStartVersion || 0,
              punch_events: [],
            }),
          });
        } catch (forcePersistError) {
          console.warn("force-only sessionize failed", forcePersistError);
        }
      }
      if (savedSession?.id && !reportAlreadySaved) {
        const localReport = state.training.report || {};
        try {
          await api("/api/ai/reports", {
            method: "POST",
            body: JSON.stringify({
              session_id: savedSession.id,
              summary: localReport.headline || feedback,
              strengths: Array.isArray(localReport.strengths) ? localReport.strengths : [],
              improvements: Array.isArray(localReport.improvements) ? localReport.improvements : [],
              next_training: localReport.next_training || "",
              coach_message: localReport.coach_message || feedback,
              model: localReport.model || "local-rules",
              headline: localReport.headline || "이번 훈련 코칭 요약",
              force_analysis: localReport.force_analysis || "",
              progress_message: localReport.progress_message || "",
              progress: localReport.progress || null,
            }),
          });
        } catch (reportError) {
          console.warn("fallback report save failed", reportError);
        }
      }
      await loadUsers();
      await checkDatabaseStatus();
    } catch (error) {
      showToast("훈련 기록 저장에 실패했습니다.", "error");
    }
  }
}

async function sendRobotCommand(command, payload = {}) {
  try {
    return await api("/api/robot/command", { method: "POST", body: JSON.stringify({ command, payload }) });
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
    if ($("#databaseAdminStatusText")) $("#databaseAdminStatusText").textContent = status.ok ? `${status.users || 0} users · ${status.sessions || 0} sessions` : "확인 필요";
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
    if ($("#sttAdminStatusText")) $("#sttAdminStatusText").textContent = state.sttConfigured ? String(status.model || "READY") : "API key 없음";
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
  else hint.textContent = `대기 중 · “${wake.display_name || "웨이크 업 케이오"}”라고 부른 뒤 명령 1개를 말해 주세요`;
  if ($("#wakeAdminStatusText")) $("#wakeAdminStatusText").textContent = wakeReady ? `${wake.display_name || "웨이크 업 케이오"} 대기` : (wake.message || "확인 필요");
  updateContextVoiceHelp(state.currentScreen);
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
    const previous = ({ register: "profiles", measure: "dashboard", "measure-result": "measure", setup: "dashboard", history: "dashboard", "report-detail": "history", ready: "setup", settings: "home" })[state.currentScreen] || "home";
    showScreen(previous); speak("이전 화면으로 이동합니다."); closeVoicePanelSoon(); return;
  }
  if (text.includes("시스템설정") || text.includes("연결상태")) {
    if (state.appMode === "admin") {
      showScreen("settings"); speak("관리자 시스템 상태를 보여드릴게요.");
    } else {
      speak("시스템 설정은 관리자 모드에서만 사용할 수 있습니다.");
    }
    closeVoicePanelSoon(); return;
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
    const nameMatch = rawText.match(/(?:이름(?:은|이)?|나는)\s*([가-힣A-Za-z0-9]{1,20}?)(?=\s*(?:이고|이고요|키|신장|,|주\s*사용\s*손|주손|오른손|왼손|$))/);
    const cm = parseCentimeters(rawText);
    const rightHand = text.includes("오른손잡이") || text.includes("주사용손은오른손") || text.includes("주손은오른손") || text === "오른손";
    const leftHand = text.includes("왼손잡이") || text.includes("주사용손은왼손") || text.includes("주손은왼손") || text === "왼손";
    const filled = [];
    if (nameMatch) { $("#userName").value = nameMatch[1]; filled.push(`이름 ${nameMatch[1]}`); }
    if ((text.includes("키") || text.includes("신장")) && cm) { $("#userHeight").value = cm; filled.push(`키 ${cm}센티미터`); }
    if (rightHand) { $('#registerForm input[name="dominant_hand"][value="right"]').checked = true; filled.push("주 사용 손 오른손"); }
    else if (leftHand) { $('#registerForm input[name="dominant_hand"][value="left"]').checked = true; filled.push("주 사용 손 왼손"); }
    if (filled.length) { speak(`${filled.join(", ")}으로 입력했습니다.`); closeVoicePanelSoon(); return; }
    if (text.includes("등록완료") || text.includes("저장하고측정") || text === "저장해") { $("#registerForm").requestSubmit(); closeVoicePanelSoon(); return; }
  }

  if (text.includes("리치측정시작") || text.includes("카메라시작") || text.includes("리치재측정")) {
    voiceStartReachMeasurement(); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "measure") {
    if (text.includes("측정시작") || text.includes("현재단계측정") || text === "측정해") { beginMeasurementStage(false); speak("현재 단계를 다시 측정합니다. 자세를 유지해 주세요."); closeVoicePanelSoon(); return; }
    if (text.includes("직접입력")) { openModal("manualMeasureModal"); speak("리치 값을 직접 입력해 주세요."); closeVoicePanelSoon(); return; }
  }
  if (state.currentScreen === "measure-result") {
    if (text.includes("다시측정") || text.includes("다시해")) { resetMeasurement(); showScreen("measure"); voiceStartReachMeasurement(); closeVoicePanelSoon(); return; }
    if (text.includes("저장") || text.includes("완료")) { saveMeasurement(); closeVoicePanelSoon(); return; }
  }

  if (state.currentScreen === "result" && (text.includes("결과읽어") || text.includes("리포트읽어") || text.includes("보고서읽어") || text.includes("코칭읽어"))) {
    readCurrentReport(); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "report-detail" && (text.includes("결과읽어") || text.includes("리포트읽어") || text.includes("보고서읽어") || text.includes("코칭읽어"))) {
    readSavedReport(); closeVoicePanelSoon(); return;
  }
  if (state.currentScreen === "result" && (text.includes("다시훈련") || text.includes("추천훈련시작"))) {
    goToTrainingReady(); closeVoicePanelSoon(); return;
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

  const comboAliases = [
    { id: 2, words: ["잽잽스트레이트"] },
    { id: 3, words: ["원투훅"] },
    { id: 4, words: ["원투원투"] },
    { id: 5, words: ["원투어퍼", "원투어퍼컷"] },
    { id: 1, words: ["원투"] },
  ];
  const comboNumberMatch = text.match(/콤비네이션([1-5])/);
  const aliasCombo = comboAliases.find((item) => item.words.some((word) => text.includes(word)));
  const comboId = comboNumberMatch ? Number(comboNumberMatch[1]) : aliasCombo?.id;
  if (comboId && COMBINATIONS[comboId]) {
    if (!state.currentUser) { speak("먼저 사용자를 선택해 주세요."); showScreen("profiles"); closeVoicePanelSoon(); return; }
    const durationSec = parseDuration(rawText) || 60;
    configureTraining({ hand: "both", type: `combination_${comboId}`, durationSec, combinationId: comboId });
    speak(`콤비네이션 ${comboId}, ${COMBINATIONS[comboId].name} ${durationSec}초 훈련을 준비합니다.`);
    closeVoicePanelSoon(); goToTrainingReady(); return;
  }

  const wantsTraining = text.includes("운동시작") || text.includes("훈련시작") || text.includes("연습할게") || text.includes("스트레이트") || text.includes("잽") || text.includes("훅") || text.includes("어퍼");
  if (wantsTraining) {
    if (!state.currentUser) { speak("먼저 사용자를 선택해 주세요."); showScreen("profiles"); closeVoicePanelSoon(); return; }
    const hand = text.includes("왼손") ? "left" : text.includes("양손") ? "both" : text.includes("오른손") ? "right" : state.currentUser.dominant_hand || "right";
    const type = text.includes("어퍼") ? "uppercut" : text.includes("훅") ? "hook" : text.includes("잽") ? "jab" : "straight";
    const durationSec = parseDuration(rawText) || 60;
    configureTraining({ hand, type, durationSec, combinationId: null });
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
  document.addEventListener("pointerdown", prepareImpactAudio, { once: true, capture: true });
  document.addEventListener("keydown", prepareImpactAudio, { once: true, capture: true });
  $$('[data-screen]').forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.screen === "history" && state.currentUser) loadHistory();
    else if (button.dataset.screen === "history") {
      showToast("먼저 사용자를 선택해 주세요.");
      showScreen("profiles");
    }
    else showScreen(button.dataset.screen);
  }));
  $("#registerForm").addEventListener("submit", registerUser);
  $("#trainingSetupForm").addEventListener("submit", submitTrainingSetup);
  $("#openTrainingSetupButton").addEventListener("click", openTrainingSetup);
  $("#openHistoryButton").addEventListener("click", loadHistory);
  $("#historyStartButton").addEventListener("click", openTrainingSetup);
  $("#savedReportBackButton").addEventListener("click", loadHistory);
  $("#savedReportCloseButton").addEventListener("click", loadHistory);
  $("#readSavedReportButton").addEventListener("click", readSavedReport);
  $("#remeasureButton").addEventListener("click", async () => { resetMeasurement(); showScreen("measure"); await startMeasurementCamera(); });
  $("#cameraStartButton").addEventListener("click", startMeasurementCamera);
  $("#captureMeasureButton")?.addEventListener("click", () => beginMeasurementStage(false));
  $("#manualMeasureButton").addEventListener("click", () => openModal("manualMeasureModal"));
  $("#manualMeasureForm").addEventListener("submit", applyManualMeasurement);
  $("#finishProfileButton").addEventListener("click", saveMeasurement);
  $("#measureAgainButton")?.addEventListener("click", async () => { resetMeasurement(); showScreen("measure"); await startMeasurementCamera(); });
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
  await loadAppConfig();
  wireEvents();
  await checkSttStatus();
  await checkDatabaseStatus();
  await checkWakeWordStatus();
  startWakeEventPolling();
  startVisionPolling();
  startAdminStatusPolling();
  await loadUsers();
  renderProfiles();
  updateTrainingLabels();
  updateContextVoiceHelp(state.currentScreen);
}

init();
