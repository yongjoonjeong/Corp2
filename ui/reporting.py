from __future__ import annotations

import json
from typing import Any, Iterable

# Metric registry used for deterministic previous-vs-current comparisons.
# Higher/lower direction is decided here, not by the language model.
METRICS: dict[str, dict[str, str]] = {
    "accuracy_percent": {"label": "타격 정확도", "direction": "higher"},
    "average_reaction_sec": {"label": "평균 반응시간", "direction": "lower"},
    "guard_error_rate": {"label": "가드 오류율", "direction": "lower"},
    "guard_score": {"label": "가드 점수", "direction": "higher"},
    "arm_extension_score": {"label": "팔 신전 점수", "direction": "higher"},
    "torso_balance_score": {"label": "상체 밸런스 점수", "direction": "higher"},
    "average_guard_return_sec": {"label": "가드 복귀시간", "direction": "lower"},
    "force_accuracy_score": {"label": "미트 타격 정확도", "direction": "higher"},
    "peak_force_n": {"label": "최대 타격 힘", "direction": "higher"},
    "center_error_mm": {"label": "미트 중심 오차", "direction": "lower"},
}


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_metrics(session: dict[str, Any] | None, vision: dict[str, Any] | None, force_summary: dict[str, Any] | None = None) -> dict[str, float]:
    session = session or {}
    vision = vision or {}
    force_summary = force_summary or {}
    out: dict[str, float] = {}

    mapping = {
        "accuracy_percent": vision.get("accuracy_percent", session.get("success_rate")),
        "average_reaction_sec": vision.get("average_reaction_sec"),
        "guard_score": vision.get("guard_score"),
        "arm_extension_score": vision.get("arm_extension_score"),
        "torso_balance_score": vision.get("torso_balance_score", session.get("posture_score")),
        "average_guard_return_sec": vision.get("average_guard_return_sec"),
        "force_accuracy_score": force_summary.get("average_accuracy_score"),
        "peak_force_n": force_summary.get("peak_force_n"),
        "center_error_mm": force_summary.get("average_center_error_mm"),
    }
    for key, value in mapping.items():
        number = _number(value)
        if number is not None:
            out[key] = number

    punches = _number(vision.get("total_punches", session.get("punch_count")))
    guard_drops = _number(vision.get("guard_drop_count"))
    if punches and punches > 0 and guard_drops is not None:
        out["guard_error_rate"] = min(100.0, max(0.0, guard_drops / punches * 100.0))
    return out


def compare_metric(metric_key: str, previous: float | None, current: float | None) -> dict[str, Any] | None:
    definition = METRICS.get(metric_key)
    if definition is None or previous is None or current is None:
        return None
    previous = float(previous)
    current = float(current)
    delta = current - previous
    # A small neutral band prevents tiny sensor fluctuations being described as progress/regression.
    if metric_key.endswith("_sec"):
        neutral = 0.04
    elif metric_key in {"center_error_mm", "peak_force_n"}:
        neutral = 2.0
    else:
        neutral = 3.0
    effective = delta if definition["direction"] == "higher" else -delta
    status = "maintained" if abs(effective) < neutral else ("improved" if effective > 0 else "declined")
    return {
        "metric_key": metric_key,
        "metric_label": definition["label"],
        "direction": definition["direction"],
        "previous": round(previous, 3),
        "current": round(current, 3),
        "delta": round(delta, 3),
        "status": status,
    }


def build_progress(previous_metrics: dict[str, float], current_metrics: dict[str, float], tracked_metric: str | None = None) -> dict[str, Any]:
    comparisons = []
    for key in METRICS:
        compared = compare_metric(key, previous_metrics.get(key), current_metrics.get(key))
        if compared:
            comparisons.append(compared)
    tracked = next((item for item in comparisons if item["metric_key"] == tracked_metric), None)
    return {
        "has_previous": bool(comparisons),
        "tracked_metric": tracked_metric,
        "tracked_result": tracked,
        "comparisons": comparisons,
    }



def infer_tracked_metric(feedback_text: str | None, previous_metrics: dict[str, float], current_metrics: dict[str, float]) -> str | None:
    """Map legacy free-text feedback to a measurable metric without asking the model to infer progress.

    Only returns a key when that metric exists in both sessions. This lets Phase 2 keep using
    reports saved before structured feedback_goals were introduced.
    """
    text = str(feedback_text or "").replace(" ", "").lower()
    candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("가드복귀", "복귀속도", "손을되돌", "가드로복귀"), ("average_guard_return_sec", "guard_error_rate", "guard_score")),
        (("가드", "반대손", "손을높", "턱옆", "얼굴높이"), ("guard_error_rate", "guard_score")),
        (("반응", "타이밍"), ("average_reaction_sec",)),
        (("정확", "적중", "타깃"), ("accuracy_percent", "force_accuracy_score", "center_error_mm")),
        (("신전", "팔을끝까지", "팔을뻗"), ("arm_extension_score",)),
        (("상체", "균형", "밸런스", "쏠"), ("torso_balance_score",)),
        (("힘", "파워"), ("peak_force_n",)),
        (("중심", "방향", "치우"), ("center_error_mm", "force_accuracy_score")),
    ]
    for words, keys in candidates:
        if not any(word in text for word in words):
            continue
        for key in keys:
            if key in previous_metrics and key in current_metrics:
                return key
    return None

def event_score(event: dict[str, Any], force: dict[str, Any] | None = None) -> float:
    """Score one captured punch using only measurements already supplied by the system.

    Phase 2 deliberately does not invent pose quality from a still image. The realtime
    vision score is the base, violation tags apply explicit penalties, and verified
    force-analyzer accuracy can contribute when /mitt/hit_result is available.
    """
    base = _number(event.get("total_score"))
    if base is None:
        confidence = _number((event.get("quality") or {}).get("confidence"))
        base = (confidence * 100.0) if confidence is not None else 50.0
    score = max(0.0, min(100.0, base))
    violations = event.get("violations") or []
    if isinstance(violations, list):
        score -= min(35.0, 8.0 * len(violations))
    if event.get("passed") is False:
        score -= 15.0
    if force:
        accuracy = _number(force.get("accuracy_score"))
        valid_hit = force.get("valid_hit")
        if accuracy is not None and valid_hit is not False:
            accuracy = max(0.0, min(100.0, accuracy))
            score = score * 0.65 + accuracy * 0.35
        elif valid_hit is False:
            score -= 15.0
    return round(max(0.0, min(100.0, score)), 2)


def select_best_worst(events: Iterable[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    items = list(events)
    if not items:
        return None, None
    ordered = sorted(items, key=lambda item: (float(item.get("event_score", 0.0)), int(item.get("punch_index", 0))))
    worst = ordered[0]
    best = ordered[-1]
    return best, worst


def issue_tags(event: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for violation in event.get("violations") or []:
        if isinstance(violation, dict):
            code = str(violation.get("code", "")).strip()
            if code:
                tags.append(code)
    force = event.get("force") or {}
    if force.get("valid_hit") is False:
        tags.append("invalid_force_hit")
    return tags


def compact_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    return {
        "id": event.get("id"),
        "punch_index": event.get("punch_index"),
        "vision_punch_id": event.get("vision_punch_id"),
        "punch_side": event.get("punch_side"),
        "punch_type": event.get("punch_type"),
        "event_score": event.get("event_score"),
        "evidence_url": event.get("evidence_url"),
        "issue_tags": event.get("issue_tags", []),
        "force": event.get("force"),
    }


def json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []
