from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# OpenAI official image/Responses examples currently use GPT-5.6.
# Keep an environment override so deployment can pin another supported vision model.
DEFAULT_VISION_COACH_MODEL = "gpt-5.6"

ENGLAND_BOXING_LEVEL1_REFERENCE = {
    "title": "England Boxing Level 1 Coaching Handbook",
    "publisher": "England Boxing",
    "url": "https://www.englandboxing.org/wp-content/uploads/2022/03/EB_Boxing-Coaching-Handbook-Part-1_v8-002.pdf",
}

# Only the handbook principles that can be grounded in the KO system's current
# jab/straight measurements are supplied to the model.  The handbook must not
# override the measured evidence or invite unsupported inference from a still.
ENGLAND_BOXING_LEVEL1_COACHING_GUIDE = (
    "England Boxing Level 1 Coaching Handbook 기준을 참고한다. "
    "현재 잽/스트레이트 코칭에서는 다음 원칙만 측정 근거가 있을 때 적용한다: "
    "좋은 균형과 안정된 스탠스, 타깃을 보호하는 가드와 빠른 가드 복귀, "
    "직선 펀치는 올바른 스탠스/가드에서 시작하고 끝나며 회전·팔 신전·회수를 갖춘다. "
    "피드백은 충분히 관찰·분석한 뒤 1~2개의 핵심 포인트로 제한하고, 구체적이고 단순하며 긍정적으로 제시한다. "
    "핸드북의 일반 원칙보다 제공된 실제 측정값과 이미지 증거를 우선하며, 측정되지 않은 발동작·충격·회수 속도는 추정하지 않는다."
)


class VisionCoachError(RuntimeError):
    pass


def select_representative_images(
    images: Sequence[tuple[int, bytes, str]],
    limit: int = 3,
) -> list[tuple[int, bytes, str]]:
    """Select evenly spaced evidence frames from one training session."""
    items = list(images)
    limit = max(1, int(limit))
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def _output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = str(content.get("text", "")).strip()
                if text:
                    return text
    raise VisionCoachError("OpenAI 응답에서 코칭 문구를 찾지 못했습니다.")


def analyze_boxing_images(
    images: Sequence[tuple[int, bytes, str]],
    metrics: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Create a grounded coaching report from selected evidence images + measured metrics.

    Phase 2 uses B-mode evidence: when available, image 1 is BEST PUNCH and image 2 is
    CHECK POINT (lowest deterministic event score). Progress status is computed locally
    and the model is explicitly instructed not to override that judgment.
    """
    key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        raise VisionCoachError("OpenAI API 키가 설정되지 않았습니다.")
    if not images:
        raise VisionCoachError("이번 훈련에서 분석할 타격 이미지가 없습니다.")

    model = os.environ.get("OPENAI_VISION_COACH_MODEL", DEFAULT_VISION_COACH_MODEL).strip() or DEFAULT_VISION_COACH_MODEL
    detail = os.environ.get("OPENAI_VISION_COACH_DETAIL", "auto").strip().lower()
    if detail not in {"low", "high", "auto"}:
        detail = "auto"

    system_prompt = (
        "당신은 KO AI 복싱 코치입니다. 제공된 수치와 정지영상에 근거해서만 코칭하세요. "
        "progress.comparisons와 progress.tracked_result의 improved/maintained/declined 판정은 로컬 알고리즘의 결정이므로 "
        "절대로 뒤집거나 새로 추측하지 마세요. 이전 피드백이 실제 측정 항목으로 추적되었다면 그 발전 여부를 먼저 설명하세요. "
        "각 이미지는 LEFT/FRONT/RIGHT 3시점 합성 JPEG이며 초록 선은 관절 추정입니다. image_roles가 best_punch면 가장 평가가 좋은 타격, "
        "check_point면 가장 낮은 이벤트 점수의 타격입니다. 사진에서 직접 보이는 팔 신전, 반대손 가드, 어깨·상체 정렬만 시각적으로 설명하세요. "
        "정지사진만으로 타격 세기, 충격량, 접촉 성공 여부, 전체 궤적, 가드 복귀 속도를 추측하지 마세요. 힘 관련 설명은 force_summary나 force 필드가 "
        "실제로 있을 때만 하세요. power_score나 safety_stop처럼 아직 미완성/고정값인 필드는 근거로 사용하지 마세요. "
        "측정되지 않은 사실을 만들지 말고, 사용자가 다음 훈련에서 신경 쓸 내용은 측정 가능한 metric 중 하나를 선택하세요. "
        "보고서는 칭찬만 하지 말고 실제 개선 여부, 현재 강점, 가장 중요한 개선점, 다음 훈련 목표가 서로 연결되도록 작성하세요. "
        "콤비네이션 훈련이면 training_type과 sequence에 있는 순서만 사용하고 실제로 어떤 펀치를 잘못 수행했는지는 측정 데이터가 없으면 단정하지 마세요. "
        + ENGLAND_BOXING_LEVEL1_COACHING_GUIDE
    )

    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "훈련 데이터:\n" + json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
        }
    ]
    roles = metrics.get("image_roles") if isinstance(metrics.get("image_roles"), list) else []
    for index, (_version, data, content_type) in enumerate(images):
        role = roles[index] if index < len(roles) and isinstance(roles[index], dict) else {"role": "representative"}
        user_content.append({
            "type": "input_text",
            "text": "다음 이미지 역할: " + json.dumps(role, ensure_ascii=False, separators=(",", ":")),
        })
        mime = "image/png" if "png" in content_type.lower() else "image/jpeg"
        encoded = base64.b64encode(data).decode("ascii")
        user_content.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{encoded}",
            "detail": detail,
        })

    metric_enum = [
        "accuracy_percent", "average_reaction_sec", "guard_error_rate", "guard_score",
        "arm_extension_score", "torso_balance_score", "average_guard_return_sec",
        "force_accuracy_score", "peak_force_n", "center_error_mm", "none",
    ]
    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "coach_message": {"type": "string"},
            "progress_message": {"type": "string"},
            "observed_strength": {"type": "string"},
            "improvement": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "improvements": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "force_analysis": {"type": "string"},
            "next_focus": {"type": "string"},
            "next_focus_metric": {"type": "string", "enum": metric_enum},
            "next_training": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "duration_sec": {"type": "integer", "minimum": 30, "maximum": 600},
                    "goal": {"type": "string"}
                },
                "required": ["title", "duration_sec", "goal"],
                "additionalProperties": False
            },
            "best_punch_comment": {"type": "string"},
            "check_point_comment": {"type": "string"},
            "visual_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": [
            "headline", "coach_message", "progress_message", "observed_strength", "improvement",
            "strengths", "improvements", "force_analysis", "next_focus", "next_focus_metric", "next_training",
            "best_punch_comment", "check_point_comment", "visual_confidence",
        ],
        "additionalProperties": False,
    }
    request_payload = {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ko_boxing_progress_report",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 1200,
    }
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = Request(
        f"{base_url}/responses",
        data=json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "KO-Boxing-Progress-Coach/All-In-One",
        },
    )
    try:
        with urlopen(request, timeout=75) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise VisionCoachError(f"OpenAI 이미지 분석 요청이 거부되었습니다 (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError) as exc:
        raise VisionCoachError("OpenAI 이미지 분석 서버에 연결하지 못했습니다.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionCoachError("OpenAI 이미지 분석 응답을 해석하지 못했습니다.") from exc

    try:
        result = json.loads(_output_text(response_payload))
    except json.JSONDecodeError as exc:
        raise VisionCoachError("OpenAI 코칭 결과가 JSON 형식이 아닙니다.") from exc
    if not isinstance(result, dict):
        raise VisionCoachError("OpenAI 코칭 결과 형식이 올바르지 않습니다.")
    for key_name in (
        "headline", "coach_message", "progress_message", "observed_strength", "improvement", "force_analysis", "next_focus",
        "best_punch_comment", "check_point_comment",
    ):
        result[key_name] = re.sub(r"\s+", " ", str(result.get(key_name, ""))).strip()[:1000]
    for array_key in ("strengths", "improvements"):
        values = result.get(array_key) if isinstance(result.get(array_key), list) else []
        result[array_key] = [re.sub(r"\s+", " ", str(value)).strip()[:500] for value in values if str(value).strip()][:3]
    next_training = result.get("next_training") if isinstance(result.get("next_training"), dict) else {}
    result["next_training"] = {
        "title": re.sub(r"\s+", " ", str(next_training.get("title", "다음 훈련"))).strip()[:200],
        "duration_sec": max(30, min(600, int(next_training.get("duration_sec", 60) or 60))),
        "goal": re.sub(r"\s+", " ", str(next_training.get("goal", result.get("next_focus", "")))).strip()[:500],
    }
    if not result["coach_message"]:
        raise VisionCoachError("OpenAI 코칭 한줄평이 비어 있습니다.")
    result["model"] = model
    result["image_count"] = len(images)
    result["coaching_reference"] = dict(ENGLAND_BOXING_LEVEL1_REFERENCE)
    return result
