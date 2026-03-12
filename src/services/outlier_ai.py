from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.utils.api_keys import run_with_provider_keys


@dataclass(frozen=True)
class InsightCard:
    title: str
    body: str
    support: str = ""


@dataclass(frozen=True)
class OutlierAIReport:
    provider: str
    model: str
    executive_headline: str
    key_takeaway: str
    confidence_label: str
    breakout_themes: Tuple[InsightCard, ...]
    title_patterns: Tuple[InsightCard, ...]
    repeatable_angles: Tuple[InsightCard, ...]
    next_steps: Tuple[str, ...]
    warnings: Tuple[str, ...]
    raw_fallback: str = ""


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    candidate = str(text or "").strip()
    if not candidate:
        return None

    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _card_tuple(payload: Any) -> Tuple[InsightCard, ...]:
    cards: List[InsightCard] = []
    if not isinstance(payload, list):
        return tuple(cards)
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        title = _safe_text(item.get("title"))
        body = _safe_text(item.get("body"))
        if not title or not body:
            continue
        cards.append(
            InsightCard(
                title=title,
                body=body,
                support=_safe_text(item.get("support")),
            )
        )
    return tuple(cards)


def _string_tuple(payload: Any) -> Tuple[str, ...]:
    if not isinstance(payload, list):
        return tuple()
    return tuple(_safe_text(item) for item in payload if _safe_text(item))


def _fallback_report(provider: str, model: str, raw_text: str) -> OutlierAIReport:
    clipped = _safe_text(raw_text).replace("\n", " ")
    if len(clipped) > 420:
        clipped = clipped[:417].rstrip() + "..."
    return OutlierAIReport(
        provider=provider,
        model=model,
        executive_headline="AI Research Available in Fallback Mode",
        key_takeaway=clipped or "The provider returned an unstructured answer, so the report is being shown in simplified form.",
        confidence_label="Low",
        breakout_themes=tuple(),
        title_patterns=tuple(),
        repeatable_angles=tuple(),
        next_steps=tuple(),
        warnings=("Structured parsing failed; review the fallback summary before acting on it.",),
        raw_fallback=raw_text,
    )


def _build_prompt(
    query_context: Mapping[str, Any],
    summary_stats: Mapping[str, Any],
    top_rows: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "You are a YouTube strategist analyzing public outlier-video research.\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "executive_headline": "short string",\n'
        '  "key_takeaway": "one concise sentence",\n'
        '  "confidence_label": "High|Medium|Low",\n'
        '  "breakout_themes": [{"title":"", "body":"", "support":""}],\n'
        '  "title_patterns": [{"title":"", "body":"", "support":""}],\n'
        '  "repeatable_angles": [{"title":"", "body":"", "support":""}],\n'
        '  "next_steps": ["", "", ""],\n'
        '  "warnings": ["", ""]\n'
        "}\n\n"
        "Rules:\n"
        "- Keep each body under 35 words.\n"
        "- Use only evidence available in the payload.\n"
        "- Mention caveats if confidence is limited.\n"
        "- Do not mention hidden internal reasoning.\n\n"
        f"Query Context: {json.dumps(dict(query_context), ensure_ascii=True)}\n"
        f"Summary Stats: {json.dumps(dict(summary_stats), ensure_ascii=True)}\n"
        f"Top Outliers: {json.dumps(list(top_rows), ensure_ascii=True)}"
    )


def _gemini_generate_text(api_key: str, model: str, prompt: str) -> str:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    response = requests.post(
        endpoint,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text[:500]}")
    body = response.json()
    return (
        body.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )


def _openai_generate_text(api_key: str, model: str, prompt: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce concise structured research outputs."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API error ({response.status_code}): {response.text[:500]}")
    body = response.json()
    return (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def _report_from_payload(provider: str, model: str, payload: Mapping[str, Any], raw_text: str) -> OutlierAIReport:
    return OutlierAIReport(
        provider=provider,
        model=model,
        executive_headline=_safe_text(payload.get("executive_headline"), "What Is Breaking Out Right Now"),
        key_takeaway=_safe_text(payload.get("key_takeaway"), "The scanned cohort shows a few packaging and format patterns that are outperforming their peer set."),
        confidence_label=_safe_text(payload.get("confidence_label"), "Medium"),
        breakout_themes=_card_tuple(payload.get("breakout_themes")),
        title_patterns=_card_tuple(payload.get("title_patterns")),
        repeatable_angles=_card_tuple(payload.get("repeatable_angles")),
        next_steps=_string_tuple(payload.get("next_steps")),
        warnings=_string_tuple(payload.get("warnings")),
        raw_fallback=raw_text if not payload else "",
    )


def generate_outlier_ai_report(
    provider: str,
    model: str,
    query_context: Mapping[str, Any],
    summary_stats: Mapping[str, Any],
    result_frame: pd.DataFrame,
) -> OutlierAIReport:
    provider_name = provider.lower().strip()
    top_rows = result_frame.head(12)[
        [
            "video_title",
            "channel_title",
            "outlier_score",
            "views",
            "views_per_day",
            "engagement_rate",
            "duration_bucket",
            "title_pattern",
            "language_confidence_label",
            "why_outlier",
            "research_cue",
        ]
    ].to_dict(orient="records")
    prompt = _build_prompt(query_context=query_context, summary_stats=summary_stats, top_rows=top_rows)

    if provider_name == "gemini":
        raw_text = run_with_provider_keys(
            "gemini",
            lambda key: _gemini_generate_text(key, model, prompt),
        )
    elif provider_name == "openai":
        raw_text = run_with_provider_keys(
            "openai",
            lambda key: _openai_generate_text(key, model, prompt),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    parsed = _extract_json_block(raw_text)
    if not parsed:
        return _fallback_report(provider_name, model, raw_text)
    return _report_from_payload(provider_name, model, parsed, raw_text)
