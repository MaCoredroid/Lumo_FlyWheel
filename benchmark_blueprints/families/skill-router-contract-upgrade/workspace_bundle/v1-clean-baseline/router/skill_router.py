from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "skill_router.toml"


@dataclass
class Skill:
    name: str
    trigger: str | None = None
    triggers: tuple[str, ...] = ()
    negative_triggers: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouterConfig:
    default_skill: str
    fallback_when_no_eligible: bool
    skills: tuple[Skill, ...]


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    matched_triggers: tuple[str, ...]


def _coerce_string_list(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        return (normalized,) if normalized else ()
    if isinstance(raw_value, list):
        return tuple(
            item.strip()
            for item in raw_value
            if isinstance(item, str) and item.strip()
        )
    return ()


def _parse_skill(raw_skill: Mapping[str, Any]) -> Skill:
    trigger = raw_skill.get("trigger")
    normalized_trigger = trigger.strip() if isinstance(trigger, str) and trigger.strip() else None
    return Skill(
        name=str(raw_skill["name"]),
        trigger=normalized_trigger,
        triggers=_coerce_string_list(raw_skill.get("triggers")),
        negative_triggers=_coerce_string_list(raw_skill.get("negative_triggers")),
        required_inputs=_coerce_string_list(raw_skill.get("required_inputs")),
    )


def _load_router_config() -> RouterConfig:
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    router_config = config["router"]
    return RouterConfig(
        default_skill=str(router_config["default_skill"]),
        fallback_when_no_eligible=bool(router_config.get("fallback_when_no_eligible", True)),
        skills=tuple(_parse_skill(raw_skill) for raw_skill in config.get("skills", [])),
    )


ROUTER_CONFIG = _load_router_config()
DEFAULT_SKILL = ROUTER_CONFIG.default_skill
SKILLS = ROUTER_CONFIG.skills


def _normalized_triggers(skill: Skill) -> tuple[str, ...]:
    triggers: list[str] = []
    if skill.trigger:
        triggers.append(skill.trigger)
    triggers.extend(skill.triggers)
    return tuple(trigger.lower() for trigger in triggers if trigger)


def _has_required_inputs(skill: Skill, provided_inputs: Mapping[str, Any]) -> bool:
    if not skill.required_inputs:
        return True

    for input_name in skill.required_inputs:
        value = provided_inputs.get(input_name)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if hasattr(value, "__len__") and not isinstance(value, str) and len(value) == 0:
            return False
    return True


def _is_suppressed(skill: Skill, lowered_request: str) -> bool:
    if not skill.negative_triggers:
        return False
    return any(trigger.lower() in lowered_request for trigger in skill.negative_triggers)


def _match_skill(skill: Skill, lowered_request: str, provided_inputs: dict[str, str]) -> SkillMatch | None:
    matched_triggers = tuple(
        trigger for trigger in _normalized_triggers(skill) if trigger in lowered_request
    )
    if not matched_triggers:
        return None
    if _is_suppressed(skill, lowered_request):
        return None
    if not _has_required_inputs(skill, provided_inputs):
        return None
    return SkillMatch(skill=skill, matched_triggers=matched_triggers)


def _match_rank(match: SkillMatch) -> tuple[int, int]:
    return (len(match.matched_triggers), max(len(trigger) for trigger in match.matched_triggers))


def route(request_text: str, provided_inputs: Mapping[str, Any]) -> str:
    lowered = request_text.lower()
    eligible_matches = [
        match
        for skill in SKILLS
        if (match := _match_skill(skill, lowered, provided_inputs)) is not None
    ]
    if not eligible_matches:
        if ROUTER_CONFIG.fallback_when_no_eligible:
            return DEFAULT_SKILL
        return ""

    return max(enumerate(eligible_matches), key=lambda item: (_match_rank(item[1]), -item[0]))[1].skill.name
