from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import math
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class SourceSegment:
    id: str
    start: int
    end: int
    source_text: str
    source_hash: str


@dataclass(frozen=True)
class SegmentBatch:
    segments: tuple[SourceSegment, ...]
    left_context: str
    right_context: str


@dataclass(frozen=True)
class FidelityResult:
    accepted: bool
    reason: str
    replaced: int = 0
    inserted: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class SegmentDecision:
    segment_id: str
    text: str
    status: str
    reason: str
    source_hash: str
    result_hash: str
    start: int
    end: int
    replaced: int = 0
    inserted: int = 0
    deleted: int = 0


_BOUNDARY_PATTERNS = (
    re.compile(r"\n\s*\n"),
    re.compile(r"[。！？.!?]+\s*"),
    re.compile(r"[，；：、,;:]\s*"),
    re.compile(r"\s+"),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _segment_id(index: int, start: int, end: int) -> str:
    return f"seg-{index:06d}-{start:08d}-{end:08d}"


def _inside_ascii_token(text: str, position: int) -> bool:
    if position <= 0 or position >= len(text):
        return False
    return (
        text[position - 1].isascii()
        and not text[position - 1].isspace()
        and text[position].isascii()
        and not text[position].isspace()
    )


def _safe_cut(text: str, start: int, target: int, minimum: int, maximum: int) -> int:
    low = min(len(text), start + minimum)
    high = min(len(text), start + maximum)
    if high >= len(text):
        return len(text)

    for pattern in _BOUNDARY_PATTERNS:
        positions = [
            match.end()
            for match in pattern.finditer(text, low, high)
            if not _inside_ascii_token(text, match.end())
        ]
        if positions:
            return min(positions, key=lambda pos: (abs(pos - target), pos < target))

    end = min(target, high)
    if _inside_ascii_token(text, end):
        forward = end
        while forward < high and _inside_ascii_token(text, forward):
            forward += 1
        if not _inside_ascii_token(text, forward):
            end = forward
        else:
            backward = end
            while backward > low and _inside_ascii_token(text, backward):
                backward -= 1
            if backward > start:
                end = backward
    return max(start + 1, end)


def split_source_segments(
    text: str,
    target_chars: int = 300,
    min_chars: int = 200,
    max_chars: int = 450,
) -> list[SourceSegment]:
    if not text:
        return []
    if not 0 < min_chars <= target_chars <= max_chars:
        raise ValueError("segment sizes must satisfy 0 < min <= target <= max")

    segments: list[SourceSegment] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_chars:
            end = len(text)
        else:
            end = _safe_cut(
                text,
                start,
                start + target_chars,
                min_chars,
                max_chars,
            )
        source = text[start:end]
        index = len(segments)
        segments.append(
            SourceSegment(
                id=_segment_id(index, start, end),
                start=start,
                end=end,
                source_text=source,
                source_hash=sha256_text(source),
            )
        )
        start = end
    return segments


def build_batches(
    text: str,
    segments: Sequence[SourceSegment],
    target_chars: int = 2700,
    context_chars: int = 200,
) -> list[SegmentBatch]:
    if not segments:
        return []
    batches: list[SegmentBatch] = []
    current: list[SourceSegment] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        first, last = current[0], current[-1]
        batches.append(
            SegmentBatch(
                segments=tuple(current),
                left_context=text[max(0, first.start - context_chars):first.start],
                right_context=text[last.end:min(len(text), last.end + context_chars)],
            )
        )
        current = []
        current_chars = 0

    for segment in segments:
        if current and current_chars + len(segment.source_text) > target_chars:
            flush()
        current.append(segment)
        current_chars += len(segment.source_text)
    flush()
    return batches


def _effective(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def validate_segment_text(source: str, candidate: str) -> FidelityResult:
    if not isinstance(candidate, str) or not candidate.strip():
        return FidelityResult(False, "empty")

    source_effective = _effective(source)
    candidate_effective = _effective(candidate)
    matcher = SequenceMatcher(None, source_effective, candidate_effective, autojunk=False)
    replaced = inserted = deleted = 0
    aligned_source_indexes: set[int] = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        source_len = i2 - i1
        candidate_len = j2 - j1
        if tag == "equal":
            aligned_source_indexes.update(range(i1, i2))
        elif tag == "delete":
            deleted += source_len
            return FidelityResult(False, "content_deleted", replaced, inserted, deleted)
        elif tag == "replace":
            replaced += source_len
            if candidate_len < source_len:
                return FidelityResult(False, "content_deleted", replaced, inserted, deleted)
            if source_len > 12 or candidate_len > 12:
                return FidelityResult(False, "replace_span_too_large", replaced, inserted, deleted)
        elif tag == "insert":
            inserted += candidate_len
            if candidate_len > 12:
                return FidelityResult(False, "insert_span_too_large", replaced, inserted, deleted)

    replace_limit = max(4, math.floor(len(source_effective) * 0.05))
    insert_limit = max(2, math.floor(len(source_effective) * 0.03))
    if replaced > replace_limit:
        return FidelityResult(False, "too_much_replaced", replaced, inserted, deleted)
    if inserted > insert_limit:
        return FidelityResult(False, "too_much_inserted", replaced, inserted, deleted)

    if len(source_effective) >= 12:
        head_matches = sum(index in aligned_source_indexes for index in range(12))
        tail_matches = sum(
            index in aligned_source_indexes
            for index in range(len(source_effective) - 12, len(source_effective))
        )
        if head_matches < 8:
            return FidelityResult(False, "head_anchor_missing", replaced, inserted, deleted)
        if tail_matches < 8:
            return FidelityResult(False, "tail_anchor_missing", replaced, inserted, deleted)

    return FidelityResult(True, "accepted", replaced, inserted, deleted)


def _fallback_decision(segment: SourceSegment, status: str, reason: str) -> SegmentDecision:
    return SegmentDecision(
        segment_id=segment.id,
        text=segment.source_text,
        status=status,
        reason=reason,
        source_hash=segment.source_hash,
        result_hash=segment.source_hash,
        start=segment.start,
        end=segment.end,
    )


def evaluate_batch_response(
    source_segments: Sequence[SourceSegment],
    payload: Any,
) -> dict[str, SegmentDecision]:
    expected = {segment.id: segment for segment in source_segments}
    grouped: dict[str, list[Any]] = defaultdict(list)
    entries = payload.get("segments", []) if isinstance(payload, dict) else []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            segment_id = entry.get("id")
            if segment_id in expected:
                grouped[segment_id].append(entry.get("text"))

    decisions: dict[str, SegmentDecision] = {}
    for segment in source_segments:
        candidates = grouped.get(segment.id, [])
        if not candidates:
            decisions[segment.id] = _fallback_decision(
                segment, "fallback_structure", "missing_id"
            )
            continue
        if len(candidates) != 1:
            decisions[segment.id] = _fallback_decision(
                segment, "fallback_structure", "duplicate_id"
            )
            continue
        candidate = candidates[0]
        if not isinstance(candidate, str) or not candidate.strip():
            decisions[segment.id] = _fallback_decision(
                segment, "fallback_empty", "empty"
            )
            continue
        fidelity = validate_segment_text(segment.source_text, candidate)
        if not fidelity.accepted:
            decision = _fallback_decision(
                segment, "fallback_fidelity", fidelity.reason
            )
            decisions[segment.id] = SegmentDecision(
                **{
                    **decision.__dict__,
                    "replaced": fidelity.replaced,
                    "inserted": fidelity.inserted,
                    "deleted": fidelity.deleted,
                }
            )
            continue
        decisions[segment.id] = SegmentDecision(
            segment_id=segment.id,
            text=candidate.strip(),
            status="optimized",
            reason="accepted",
            source_hash=segment.source_hash,
            result_hash=sha256_text(candidate.strip()),
            start=segment.start,
            end=segment.end,
            replaced=fidelity.replaced,
            inserted=fidelity.inserted,
            deleted=fidelity.deleted,
        )
    return decisions


def assemble_segments(
    source_segments: Sequence[SourceSegment],
    decisions: dict[str, SegmentDecision],
) -> str:
    if set(decisions) != {segment.id for segment in source_segments}:
        return "\n\n".join(segment.source_text for segment in source_segments)
    return "\n\n".join(decisions[segment.id].text for segment in source_segments)


def fallback_batch(
    source_segments: Sequence[SourceSegment],
    status: str,
    reason: str,
) -> dict[str, SegmentDecision]:
    return {
        segment.id: _fallback_decision(segment, status, reason)
        for segment in source_segments
    }
