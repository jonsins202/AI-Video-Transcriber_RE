# Lossless Transcript Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fail-open free-text optimization path with a fail-closed, ID-preserving pipeline in which every source segment is either safely optimized or deterministically restored from the original transcript.

**Architecture:** Keep the raw transcript immutable, extract pure body text before optimization, split it into stable non-overlapping source segments, send batches of ID-tagged segments with read-only context to the LLM, validate response structure and character conservation per ID, and assemble strictly in original ID order. Retain the current optimizer only as a compatibility path while adding immediate empty-output guards; the new lossless path becomes the default.

**Tech Stack:** Python 3.10+, `unittest`, `asyncio`, OpenAI-compatible chat completions, standard-library `dataclasses`, `difflib`, `hashlib`, `json`, `re`.

---

## File map

- Create `backend/lossless_optimizer.py`: pure segmentation, batching, response validation, fidelity validation, deterministic assembly, and audit structures.
- Create `tests/test_lossless_optimizer.py`: unit tests for lossless invariants.
- Create `tests/test_transcriber_language.py`: SenseVoice language-tag regressions.
- Modify `backend/summarizer.py`: empty-output guards, body extraction, lossless LLM orchestration, retries, and audit exposure.
- Modify `backend/transcriber.py`: whitelist language tags and ignore `woitn`-style noise.
- Modify `backend/main.py`: persist audit metadata.
- Modify `tests/test_summarizer_optimize.py`: empty-output and LLM integration regressions.

### Task 1: Close the current empty-output holes

**Files:**
- Modify: `tests/test_summarizer_optimize.py:120-302`
- Modify: `backend/summarizer.py:306-322, 417-448`

- [ ] **Step 1: Write failing regressions**

Add an empty responder and tests:

```python
@staticmethod
def _empty_first_responder_factory():
    calls = {"count": 0}
    def responder(kwargs):
        calls["count"] += 1
        body = _extract_body(kwargs["messages"][-1]["content"])
        return _Response("" if calls["count"] == 1 else body)
    return responder

async def test_empty_chunk_falls_back_to_original(self):
    s = self._make_summarizer(self._empty_first_responder_factory())
    text = "甲" * 4000 + "乙" * 4000 + "丙" * 4000
    result = await s._format_long_transcript_in_chunks(text, "zh", 4000)
    self.assertIn("甲" * 100, result)

def test_empty_overall_result_falls_back(self):
    s = Summarizer(api_key="sk-test-dummy")
    raw = "这是不可丢失的原始正文" * 50
    self.assertIn("这是不可丢失的原始正文", s._maybe_overall_fallback("", raw))
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
venv\Scripts\python.exe -m unittest tests.test_summarizer_optimize.OptimizeChunkContextTests.test_empty_chunk_falls_back_to_original tests.test_summarizer_optimize.OptimizeChunkContextTests.test_empty_overall_result_falls_back -v
```

Expected: first chunk is absent and the overall fallback returns empty.

- [ ] **Step 3: Implement minimal fail-closed guards**

```python
if not oc.strip():
    logger.warning(f"第 {i+1} 块优化结果为空，使用原始块兜底")
    oc = self._apply_basic_formatting(c)
elif len(oc.strip()) < len(c.strip()) * 0.7:
    logger.warning(f"第 {i+1} 块优化后异常短，使用原始块兜底")
    oc = self._apply_basic_formatting(c)
```

At document level:

```python
if not merged.strip():
    logger.warning("优化后整体为空，整体使用原文兜底")
    return self._apply_basic_formatting(raw_transcript)
```

- [ ] **Step 4: Run `venv\Scripts\python.exe tests\test_summarizer_optimize.py`**

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/summarizer.py tests/test_summarizer_optimize.py
git commit -m "fix: fail closed on empty transcript optimization"
```

### Task 2: Implement immutable source segmentation and batching

**Files:**
- Create: `backend/lossless_optimizer.py`
- Create: `tests/test_lossless_optimizer.py`

- [ ] **Step 1: Write failing segmentation tests**

```python
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from lossless_optimizer import build_batches, split_source_segments

class SourceSegmentationTests(unittest.TestCase):
    def test_segments_cover_source_exactly_once(self):
        text = "无标点中文正文" * 200
        segments = split_source_segments(text)
        self.assertEqual("".join(s.source_text for s in segments), text)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, len(text))
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left.end, right.start)

    def test_split_does_not_cut_ascii_token(self):
        text = "中" * 290 + " https://example.com/path?q=123 " + "文" * 320
        segments = split_source_segments(text)
        self.assertTrue(any("https://example.com/path?q=123" in s.source_text for s in segments))

    def test_batches_do_not_repeat_core_segments(self):
        text = "甲乙丙丁" * 1000
        segments = split_source_segments(text)
        batches = build_batches(text, segments)
        self.assertEqual(
            [s.id for b in batches for s in b.segments],
            [s.id for s in segments],
        )
```

- [ ] **Step 2: Run `venv\Scripts\python.exe -m unittest tests.test_lossless_optimizer.SourceSegmentationTests -v`**

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement segment and batch primitives**

```python
from dataclasses import dataclass
import hashlib

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

def _segment_id(index, start, end):
    return f"seg-{index:06d}-{start:08d}-{end:08d}"

def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Implement `split_source_segments(text, target_chars=300, min_chars=200, max_chars=450)` by preferring paragraph, sentence, phrase, and space boundaries near the target, otherwise hard-cutting without splitting a continuous ASCII token. Never strip slices. Implement `build_batches(text, segments, target_chars=2700, context_chars=200)` with each segment in exactly one core batch and contexts sliced from the immutable source.

- [ ] **Step 4: Run the segmentation tests**

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/lossless_optimizer.py tests/test_lossless_optimizer.py
git commit -m "feat: add immutable transcript source segments"
```

### Task 3: Implement fidelity validation

**Files:**
- Modify: `backend/lossless_optimizer.py`
- Modify: `tests/test_lossless_optimizer.py`

- [ ] **Step 1: Write failing validator tests**

```python
from lossless_optimizer import validate_segment_text

class FidelityValidationTests(unittest.TestCase):
    def test_punctuation_passes(self):
        result = validate_segment_text("大家好我是小俊今天开始", "大家好，我是小俊。今天开始。")
        self.assertTrue(result.accepted, result.reason)

    def test_content_deletion_fails(self):
        result = validate_segment_text("主持人介绍完整内容", "主持人介绍内容")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "content_deleted")

    def test_summary_rewrite_fails_even_when_long(self):
        source = "这是完整访谈原文包含开场寒暄引用举例以及详细论述" * 20
        candidate = "本段主要讨论访谈主题。" * 50
        self.assertFalse(validate_segment_text(source, candidate).accepted)

    def test_empty_fails(self):
        self.assertEqual(validate_segment_text("不可丢失", "").reason, "empty")
```

- [ ] **Step 2: Run validator tests and verify missing symbol failures**

```powershell
venv\Scripts\python.exe -m unittest tests.test_lossless_optimizer.FidelityValidationTests -v
```

- [ ] **Step 3: Implement ordered validation**

```python
from difflib import SequenceMatcher
from math import floor

@dataclass(frozen=True)
class FidelityResult:
    accepted: bool
    reason: str
    replaced: int = 0
    inserted: int = 0
    deleted: int = 0

def _effective(text):
    return "".join(ch.lower() for ch in text if ch.isalnum())
```

`validate_segment_text()` must reject every delete opcode; restrict each replace and insert span to 12 characters; restrict cumulative replacement to 5% and insertion to 3%; reject replacements shorter than their source; and require at least 8 of the first and last 12 effective source characters to align exactly. Use `SequenceMatcher(autojunk=False)` and stable audit reason strings.

- [ ] **Step 4: Run `venv\Scripts\python.exe -m unittest tests.test_lossless_optimizer -v`**

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/lossless_optimizer.py tests/test_lossless_optimizer.py
git commit -m "feat: validate transcript edits for content conservation"
```

### Task 4: Map responses and assemble by source ID

**Files:**
- Modify: `backend/lossless_optimizer.py`
- Modify: `tests/test_lossless_optimizer.py`

- [ ] **Step 1: Write failing response tests**

```python
from lossless_optimizer import assemble_segments, evaluate_batch_response

class ResponseAssemblyTests(unittest.TestCase):
    def test_bad_ids_fall_back_without_loss(self):
        text = "甲" * 300 + "乙" * 300 + "丙" * 300
        segments = split_source_segments(text)
        payload = {"segments": [
            {"id": segments[1].id, "text": segments[1].source_text},
            {"id": segments[0].id, "text": ""},
            {"id": segments[1].id, "text": segments[1].source_text},
            {"id": "unknown", "text": "invented"},
        ]}
        decisions = evaluate_batch_response(segments, payload)
        self.assertEqual(assemble_segments(segments, decisions).replace("\n", ""), text)

    def test_response_order_cannot_change_output_order(self):
        text = "第一段。" * 60 + "第二段。" * 60
        segments = split_source_segments(text)
        payload = {"segments": [{"id": s.id, "text": s.source_text} for s in reversed(segments)]}
        decisions = evaluate_batch_response(segments, payload)
        self.assertEqual(assemble_segments(segments, decisions).replace("\n", ""), text)
```

- [ ] **Step 2: Run response tests and verify missing symbol failures**

- [ ] **Step 3: Implement decisions and assembly**

```python
@dataclass(frozen=True)
class SegmentDecision:
    segment_id: str
    text: str
    status: str
    reason: str
    source_hash: str
    result_hash: str
    replaced: int = 0
    inserted: int = 0
    deleted: int = 0

def assemble_segments(source_segments, decisions):
    if set(decisions) != {s.id for s in source_segments}:
        return "\n\n".join(s.source_text for s in source_segments)
    return "\n\n".join(decisions[s.id].text for s in source_segments)
```

`evaluate_batch_response()` groups entries by ID, ignores unknown IDs, and adopts a candidate only when it occurs once, is non-empty, and passes fidelity validation. Missing, duplicate, empty, or rejected candidates store the original source text and an explicit fallback status.

- [ ] **Step 4: Run all lossless optimizer tests**

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/lossless_optimizer.py tests/test_lossless_optimizer.py
git commit -m "feat: assemble optimized transcript by immutable segment id"
```

### Task 5: Add lossless LLM orchestration

**Files:**
- Modify: `backend/summarizer.py:1-208`
- Modify: `tests/test_summarizer_optimize.py`

- [ ] **Step 1: Add fake-client integration tests**

Create `LosslessOptimizeTests` for: an empty segment result, a `finish_reason="length"` response followed by two successful half-batch responses, and a long summary rewrite. Fake responses must expose both `message.content` and `finish_reason`; extract request IDs from the JSON user message rather than hardcoding IDs.

- [ ] **Step 2: Run `venv\Scripts\python.exe -m unittest tests.test_summarizer_optimize.LosslessOptimizeTests -v`**

Expected: failures because lossless orchestration does not exist.

- [ ] **Step 3: Implement JSON request and response handling**

Add `_extract_json_object()` that strips optional fences, takes the first `{` through the last `}`, and runs `json.loads()`. Add `_call_lossless_batch()` with a system prompt that declares contexts read-only, requires every editable ID once, and permits only typo/proper-noun/punctuation/paragraph corrections.

Capture the provider result exactly:

```python
choice = response.choices[0]
content = choice.message.content or ""
finish_reason = getattr(choice, "finish_reason", None)
```

Return structured failure metadata for missing choices, empty content, JSON errors, and abnormal finish reasons.

- [ ] **Step 4: Implement serial batching, retry, split-on-length, and audit**

`_optimize_lossless_body()` must split source segments, build batches, process serially, retry transient/JSON errors once, split truncated multi-segment batches in half once, evaluate parsed responses per ID, fallback unparseable batches, assemble by ID, and populate `self.last_optimization_audit` without source or candidate text.

Expose `optimize_transcript_body(body, language=None)` as the pure-body entry point and make `optimize_transcript(markdown_or_text)` a compatibility wrapper that extracts the body before calling it. Keep `_format_long_transcript_in_chunks()` only for compatibility tests.

- [ ] **Step 5: Run optimizer and pure lossless tests**

```powershell
venv\Scripts\python.exe -m unittest tests.test_summarizer_optimize tests.test_lossless_optimizer -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add backend/summarizer.py tests/test_summarizer_optimize.py backend/lossless_optimizer.py tests/test_lossless_optimizer.py
git commit -m "feat: optimize transcripts through lossless id batches"
```

### Task 6: Keep Markdown metadata out of LLM input

**Files:**
- Modify: `backend/summarizer.py:56-87, 452-469`
- Modify: `backend/main.py:205-223`
- Modify: `tests/test_summarizer_optimize.py`

- [ ] **Step 1: Write body extraction tests**

```python
def test_extract_body_removes_english_metadata(self):
    raw = """# Video Transcription

**Detected Language:** woitn
**Language Probability:** —

## Transcription Content

hello 大家好这是正文
"""
    self.assertEqual(Summarizer.extract_transcript_body(raw), "hello 大家好这是正文")

def test_extract_body_removes_timestamps_only(self):
    raw = "**[00:00 - 00:05]**\n\n第一段\n\n**[00:05 - 00:10]**\n\n第二段"
    self.assertEqual(Summarizer.extract_transcript_body(raw), "第一段\n\n第二段")
```

- [ ] **Step 2: Run tests and verify missing method failure**

- [ ] **Step 3: Implement `extract_transcript_body()`**

Remove only recognized generated titles, Chinese/English language fields, timestamp-only lines, and recognized `Transcription Content` headings. Preserve all other line content and normalize blank-line runs. Make `_remove_timestamps_and_meta()` delegate to this method.

- [ ] **Step 4: Pass pure body text into the default optimize path**

```python
transcript_body = request_summarizer.extract_transcript_body(raw_script)
with _stage(task_id, "optimize"):
    script = await request_summarizer.optimize_transcript_body(transcript_body)
```

The archived `raw_script` remains unchanged while no Markdown metadata reaches the LLM.

- [ ] **Step 5: Run optimizer tests; expect all pass**

- [ ] **Step 6: Commit**

```powershell
git add backend/summarizer.py backend/main.py tests/test_summarizer_optimize.py
git commit -m "fix: keep transcript metadata out of llm input"
```

### Task 7: Reject unknown SenseVoice language tags

**Files:**
- Create: `tests/test_transcriber_language.py`
- Modify: `backend/transcriber.py:156-161, 249-262`

- [ ] **Step 1: Write failing language tests**

```python
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from transcriber import _FunASRBackend

class SenseVoiceLanguageTests(unittest.TestCase):
    def test_noise_tag_is_not_language(self):
        self.assertIsNone(_FunASRBackend._parse_language("<|woitn|>正文", {}))

    def test_known_tag_after_noise_is_found(self):
        self.assertEqual(_FunASRBackend._parse_language("<|HAPPY|><|woitn|><|zh|>正文", {}), "zh")

    def test_known_iso_tag_is_kept(self):
        self.assertEqual(_FunASRBackend._parse_language("<|fr|>bonjour", {}), "fr")
```

- [ ] **Step 2: Run `venv\Scripts\python.exe -m unittest tests.test_transcriber_language -v`**

Expected: `woitn` is returned and later known tags are not scanned.

- [ ] **Step 3: Implement whitelist scanning**

Iterate every `<|tag|>` in order. Return the first `_SENSEVOICE_LANG_TAG` match or supported ISO value from `LANG_NAME_TO_ISO.values()`; return `None` when no known tag exists. Preserve the Paraformer `result["language"]` fallback.

- [ ] **Step 4: Run language and optimizer tests; expect all pass**

- [ ] **Step 5: Commit**

```powershell
git add backend/transcriber.py tests/test_transcriber_language.py
git commit -m "fix: ignore unknown SenseVoice language tags"
```

### Task 8: Persist audit metadata and cover the historical failure

**Files:**
- Modify: `backend/main.py:194-223`
- Modify: `backend/summarizer.py`
- Modify: `tests/test_summarizer_optimize.py`

- [ ] **Step 1: Write an audit test**

Assert `get_last_optimization_audit()` is JSON serializable and includes model, source length, segment/batch counts, status counts, IDs, offsets, hashes, reasons, and edit counts while excluding `source_text`, `candidate_text`, prompts, and the complete transcript.

- [ ] **Step 2: Run the audit test and verify failure**

- [ ] **Step 3: Save audit sidecar after optimize**

```python
audit = request_summarizer.get_last_optimization_audit()
if audit:
    audit_filename = f"optimization_audit_{safe_title}_{short_id}.json"
    async with aiofiles.open(TEMP_DIR / audit_filename, "w", encoding="utf-8") as f:
        await f.write(json.dumps(audit, ensure_ascii=False, indent=2))
    tasks[task_id]["optimization_audit_file"] = audit_filename
```

Audit write failures are logged and do not replace the already safe transcript result.

- [ ] **Step 4: Make raw transcript persistence failure fatal**

```python
except Exception:
    logger.exception("保存原始转录 Markdown 失败，停止后续优化")
    raise
```

The pipeline must not optimize until its immutable fact source is durable.

- [ ] **Step 5: Add a 21-range empty-response regression**

Generate no-punctuation source with 21 identifiable ranges; fake empty output for historical ranges 1, 3, 4, 12, 13, 14, 16, and 20; assert every marker remains in output and the audit reports fallbacks rather than silent deletion.

- [ ] **Step 6: Run complete tests**

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all pass.

- [ ] **Step 7: Run compilation checks**

```powershell
venv\Scripts\python.exe -m py_compile backend\lossless_optimizer.py backend\summarizer.py backend\transcriber.py backend\main.py
```

Expected: exit code 0.

- [ ] **Step 8: Verify the real artifact without a paid API call**

Extract `temp/raw_146_*_2f0c1c.md`, split it, and assert concatenated source segments exactly equal the extracted body. Print source length, segment count, and batch count.

- [ ] **Step 9: Commit**

```powershell
git add backend/main.py backend/summarizer.py tests/test_summarizer_optimize.py
git commit -m "feat: audit lossless transcript optimization"
```

### Task 9: Final review and documentation

**Files:**
- Modify: `docs/transcript-loss-investigation.md`

- [ ] **Step 1: Add the verified root-cause addendum**

State that empty LLM outputs bypassing `if oc.strip()` were the direct loss mechanism; hard chunk boundaries remain a contributing reliability risk, not the proven primary cause.

- [ ] **Step 2: Check scope and whitespace**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and no unrelated implementation files added to commits.

- [ ] **Step 3: Re-run full tests and compilation from Task 8**

Expected: all tests pass and compilation succeeds.

- [ ] **Step 4: Commit documentation**

```powershell
git add docs/transcript-loss-investigation.md
git commit -m "docs: record verified transcript loss mechanism"
```
