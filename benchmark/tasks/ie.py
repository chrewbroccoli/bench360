import os
import json
import gzip
import glob
import random
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union

from benchmark.utils import normalize_answer
from benchmark.tasks.base_task import BaseTask


class InfoExtractionTask(BaseTask):
    """
    Task for field extraction on the VRDU dataset.

    Assumes the following layout under base_path (default: ./datasets/vrdu-main):

        ./datasets/vrdu-main/
          registration-form/
            main/
              dataset.jsonl.gz  (or dataset.jsonl)
              pdfs/
              meta.json
          ad-buy-form/
            main/
              dataset.jsonl.gz  (or dataset.jsonl)
              pdfs/
              meta.json
            few_shot-splits/   (optional)

    Each JSONL line is expected to contain OCR and annotations. We extract:
      - OCR text -> prompt context
      - annotated field names + values -> target JSON the model must output
    """

    def __init__(self, base_path: str = "./datasets/vrdu-main", seed: int = 42,
                 max_chars_context: int = 5000, max_fields: int = 30):
        """
        Args:
            base_path: Root of the VRDU dataset (contains *-form/ subfolders).
            seed: RNG seed for reproducibility.
            max_chars_context: Cap the OCR context length in prompts.
            max_fields: Cap number of fields per document to keep prompts reasonable.
        """
        super().__init__()
        random.seed(seed)
        self.base_path = base_path
        self.max_chars_context = max_chars_context
        self.max_fields = max_fields

        # Load and cache all entries across present corpora.
        self.entries: List[Dict[str, Any]] = []
        for corpus_dir in sorted(glob.glob(os.path.join(self.base_path, "*-form"))):
            main_dir = os.path.join(corpus_dir, "main")
            if not os.path.isdir(main_dir):
                continue
            jsonl_path = self._pick_jsonl(main_dir)
            if not jsonl_path:
                continue
            self.entries.extend(self._read_jsonl(jsonl_path))

        if not self.entries:
            raise FileNotFoundError(
                f"No VRDU entries found under {self.base_path}. "
                "Make sure dataset.jsonl(.gz) exists under */main/"
            )

    # ----------------------------
    # BaseTask API
    # ----------------------------
    def generate_prompts(self, num_examples: int = 100) -> Tuple[List[str], List[str]]:
        """
        Returns:
            prompts: list[str] — each prompt instructs the model to output JSON with field->value(s)
            references: list[str] — JSON strings representing the gold field->value(s) mapping
        """
        sample = random.sample(self.entries, k=min(num_examples, len(self.entries)))
        prompts: List[str] = []
        references: List[str] = []

        for ex in sample:
            fields_to_values = self._extract_gold_fields(ex)
            if not fields_to_values:
                # Skip documents with no annotations
                continue

            # Limit number of fields to keep prompt compact
            trimmed_fields = dict(list(fields_to_values.items())[: self.max_fields])

            ocr_text = self._extract_ocr_text(ex, max_chars=self.max_chars_context)

            prompt = self._build_prompt(ocr_text, list(trimmed_fields.keys()))
            ref_json = json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True)

            prompts.append(prompt)
            references.append(ref_json)

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate model JSON output versus gold JSON (strict JSON strings).

        Metrics returned:
          - subset_em: 1.0 if every gold field exactly matches (after normalization)
          - field_em: micro-avg per-field exact match (set equality after normalization)
          - field_f1: micro-avg per-field token F1 (existing behavior)
          - field_substring: micro-avg per-field substring match (gold fully contained in any pred)
          - field_fuzzy: micro-avg per-field fuzzy similarity (best SequenceMatcher ratio)
        """
        gold = self._safe_json_loads(reference)
        pred = self._safe_json_loads(generated)

        # Ensure dicts
        if not isinstance(gold, dict):
            gold = {}
        if not isinstance(pred, dict):
            pred = {}

        gold_fields = set(gold.keys())

        per_field_em: List[float] = []
        per_field_f1: List[float] = []
        per_field_sub: List[float] = []
        per_field_fuzzy: List[float] = []

        for f in gold_fields:
            gold_vals = self._to_list_of_str(gold.get(f, []))
            pred_vals = self._to_list_of_str(pred.get(f, []))

            # If model omitted the field entirely, score zeros across metrics.
            if len(pred_vals) == 0:
                per_field_em.append(0.0)
                per_field_f1.append(0.0)
                # substring: only 1.0 if gold also empty
                per_field_sub.append(1.0 if len(gold_vals) == 0 else 0.0)
                # fuzzy: define as 1.0 only if both empty, else 0
                per_field_fuzzy.append(1.0 if len(gold_vals) == 0 else 0.0)
                continue

            # Exact match (set equality after normalization)
            per_field_em.append(self._field_exact_em(gold_vals, pred_vals))

            # Token F1 (existing semantics)
            per_field_f1.append(self._field_token_f1(gold_vals, pred_vals))

            # Substring match (gold fully contained in any prediction)
            per_field_sub.append(self._field_substring_match(gold_vals, pred_vals))

            # Fuzzy similarity (best SequenceMatcher ratio per gold, then average)
            per_field_fuzzy.append(self._field_fuzzy_similarity(gold_vals, pred_vals))

        field_em = sum(per_field_em) / len(per_field_em) if per_field_em else 0.0
        field_f1 = sum(per_field_f1) / len(per_field_f1) if per_field_f1 else 0.0
        field_substring = sum(per_field_sub) / len(per_field_sub) if per_field_sub else 0.0
        field_fuzzy = sum(per_field_fuzzy) / len(per_field_fuzzy) if per_field_fuzzy else 0.0
        subset_em = 1.0 if field_em == 1.0 else 0.0

        return {
            "subset_em": subset_em,
            "field_em": field_em,
            "field_f1": field_f1,
            "field_substring": field_substring,
            "field_fuzzy": field_fuzzy,
        }

    # ----------------------------
    # Prompting
    # ----------------------------
    def _build_prompt(self, ocr_text: str, fields: List[str]) -> str:
        """ Construct a compact, deterministic prompt. The model must output strict JSON only, formatted as a single line (no newlines or indentation). """

        system_message = (
            "You extract structured fields from OCR text of a document.\n"
            "Output strictly a single JSON object with ONLY the requested keys.\n"
            "If a field is present multiple times, output a JSON array of unique values in reading order.\n"
            "Do not include extra commentary, explanations, or keys.\n"
            "The JSON must be compact and on a single line (no line breaks or spaces beyond those inside values)."
        )

        examples = (
            "### EXAMPLE\n"
            "OCR:\n"
            "Invoice # 12345 | Date: 2023-08-01 | Total: $19.99\n"
            "Vendor: ACME Corp.\n"
            "\n"
            "Requested keys:\n"
            "invoice_number, date, total_amount, vendor\n\n"
            "Your JSON (single line):\n"
            '{"invoice_number":"12345","date":"2023-08-01","total_amount":"$19.99","vendor":"ACME Corp."}\n')

        instruction = (
            "### INSTRUCTION\n" "Read the OCR text and return a JSON with EXACTLY these keys:\n"
            f"{', '.join(fields)}\n\n" "Rules:\n"
            " • Return strings for single values, and arrays for repeated values.\n"
            " • Preserve numbers/dates as they appear when reasonable.\n"
            " • If a key truly cannot be found, set it to null.\n"
            " • Output must be a single-line JSON string with no extra whitespace or newlines.\n")

        input_block = (
            "### OCR\n" f"{ocr_text}\n\n" "### OUTPUT (JSON only, one line)\n"
        )
        return f"{system_message}\n\n{examples}\n\n{instruction}\n{input_block}"
    # ----------------------------
    # Utilities: loading & parsing
    # ----------------------------
    def _pick_jsonl(self, main_dir: str) -> Union[str, None]:
        """Prefer .jsonl.gz if present, else .jsonl."""
        gz = os.path.join(main_dir, "dataset.jsonl.gz")
        jl = os.path.join(main_dir, "dataset.jsonl")
        if os.path.isfile(gz):
            return gz
        if os.path.isfile(jl):
            return jl
        return None

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Read (gzipped) JSONL into a list of dicts."""
        entries = []
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entries.append(obj)
                except Exception:
                    continue
        return entries

    def _extract_ocr_text(self, ex: Dict[str, Any], max_chars: int = 4000) -> str:
        """
        Best-effort OCR extraction across possible formats.
        """
        ocr = ex.get("ocr")

        def textify(obj) -> List[str]:
            out = []
            if isinstance(obj, str):
                out.append(obj)
            elif isinstance(obj, dict):
                # common keys
                if "text" in obj and isinstance(obj["text"], str):
                    out.append(obj["text"])
                # hierarchical
                for k in ("pages", "lines", "tokens", "blocks", "spans", "words", "items"):
                    if k in obj and isinstance(obj[k], list):
                        for item in obj[k]:
                            out.extend(textify(item))
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        if "text" in item and isinstance(item["text"], str):
                            out.append(item["text"])
                        else:
                            out.extend(textify(item))
                    elif isinstance(item, (list, tuple)) and len(item) >= 1:
                        # e.g., ["text", [bbox]]
                        if isinstance(item[0], str):
                            out.append(item[0])
            return out

        chunks = textify(ocr)
        text = " ".join(chunks)
        text = " ".join(text.split())  # squish whitespace
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        return text

    def _extract_gold_fields(self, ex: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        """
        Parse annotations into {field_name: value or [values]}.

        Handles:
          - annotations as dict: {field: [[text, bbox], ...]} or {field: ["text", ...]}
          - annotations as list: [[field, [[text, bbox], ...]], ...]
          - values possibly split across spans -> joined in reading order
          - FIXES:
              * collapse repeated runs inside a scalar ("A A A" -> "A")
              * exact de-dup on lists (after cleanup)
              * collapse single-item lists to scalars
        """
        ann = ex.get("annotations")
        if ann is None:
            return {}

        def spans_to_values(spans) -> List[str]:
            """
            Normalize one field's annotation bundle to a list of 0..N string values.
            - If 'spans' is a flat list of text fragments -> produce ONE concatenated value.
            - If 'spans' looks like a list of instances (each instance is a list/tuple of fragments)
              -> produce multiple values (one per instance).
            Repeated chunk runs inside a value are collapsed (e.g., "X X X" -> "X").
            """
            vals: List[str] = []
            if not isinstance(spans, list):
                return vals

            # Heuristic: list of instances?
            is_list_of_instances = (
                spans
                and all(isinstance(it, (list, tuple)) and any(self._span_text(p) for p in it) for it in spans)
            )

            if is_list_of_instances:
                for inst in spans:
                    pieces = [self._span_text(p) for p in inst if self._span_text(p)]
                    if pieces:
                        s = " ".join(pieces)
                        s = " ".join(s.split())
                        s = self._collapse_repeated_runs(s)
                        if s:
                            vals.append(s)
                return vals

            # Flat list -> single instance
            pieces = [self._span_text(s) for s in spans if self._span_text(s)]
            if pieces:
                s = " ".join(pieces)
                s = " ".join(s.split())
                s = self._collapse_repeated_runs(s)
                if s:
                    vals.append(s)
            return vals

        out: Dict[str, Union[str, List[str]]] = {}

        if isinstance(ann, dict):
            for field, spans in ann.items():
                vals = spans_to_values(spans)
                if not vals:
                    continue
                out[field] = vals if len(vals) > 1 else vals[0]

        elif isinstance(ann, list):
            # list of [field, spans] pairs
            for item in ann:
                if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                    continue
                field, spans = item[0], item[1]
                if not isinstance(field, str):
                    continue
                vals = spans_to_values(spans)
                if not vals:
                    continue
                if field in out:
                    prev = self._to_list_of_str(out[field])
                    out[field] = prev + vals
                else:
                    out[field] = vals if len(vals) > 1 else vals[0]

        # Cleanup + exact de-dup + collapse singletons
        cleaned: Dict[str, Union[str, List[str]]] = {}
        for k, v in out.items():
            if isinstance(v, list):
                seen: List[str] = []
                for s in v:
                    s2 = " ".join(s.split())
                    s2 = self._collapse_repeated_runs(s2)
                    if s2 and s2 not in seen:  # exact de-dup only
                        seen.append(s2)
                if len(seen) == 0:
                    continue  # drop empty fields
                if len(seen) == 1:
                    cleaned[k] = seen[0]  # collapse singleton list to scalar
                else:
                    cleaned[k] = seen
            else:
                s2 = " ".join(str(v).split())
                s2 = self._collapse_repeated_runs(s2)
                cleaned[k] = s2
        return cleaned

    # ----------------------------
    # Utilities: helpers (no fuzzy logic here)
    # ----------------------------
    @staticmethod
    def _span_text(x) -> str:
        """Extract textual content from a span element in various shapes."""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            t = x.get("text")
            return t if isinstance(t, str) else ""
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
            # e.g., ["text", bbox, ...]
            return x[0]
        return ""

    @staticmethod
    def _collapse_repeated_runs(s: str, max_k: int = 8) -> str:
        """
        Collapse exact repeated runs of a token sequence:
        "A A A" -> "A"
        "John Doe John Doe" -> "John Doe"
        "$20,650.00 $20,650.00 $20,650.00" -> "$20,650.00"
        Works at token level to preserve multi-word phrases.
        """
        toks = s.split()
        n = len(toks)
        if n <= 1:
            return s
        for k in range(2, min(max_k, n) + 1):
            if n % k != 0:
                continue
            chunk_len = n // k
            chunk = toks[:chunk_len]
            if chunk * k == toks:
                return " ".join(chunk)
        return s

    # ----------------------------
    # Utilities: metrics
    # ----------------------------
    def _safe_json_loads(self, s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            try:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start : end + 1])
            except Exception:
                pass
        return {}

    def _to_list_of_str(self, v: Union[str, List[Any], None]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    def _token_f1(self, gold: str, preds: List[str]) -> float:
        """Max token F1 between one gold value and a list of predicted candidates."""
        def toks(s: str) -> List[str]:
            return normalize_answer(s).split()

        g = toks(gold)
        if not g:
            return 1.0 if not any(toks(p) for p in preds) else 0.0

        best = 0.0
        for p in preds:
            pt = toks(p)
            if not pt:
                best = max(best, 0.0)
                continue
            common = set(g) & set(pt)
            if not common:
                best = max(best, 0.0)
                continue
            precision = len(common) / len(pt)
            recall = len(common) / len(g)
            f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        return best

    def _normalize_list(self, vals: List[str]) -> List[str]:
        """Normalize a list of strings using the benchmark normalize_answer."""
        return [normalize_answer(v) for v in vals if v is not None]

    def _field_exact_em(self, gold_vals: List[str], pred_vals: List[str]) -> float:
        """
        Exact match at field level:
        - Normalize values
        - Compare as sets (order/dupes ignored)
        """
        gold_norm = set(self._normalize_list(gold_vals))
        pred_norm = set(self._normalize_list(pred_vals))
        return 1.0 if gold_norm == pred_norm else 0.0

    def _field_token_f1(self, gold_vals: List[str], pred_vals: List[str]) -> float:
        """
        Per-field token F1:
        - If gold empty: 1 if pred empty else 0
        - Else: for each gold value, take max token F1 vs all preds, then average
        """
        if len(gold_vals) == 0:
            return 1.0 if len(pred_vals) == 0 else 0.0
        scores = [self._token_f1(gv, pred_vals) for gv in gold_vals]
        return sum(scores) / len(scores) if scores else 0.0

    def _field_substring_match(self, gold_vals: List[str], pred_vals: List[str]) -> float:
        """
        Substring Match:
        Each gold value must appear as a full, contiguous substring in at least
        one predicted value (after normalization). We average over gold values.
        """
        gold_norm = self._normalize_list(gold_vals)
        pred_norm = self._normalize_list(pred_vals)

        if not gold_norm:
            return 1.0 if not pred_norm else 0.0

        def contained(g: str) -> float:
            return 1.0 if any(g in p for p in pred_norm) else 0.0

        scores = [contained(g) for g in gold_norm]
        return sum(scores) / len(scores)

    def _field_fuzzy_similarity(self, gold_vals: List[str], pred_vals: List[str]) -> float:
        """
        Fuzzy similarity metric (kept for evaluation only).
        """
        gold_norm = self._normalize_list(gold_vals)
        pred_norm = self._normalize_list(pred_vals)

        if not gold_norm:
            return 1.0 if not pred_norm else 0.0
        if not pred_norm:
            return 0.0

        def best_ratio(g: str) -> float:
            best = 0.0
            for p in pred_norm:
                best = max(best, SequenceMatcher(None, g, p).ratio())
            return best

        scores = [best_ratio(g) for g in gold_norm]
        return sum(scores) / len(scores)


if __name__ == "__main__":
    task = InfoExtractionTask(base_path="../../datasets/vrdu-main", seed=42)
    prompts, references = task.generate_prompts(num_examples=3)
    for i in range(len(prompts)):
        print(f"Prompt {i+1}:\n{prompts[i]}\n")
        print(f"Reference {i+1}:\n{references[i]}\n")
