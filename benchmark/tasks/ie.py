import os
import json
import gzip
import glob
import random
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union, Literal, Optional

from benchmark.utils import normalize_answer
from benchmark.tasks.base_task import BaseTask


class InfoExtractionTask(BaseTask):
    """
    Task for field extraction on the VRDU dataset.
    """

    def __init__(self, base_path: str = "./datasets/vrdu-main",
                 dataset_name: Optional[Literal["ad-buy", "registration"]] = None,
                 seed: int = 42,
                 max_chars_context: int = 5000, max_fields: int = 30):
        super().__init__()
        random.seed(seed)
        self.base_path = base_path
        self.max_chars_context = max_chars_context
        self.max_fields = max_fields

        # Define which folders to look for
        if dataset_name:
            # Construct the specific folder name (e.g., "ad-buy-form")
            search_pattern = os.path.join(self.base_path, f"{dataset_name}-form")
        else:
            # Fallback to original behavior (all folders)
            search_pattern = os.path.join(self.base_path, "*-form")

        self.entries: List[Dict[str, Any]] = []

        # glob.glob will now only find the specific folder if dataset_name is set
        for corpus_dir in sorted(glob.glob(search_pattern)):
            main_dir = os.path.join(corpus_dir, "main")
            if not os.path.isdir(main_dir):
                continue
            jsonl_path = self._pick_jsonl(main_dir)
            if not jsonl_path:
                continue
            self.entries.extend(self._read_jsonl(jsonl_path))

        if not self.entries:
            raise FileNotFoundError(
                f"No VRDU entries found matching: {search_pattern}"
            )

    # ----------------------------
    # BaseTask API
    # ----------------------------
    def generate_prompts(self, num_examples: int = 100) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Returns:
            prompts: list[dict] — each dict has {"message": [ {role, content}, ... ]}
            references: list[str] — JSON strings representing the gold field->value(s) mapping
        """
        sample = random.sample(self.entries, k=min(num_examples, len(self.entries)))
        prompts: List[Dict[str, Any]] = []
        references: List[str] = []

        for ex in sample:
            fields_to_values = self._extract_gold_fields(ex)
            if not fields_to_values:
                continue

            trimmed_fields = dict(list(fields_to_values.items())[: self.max_fields])
            ocr_text = self._extract_ocr_text(ex, max_chars=self.max_chars_context)

            messages = self._build_prompt(ocr_text, list(trimmed_fields.keys()))
            ref_json = json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True)

            # IMPORTANT: matches your router: dict with at least one "message"
            prompts.append({"message": messages})
            references.append(ref_json)

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculates the F1 score based on Exact Match logic for the JSON object.
        """
        # Parse inputs
        gold = self._safe_json_loads(reference)
        pred = self._safe_json_loads(generated)

        if not isinstance(gold, dict):
            gold = {}
        if not isinstance(pred, dict):
            pred = {}

        tp = 0
        fp = 0
        fn = 0

        # 1. Evaluate keys present in Ground Truth
        for key, gt_val in gold.items():
            # Skip empty gold values
            if gt_val in [None, ""]:
                continue

            pred_val = pred.get(key)

            if pred_val in [None, ""]:
                # Key exists in Gold but missing/empty in Prediction -> False Negative
                fn += 1
            else:
                # Both exist, check for match (Exact Match Logic)
                # Normalize to lists of strings for robust comparison

                # Handle Ground Truth
                if isinstance(gt_val, (str, int, float, bool)):
                    norm_gt = [str(gt_val)]
                elif isinstance(gt_val, list):
                    norm_gt = [str(v) for v in gt_val]
                else:
                    norm_gt = [str(gt_val)]  # Fallback

                # Handle Prediction
                if isinstance(pred_val, (str, int, float, bool)):
                    norm_pred = [str(pred_val)]
                elif isinstance(pred_val, list):
                    norm_pred = [str(v) for v in pred_val]
                else:
                    norm_pred = [str(pred_val)]  # Fallback

                # Sort to ignore order in lists
                if sorted(norm_gt) == sorted(norm_pred):
                    tp += 1
                else:
                    # Value mismatch -> False Positive
                    fp += 1

        # 2. Handle Extra Keys in Predictions (Hallucinations)
        for pred_key in pred:
            if pred_key not in gold:
                # Only count if the prediction actually has a value
                if pred.get(pred_key) not in [None, ""]:
                    fp += 1

        # 3. Calculate F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Note: subset_em is strictly 1.0 only if F1 is 1.0 (perfect match)
        subset_em = 1.0 if f1 == 1.0 else 0.0

        # For backward compatibility or extra info, we can keep average field stats if needed,
        # but here we follow the requested overwrite logic.
        return {
            "subset_em": subset_em,
            "field_f1": f1,
            # We can zero out or remove the fuzzy/substring metrics if they are no longer needed
            # or implement them similarly if required.
            "field_em": precision,  # Using precision as a proxy for field_em in this context
            "field_substring": 0.0,
            "field_fuzzy": 0.0,
        }

    # ----------------------------
    # Prompting (CHAT)
    # ----------------------------
    def _build_prompt(self, ocr_text: str, fields: List[str]) -> List[Dict[str, str]]:
        """
        Construct chat messages.
        The model must output strict JSON only, formatted as a single line.
        """
        fields_str = ", ".join(fields)

        system_message = (
            # "/no_think \n"
            "You are an information extraction engine.\n"
            "Your task is to read OCR text from a document and extract specific fields.\n"
            "You must output ONLY one JSON object, with EXACTLY the requested keys.\n"
            "Rules:\n"
            "  - The JSON must be on a single line (no line breaks or indentation).\n"
            "  - Each requested key MUST be present in the JSON.\n"
            "  - If a field appears multiple times, use a JSON array of unique values in reading order.\n"
            "  - If a field is not present in the OCR text, set its value to null.\n"
            "  - Do NOT add any keys that were not requested.\n"
            "  - Do NOT output any explanations, comments, or text outside the JSON object.\n"
        )

        user_task = (
            "Extract the requested keys from the OCR\n"
            "OCR:\n"
            f"{ocr_text}\n"
            "\n"
            "Requested keys:\n"
            f"{fields_str}\n"
            "\n"
            "Output JSON (single line, no extra text):"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_task},
        ]

    # ----------------------------
    # Utilities: loading & parsing
    # ----------------------------
    def _pick_jsonl(self, main_dir: str) -> Union[str, None]:
        gz = os.path.join(main_dir, "dataset.jsonl.gz")
        jl = os.path.join(main_dir, "dataset.jsonl")
        if os.path.isfile(gz):
            return gz
        if os.path.isfile(jl):
            return jl
        return None

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
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

    def _extract_ocr_text(
            self,
            ex: Dict[str, Any],
            max_chars: int = 14000,
            *,
            mode: str = "page",  # "page" or "item"
            prefer_levels: Tuple[str, ...] = ("blocks", "paragraphs", "lines"),
            dedupe: bool = True,
            dedupe_scope: str = "document",  # "document" or "page"
            add_page_headers: bool = True,
    ) -> str:
        """
        Production-ready OCR-to-context renderer for LLM IE.
        """
        ocr = ex.get("ocr") or {}
        if not isinstance(ocr, dict):
            s = str(ocr)
            return (s[:max_chars] + " …") if len(s) > max_chars else s

        pages = ocr.get("pages")
        if not isinstance(pages, list) or not pages:
            t = " ".join(str(ocr.get("text", "")).split())
            return (t[:max_chars] + " …") if len(t) > max_chars else t

        out_parts: List[str] = []
        total = 0

        def add(s: str) -> bool:
            nonlocal total
            if not s:
                return True
            if total + len(s) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    out_parts.append(s[:remaining] + " …")
                return False
            out_parts.append(s)
            total += len(s)
            return True

        def pick_items(page: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            for key in prefer_levels:
                v = page.get(key)
                if isinstance(v, list) and v:
                    items = [
                        it for it in v
                        if isinstance(it, dict) and isinstance(it.get("text"), str) and it.get("text").strip()
                    ]
                    if items:
                        return items
            return None

        def sort_reading(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            def key_fn(it: Dict[str, Any]):
                bbox = it.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    return (y0, x0, y1, x1)
                return (1e9, 1e9, 1e9, 1e9)

            return sorted(items, key=key_fn)

        # Dedupe state
        seen_doc = set()

        for pi, p in enumerate(pages, start=1):
            if not isinstance(p, dict):
                continue

            items = pick_items(p)
            if not items:
                continue
            items = sort_reading(items)

            seen_page = set()

            if add_page_headers:
                if not add(f"[PAGE {pi}]\n"):
                    break

            if mode == "page":
                buf: List[str] = []
                for it in items:
                    text = " ".join(it.get("text", "").split())
                    if not text:
                        continue

                    if dedupe:
                        sig = text.casefold()
                        if dedupe_scope == "page":
                            if sig in seen_page:
                                continue
                            seen_page.add(sig)
                        else:
                            if sig in seen_doc:
                                continue
                            seen_doc.add(sig)

                    buf.append(text)

                chunk = " ".join(buf).strip()
                if chunk:
                    if not add(chunk + "\n"):
                        break

            else:  # mode == "item"
                for it in items:
                    text = " ".join(it.get("text", "").split())
                    if not text:
                        continue

                    if dedupe:
                        sig = text.casefold()
                        if dedupe_scope == "page":
                            if sig in seen_page:
                                continue
                            seen_page.add(sig)
                        else:
                            if sig in seen_doc:
                                continue
                            seen_doc.add(sig)

                    if not add(text + "\n"):
                        break
                else:
                    continue
                break

        return "".join(out_parts)

    def _extract_gold_fields(self, ex: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        """
        Parse annotations into {field_name: value or [values]}.
        """
        ann = ex.get("annotations")
        if ann is None:
            return {}

        def spans_to_values(spans) -> List[str]:
            vals: List[str] = []
            if not isinstance(spans, list):
                return vals

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

        cleaned: Dict[str, Union[str, List[str]]] = {}
        for k, v in out.items():
            if isinstance(v, list):
                seen: List[str] = []
                for s in v:
                    s2 = " ".join(s.split())
                    s2 = self._collapse_repeated_runs(s2)
                    if s2 and s2 not in seen:
                        seen.append(s2)
                if len(seen) == 0:
                    continue
                if len(seen) == 1:
                    cleaned[k] = seen[0]
                else:
                    cleaned[k] = seen
            else:
                s2 = " ".join(str(v).split())
                s2 = self._collapse_repeated_runs(s2)
                cleaned[k] = s2
        return cleaned

    # ----------------------------
    # Utilities: helpers
    # ----------------------------
    @staticmethod
    def _span_text(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            t = x.get("text")
            return t if isinstance(t, str) else ""
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
            return x[0]
        return ""

    @staticmethod
    def _collapse_repeated_runs(s: str, max_k: int = 8) -> str:
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

    def _safe_json_loads(self, s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            try:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start: end + 1])
            except Exception:
                pass
        return {}

    def _to_list_of_str(self, v: Union[str, List[Any], None]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


if __name__ == "__main__":
    task = InfoExtractionTask(base_path="../../datasets/vrdu-main", seed=42)
    prompts, references = task.generate_prompts(num_examples=3)
    for i in range(len(prompts)):
        print(f"Prompt {i + 1}:\n{prompts[i]}\n")
        print(f"Reference {i + 1}:\n{references[i]}\n")