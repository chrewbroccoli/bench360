from __future__ import annotations

from typing import List, Tuple, Dict, Iterable
import random
import re
from itertools import cycle

from datasets import load_dataset

from benchmark.tasks.base_task import BaseTask


class MMLUTask(BaseTask):
    """MMLU task wrapper with **lazy loading** and **near‑uniform sampling**.

    *Duplicates are **never** returned.*  If the caller requests more prompts
    than exist across the chosen subjects, the method raises ``ValueError``.

    Parameters
    ----------
    subjects : list[str] | None
        Which MMLU subjects to load. ``None`` ⇒ all 57.
    split : str
        HF split to evaluate on (``"test"`` recommended for benchmark reports).
    seed : int
        RNG seed for reproducibility.
    lazy : bool, default ``True``
        Stream datasets so we don’t download everything upfront.
    cache_dir : str | None
        Optional HF cache directory.
    """

    LETTERS = ["A", "B", "C", "D"]

    def __init__(
        self,
        subjects: List[str] | None = None,
        split: str = "test",
        seed: int = 42,
        *,
        lazy: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.subjects = subjects or self._all_subjects()
        self.split = split
        self.cache_dir = cache_dir
        self.lazy = lazy

        if not self.lazy:
            # Eagerly load selected subjects into memory
            self.by_subject: Dict[str, List[Dict]] = {
                sbj: list(load_dataset("cais/mmlu", sbj, split=split, cache_dir=cache_dir))
                for sbj in self.subjects
            }
            if not any(self.by_subject.values()):
                raise ValueError("No MMLU examples loaded – verify dataset availability.")
        else:
            self.by_subject = None  # type: ignore

    # ───────────────────────── subject utilities ─────────────────────────

    @staticmethod
    def _all_subjects() -> List[str]:
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
            "high_school_biology", "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
            "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
            "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology", "public_relations",
            "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
        ]

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _subject_rows(self, subject: str) -> List[Dict]:
        """Return every row for *subject* (cached in eager mode)."""
        if not self.lazy:
            return self.by_subject[subject]
        return list(
            load_dataset(
                "cais/mmlu", subject, split=self.split, streaming=True, cache_dir=self.cache_dir
            )
        )

    def _format_prompt(self, ex: Dict) -> Tuple[str, str]:
        question = ex["question"]
        choices = ex["choices"]
        answer = self.LETTERS[int(ex["answer"])]

        # SYSTEM instruction
        system_message = (
            "You are a multiple-choice question-answering assistant. Choose the correct answer "
            "from A, B, C, or D. Respond with only the letter. Do not include any additional text or reasoning.\n"
        )

        # EXAMPLES block (hardcoded single example)
        demo_block = (
            "### EXAMPLES\n"

            "Question:\n"
            "Which scientist proposed the law of universal gravitation?\n"
            "Choices:\n"
            "A. Isaac Newton\n"
            "B. Albert Einstein\n"
            "C. Galileo Galilei\n"
            "D. Johannes Kepler\n"
            "Answer:\n"
            "A\n\n"

            "Question:\n"
            "Which element has the chemical symbol “Na”?\n"
            "Choices:\n"
            "A. Nitrogen\n"
            "B. Sodium\n"
            "C. Neon\n"
            "D. Nickel\n"
            "Answer:\n"
            "B\n\n"

            "Question:\n"
            "What is the capital city of Australia?\n"
            "Choices:\n"
            "A. Canberra\n"
            "B. Melbourne\n"
            "C. Sydney\n"
            "D. Perth\n"
            "Answer:\n"
            "A\n\n"

            "Question:\n"
            "In DNA, which base pairs with cytosine?\n"
            "Choices:\n"
            "A. Adenine\n"
            "B. Uracil\n"
            "C. Thymine\n"
            "D. Guanine\n"
            "Answer:\n"
            "D\n\n"

            "Question:\n"
            "Who wrote the novel “Pride and Prejudice”?\n"
            "Choices:\n"
            "A. George Eliot\n"
            "B. Emily Brontë\n"
            "C. Jane Austen\n"
            "D. Virginia Woolf\n"
            "Answer:\n"
            "C\n\n"
        )


        # INSTRUCTION
        instruction = "### INSTRUCTION\nChoose the best answer.\n\n"

        # INPUT
        input_block = (
            "### INPUT\n"
            f"Question:\n{question}\n"
            "Choices:\n" + "\n".join(f"{l}. {c}" for l, c in zip(self.LETTERS, choices)) + "\n"
        )

        # OUTPUT
        output_block = "### OUTPUT\nAnswer:"

        full_prompt = (
            f"### SYSTEM\n{system_message}\n\n"
            f"{demo_block}"
            f"{instruction}"
            f"{input_block}"
            f"{output_block}"
        )
        return full_prompt, answer


    # ---------------------------------------------------------------------
    # BaseTask interface
    # ---------------------------------------------------------------------

    def generate_prompts(self, num_examples: int):
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        if not self.subjects:
            raise ValueError("No subjects specified")

        # ① Load rows for each subject
        pools: Dict[str, List[Tuple[str, str]]] = {}
        total_unique = 0
        for sbj in self.subjects:
            rows = self._subject_rows(sbj)
            formatted = [self._format_prompt(r) for r in rows]
            pools[sbj] = formatted
            total_unique += len(formatted)

        if num_examples > total_unique:
            raise ValueError(
                f"Requested {num_examples} examples but only {total_unique} unique items "
                "available across the selected subjects."
            )

        # ② Near‑uniform allocation (no replacement)
        n_subj = len(self.subjects)
        base = num_examples // n_subj
        quotas = {s: min(base, len(pools[s])) for s in self.subjects}
        allocated = sum(quotas.values())

        # Round‑robin distribute leftovers while respecting capacity
        remainder = num_examples - allocated
        rr = cycle(self.subjects)
        while remainder > 0:
            sbj = next(rr)
            if quotas[sbj] < len(pools[sbj]):
                quotas[sbj] += 1
                remainder -= 1
            # If no subject can take more, break to avoid infinite loop (shouldn't happen)
            if all(quotas[s] == len(pools[s]) for s in self.subjects):
                break

        # If still missing items → impossible under unique‑only constraint
        if sum(quotas.values()) < num_examples:
            raise ValueError(
                "Cannot satisfy the request without duplicates; fewer unique items than requested."
            )

        # ③ Sample according to quotas
        prompts, refs = [], []
        for sbj in self.subjects:
            k = quotas[sbj]
            if k == 0:
                continue
            chosen = self.rng.sample(pools[sbj], k)
            for p, r in chosen:
                prompts.append(p); refs.append(r)

        return prompts, refs

    def quality_metrics(self, generated: str, reference):
        m = re.match(r"\s*([A-D])", str(generated).strip(), flags=re.I)
        pred = m.group(1).upper() if m else ""
        return {"accuracy": float(pred == str(reference).strip().upper()[0])}


# ───────────────────────────── smoke‑test ─────────────────────────────
if __name__ == "__main__":
    import time
    try:
        start = time.time()
        task = MMLUTask(subjects=[], split="test")
        prompts, _ = task.generate_prompts(14_042)
        elapsed = time.time() - start
        print(f"Generated {len(prompts)} prompts in {elapsed:.2f} seconds.")
        for i, prompt in enumerate(prompts[:3]):
            print(f"Prompt {i+1}:\n{prompt}\n")
    except ValueError as e:
        print("Error:", e)
