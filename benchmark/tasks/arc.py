from __future__ import annotations

from typing import List, Tuple, Dict
import random
import re

from datasets import load_dataset

from benchmark.tasks.base_task import BaseTask


class ARCTask(BaseTask):
    """ARC (AI2 Reasoning Challenge) task wrapper for scientific reasoning.
    
    ARC tests a model's ability to answer grade-school science questions.
    Includes both ARC-Easy and ARC-Challenge variants.

    Parameters
    ----------
    variant : str
        Which ARC variant to use ("ARC-Easy" or "ARC-Challenge").
    split : str
        HF split to evaluate on ("test" or "validation").
    seed : int
        RNG seed for reproducibility.
    lazy : bool, default True
        Stream datasets so we don't download everything upfront.
    cache_dir : str | None
        Optional HF cache directory.
    """

    LETTERS = ["A", "B", "C", "D"]

    def __init__(
        self,
        variant: str = "ARC-Challenge",
        split: str = "test",
        seed: int = 42,
        *,
        lazy: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.variant = variant
        self.split = split
        self.cache_dir = cache_dir
        self.lazy = lazy

        if variant not in ["ARC-Easy", "ARC-Challenge"]:
            raise ValueError("variant must be 'ARC-Easy' or 'ARC-Challenge'")

        if not self.lazy:
            # Eagerly load dataset into memory
            self.dataset = list(load_dataset("allenai/ai2_arc", variant, split=split, cache_dir=cache_dir))
            if not self.dataset:
                raise ValueError("No ARC examples loaded – verify dataset availability.")
        else:
            self.dataset = None

    def _load_data(self) -> List[Dict]:
        """Load all examples from the dataset."""
        if not self.lazy:
            return self.dataset
        return list(
            load_dataset(
                "allenai/ai2_arc", self.variant, split=self.split, streaming=True, cache_dir=self.cache_dir
            )
        )

    def _format_prompt(self, ex: Dict) -> Tuple[str, str]:
        """Format an ARC example into a prompt and answer."""
        question = ex["question"]
        choices = ex["choices"]["text"]
        choice_labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        # Map answer key to our standard A, B, C, D format
        answer_idx = choice_labels.index(answer_key)
        answer = self.LETTERS[answer_idx]

        # Pad choices to exactly 4 options if needed
        while len(choices) < 4:
            choices.append("Not applicable")

        # SYSTEM instruction
        system_message = (
            "You are a science question-answering assistant. Answer grade-school level science questions "
            "by choosing the correct answer from A, B, C, or D. Respond with only the letter. "
            "Do not include any additional text or reasoning.\n"
        )

        # EXAMPLES block (5-shot science examples)
        demo_block = (
            "### EXAMPLES\n"

            "Question:\n"
            "What is the primary source of energy for most ecosystems on Earth?\n"
            "Choices:\n"
            "A. Wind\n"
            "B. The Sun\n"
            "C. Water\n"
            "D. Soil\n"
            "Answer:\n"
            "B\n\n"

            "Question:\n"
            "Which of these is a renewable resource?\n"
            "Choices:\n"
            "A. Coal\n"
            "B. Oil\n"
            "C. Natural gas\n"
            "D. Solar energy\n"
            "Answer:\n"
            "D\n\n"

            "Question:\n"
            "What happens to water when it freezes?\n"
            "Choices:\n"
            "A. It becomes lighter\n"
            "B. It expands\n"
            "C. It becomes denser\n"
            "D. It changes color\n"
            "Answer:\n"
            "B\n\n"

            "Question:\n"
            "Which part of a plant is responsible for photosynthesis?\n"
            "Choices:\n"
            "A. Roots\n"
            "B. Stem\n"
            "C. Leaves\n"
            "D. Flowers\n"
            "Answer:\n"
            "C\n\n"

            "Question:\n"
            "What force causes objects to fall toward Earth?\n"
            "Choices:\n"
            "A. Magnetism\n"
            "B. Gravity\n"
            "C. Friction\n"
            "D. Electricity\n"
            "Answer:\n"
            "B\n\n"
        )

        # INSTRUCTION
        instruction = "### INSTRUCTION\nChoose the correct answer to the science question.\n\n"

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

    def generate_prompts(self, num_examples: int) -> Tuple[List[str], List[str]]:
        """Generate prompts and reference answers."""
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")

        # Load all data
        data = self._load_data()
        
        if num_examples > len(data):
            raise ValueError(
                f"Requested {num_examples} examples but only {len(data)} available in {self.variant} {self.split} split."
            )

        # Sample the requested number of examples
        sampled_data = self.rng.sample(data, num_examples)
        
        prompts, refs = [], []
        for example in sampled_data:
            prompt, ref = self._format_prompt(example)
            prompts.append(prompt)
            refs.append(ref)

        return prompts, refs

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate accuracy metric."""
        # Extract the first letter A-D from the generated response
        m = re.match(r"\s*([A-D])", str(generated).strip(), flags=re.I)
        pred = m.group(1).upper() if m else ""
        return {"accuracy": float(pred == str(reference).strip().upper())}


# Test the implementation
if __name__ == "__main__":
    import time
    try:
        start = time.time()
        task = ARCTask(variant="ARC-Challenge", split="test")
        prompts, refs = task.generate_prompts(3)
        elapsed = time.time() - start
        print(f"Generated {len(prompts)} ARC prompts in {elapsed:.2f} seconds.")
        for i, (prompt, ref) in enumerate(zip(prompts[:2], refs[:2])):
            print(f"Prompt {i+1}:\n{prompt}\n")
            print(f"Reference: {ref}\n" + "="*50 + "\n")
    except Exception as e:
        print("Error:", e)