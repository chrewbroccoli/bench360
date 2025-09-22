from __future__ import annotations

from typing import List, Tuple, Dict
import random
import re
from itertools import cycle

from datasets import load_dataset

from benchmark.tasks.base_task import BaseTask


class HellaSwagTask(BaseTask):
    """HellaSwag task wrapper for commonsense natural language inference.
    
    HellaSwag tests a model's ability to complete sentences with the most
    plausible continuation from multiple choices.

    Parameters
    ----------
    split : str
        HF split to evaluate on ("validation" or "test").
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
        split: str = "validation",
        seed: int = 42,
        *,
        lazy: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.split = split
        self.cache_dir = cache_dir
        self.lazy = lazy

        if not self.lazy:
            # Eagerly load dataset into memory
            self.dataset = list(load_dataset("Rowan/hellaswag", split=split, cache_dir=cache_dir))
            if not self.dataset:
                raise ValueError("No HellaSwag examples loaded – verify dataset availability.")
        else:
            self.dataset = None

    def _load_data(self) -> List[Dict]:
        """Load all examples from the dataset."""
        if not self.lazy:
            return self.dataset
        return list(
            load_dataset(
                "Rowan/hellaswag", split=self.split, streaming=True, cache_dir=self.cache_dir
            )
        )

    def _format_prompt(self, ex: Dict) -> Tuple[str, str]:
        """Format a HellaSwag example into a prompt and answer."""
        context = ex["ctx"]
        endings = ex["endings"]
        answer = self.LETTERS[int(ex["label"])]

        # SYSTEM instruction
        system_message = (
            "You are a commonsense reasoning assistant. Choose the most plausible completion "
            "from A, B, C, or D. Respond with only the letter. Do not include any additional text or reasoning.\n"
        )

        # EXAMPLES block (5-shot examples)
        demo_block = (
            "### EXAMPLES\n"

            "Context:\n"
            "A woman is outside with a bucket and a dog. The dog is running around trying to avoid getting a bath. She\n"
            "Completions:\n"
            "A. rinses the bucket off with a hose and fills it with bubble solution.\n"
            "B. uses a hose to keep it from getting soapy.\n" 
            "C. gets the dog wet, then it runs away again.\n"
            "D. gets into the bath tub with the dog.\n"
            "Answer:\n"
            "C\n\n"

            "Context:\n"
            "A man is holding a small blade in his hand. He\n"
            "Completions:\n"
            "A. is shaving his beard in front of a mirror.\n"
            "B. uses the blade to cut several different fruits.\n"
            "C. puts on a hat and walks outside.\n"
            "D. starts dancing with the blade.\n"
            "Answer:\n"
            "A\n\n"

            "Context:\n"
            "Several people are playing a game of badminton in a gym. They\n"
            "Completions:\n"
            "A. are jumping and hitting the birdie back and forth over the net.\n"
            "B. walk over to a table and start eating lunch.\n"
            "C. put their rackets down and leave the gym.\n"
            "D. start playing basketball instead.\n"
            "Answer:\n"
            "A\n\n"

            "Context:\n"
            "A chef is preparing ingredients in a restaurant kitchen. She\n"
            "Completions:\n"
            "A. chops vegetables and adds them to a pot on the stove.\n"
            "B. leaves the kitchen and goes home.\n"
            "C. starts cleaning the dining room tables.\n"
            "D. begins painting the walls.\n"
            "Answer:\n"
            "A\n\n"

            "Context:\n"
            "Two children are riding bicycles down a hill. They\n"
            "Completions:\n"
            "A. stop and start walking back up the hill.\n"
            "B. pedal faster to control their speed going downhill.\n"
            "C. get off their bikes and start running.\n"
            "D. throw their bikes in a nearby lake.\n"
            "Answer:\n"
            "B\n\n"
        )

        # INSTRUCTION
        instruction = "### INSTRUCTION\nChoose the most plausible completion.\n\n"

        # INPUT
        input_block = (
            "### INPUT\n"
            f"Context:\n{context}\n"
            "Completions:\n" + "\n".join(f"{l}. {ending}" for l, ending in zip(self.LETTERS, endings)) + "\n"
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
                f"Requested {num_examples} examples but only {len(data)} available in {self.split} split."
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
        task = HellaSwagTask(split="validation")
        prompts, refs = task.generate_prompts(5)
        elapsed = time.time() - start
        print(f"Generated {len(prompts)} HellaSwag prompts in {elapsed:.2f} seconds.")
        for i, (prompt, ref) in enumerate(zip(prompts[:2], refs[:2])):
            print(f"Prompt {i+1}:\n{prompt}\n")
            print(f"Reference: {ref}\n" + "="*50 + "\n")
    except Exception as e:
        print("Error:", e)