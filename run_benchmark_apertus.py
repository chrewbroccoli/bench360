#!/usr/bin/env python3
"""
Script to run multiple benchmarks (MMLU, HellaSwag, ARC) against Apertus server.
"""
import json
import requests
import time
import argparse
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Set up Hugging Face authentication
if "HF_TOKEN" in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    # Try to login, but continue if it fails
    try:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        print("Hugging Face authentication successful")
    except Exception as e:
        print(f"Warning: Hugging Face authentication failed: {e}")
        print("Continuing without authentication...")

# Add benchmark module to path
sys.path.append(str(Path(__file__).parent / "benchmark"))
from tasks.mmlu import MMLUTask
from tasks.hellaswag import HellaSwagTask
from tasks.arc import ARCTask


class ApertusClient:
    """Client for Apertus OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from Apertus model"""
        payload = {
            "model": "adamo1139/Apertus-8B-Instruct-2509-ungated",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


def get_task_instance(benchmark: str, **kwargs):
    """Get task instance based on benchmark name."""
    if benchmark.lower() == "mmlu":
        return MMLUTask(subjects=kwargs.get("subjects"), split="test", seed=42)
    elif benchmark.lower() == "hellaswag":
        return HellaSwagTask(split="validation", seed=42)
    elif benchmark.lower() == "arc-easy":
        return ARCTask(variant="ARC-Easy", split="test", seed=42)
    elif benchmark.lower() == "arc-challenge":
        return ARCTask(variant="ARC-Challenge", split="test", seed=42)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def run_benchmark(
    client: ApertusClient,
    benchmark: str,
    num_examples: int = None,
    subjects: List[str] = None,
    output_file: str = None
) -> Dict[str, Any]:
    """Run specified benchmark"""
    
    # Set default num_examples based on benchmark
    if num_examples is None:
        defaults = {
            "mmlu": 14042,
            "hellaswag": 10042,
            "arc-easy": 2376,
            "arc-challenge": 1172
        }
        num_examples = defaults.get(benchmark.lower(), 100)
    
    # Initialize task
    task = get_task_instance(benchmark, subjects=subjects)
    
    print(f"Generating {num_examples} {benchmark.upper()} prompts...")
    prompts, references = task.generate_prompts(num_examples)
    print(f"Generated {len(prompts)} prompts")
    
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i, (prompt, reference) in enumerate(zip(prompts, references)):
        print(f"Processing example {i+1}/{len(prompts)}", end="\r")
        
        try:
            # Get model response
            response = client.generate(prompt, max_tokens=10)  # Short response for multiple choice
            
            # Evaluate
            metrics = task.quality_metrics(response, reference)
            accuracy = metrics["accuracy"]
            
            if accuracy == 1.0:
                correct += 1
            total += 1
            
            results.append({
                "example_id": i,
                "prompt": prompt,
                "reference": reference,
                "generated": response,
                "metrics": metrics
            })
            
            # Brief pause to avoid overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError processing example {i+1}: {e}")
            results.append({
                "example_id": i,
                "prompt": prompt,
                "reference": reference,
                "generated": None,
                "error": str(e),
                "metrics": {"accuracy": 0.0}
            })
            total += 1
    
    end_time = time.time()
    
    # Calculate final metrics
    overall_accuracy = correct / total if total > 0 else 0
    duration = end_time - start_time
    
    summary = {
        "benchmark": benchmark,
        "total_examples": total,
        "correct": correct,
        "accuracy": overall_accuracy,
        "duration_seconds": duration,
        "examples_per_second": total / duration if duration > 0 else 0,
        "subjects": subjects if benchmark.lower() == "mmlu" else None,
        "results": results
    }
    
    print(f"\n\nResults:")
    print(f"Accuracy: {overall_accuracy:.3f} ({correct}/{total})")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Speed: {total/duration:.1f} examples/second")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks against Apertus server")
    parser.add_argument("benchmark", choices=["mmlu", "hellaswag", "arc-easy", "arc-challenge"],
                       help="Benchmark to run")
    parser.add_argument("--server-url", default="http://localhost:8000", 
                       help="Apertus server URL (default: http://localhost:8000)")
    parser.add_argument("--num-examples", type=int,
                       help="Number of examples to test (default: full dataset)")
    parser.add_argument("--subjects", nargs="*", 
                       help="MMLU subjects to test (only for MMLU benchmark)")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize client
    client = ApertusClient(args.server_url)
    
    # Check server health
    print(f"Checking server at {args.server_url}...")
    if not client.health_check():
        print(f"Error: Cannot connect to Apertus server at {args.server_url}")
        print("Make sure the server is running and accessible.")
        sys.exit(1)
    
    print("Server is healthy!")
    
    # Run benchmark
    try:
        results = run_benchmark(
            client=client,
            benchmark=args.benchmark,
            num_examples=args.num_examples,
            subjects=args.subjects,
            output_file=args.output
        )
        
        print(f"\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()