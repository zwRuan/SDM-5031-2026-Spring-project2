"""Extract the best augmentation function from EoH logs and write to best_algorithm.py.

Usage:
    cd project2
    python extract_best.py --log_dir ./logs/<your_run_dir>

This script parses EoH profiler logs, finds the function with the highest fitness,
and writes it to best_algorithm.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def find_best_function(log_dir: str) -> tuple[str | None, float]:
    """Parse EoH logs and find the best function.

    Returns:
        (function_source_code, fitness_score)
    """
    best_fitness = -float('inf')
    best_code = None

    for root, _, files in os.walk(log_dir):
        for file in sorted(files):
            full_path = os.path.join(root, file)

            if file.endswith('.json'):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            score = entry.get('score') or entry.get('fitness')
                            code = entry.get('code') or entry.get('program')
                            if score is not None and code is not None:
                                if float(score) > best_fitness:
                                    best_fitness = float(score)
                                    best_code = code
                    elif isinstance(data, dict):
                        for key, val in data.items():
                            if isinstance(val, dict):
                                score = val.get('score') or val.get('fitness')
                                code = val.get('code') or val.get('program')
                                if score is not None and code is not None:
                                    if float(score) > best_fitness:
                                        best_fitness = float(score)
                                        best_code = code
                except (json.JSONDecodeError, ValueError):
                    pass

            elif file.endswith('.txt') or file.endswith('.log'):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    pattern = r'[Ss]core[:\s]+(-?[\d.]+).*?```(?:python)?\s*\n(.*?)```'
                    matches = re.findall(pattern, content, re.DOTALL)
                    for score_str, code in matches:
                        try:
                            score = float(score_str)
                            if score > best_fitness:
                                best_fitness = score
                                best_code = code
                        except ValueError:
                            pass
                except Exception:
                    pass

    for root, _, files in os.walk(log_dir):
        for file in sorted(files):
            if file.startswith('best_') and file.endswith('.py'):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    code = f.read()
                match = re.search(r'fitness[:\s=]+(-?[\d.]+)', code)
                if match:
                    score = float(match.group(1))
                    if score > best_fitness:
                        best_fitness = score
                        best_code = code

    return best_code, best_fitness


def write_best_algorithm(code: str, fitness: float, output_path: str):
    """Write the best function to best_algorithm.py."""
    header = f'''"""Best LLM-designed augmentation strategy discovered by EoH.

Fitness (negative avg_aug_gap): {fitness:.6f}
Estimated avg_aug_gap: {-fitness:.4f}%

This function was automatically extracted from EoH logs.
The function signature: generate_augmentations(coords, aug_factor) -> augmented
"""
import torch
import numpy as np

'''
    if 'import torch' not in code:
        full_code = header + code
    else:
        full_code = (
            f'"""Best LLM-designed augmentation strategy discovered by EoH.\n\n'
            f'Fitness: {fitness:.6f}\n'
            f'Estimated avg_aug_gap: {-fitness:.4f}%\n"""\n'
            + code
        )

    with open(output_path, 'w') as f:
        f.write(full_code)

    print(f"Best function written to: {output_path}")
    print(f"Fitness: {fitness:.6f} (avg_aug_gap ~ {-fitness:.4f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract best function from EoH logs")
    parser.add_argument("--log_dir", required=True, help="Path to EoH log directory")
    parser.add_argument("--output", default="best_algorithm.py", help="Output file path")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"ERROR: Log directory not found: {args.log_dir}")
        sys.exit(1)

    code, fitness = find_best_function(args.log_dir)
    if code is None:
        print("ERROR: No valid function found in logs!")
        print("The log directory might be empty or in an unexpected format.")
        print("You may need to manually extract the best function from the logs.")
        sys.exit(1)

    write_best_algorithm(code, fitness, args.output)


if __name__ == "__main__":
    main()