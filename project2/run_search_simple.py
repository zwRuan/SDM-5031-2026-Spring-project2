"""Simplified EoH-style search for augmentation strategies.

This script directly calls the LLM API to generate augmentation functions,
evaluates them, and keeps the best ones. Much simpler than the full LLM4AD framework.

Usage:
    cd project2
    python run_search_simple.py
"""
import sys
import os
import json
import time
import traceback
import http.client
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POMO_DIR = os.path.join(PROJECT_ROOT, "TSP", "POMO")
TSP_DIR = os.path.join(PROJECT_ROOT, "TSP")

sys.path.insert(0, POMO_DIR)
sys.path.insert(0, TSP_DIR)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from evaluation import (
    MODEL_PARAMS, CHECKPOINT_PATH, VAL_DATA_PATH,
    evaluate_with_custom_aug, normalize_to_unit_square, EVAL_AUG_FACTOR
)
from TSPModel import TSPModel as Model
from tsplib_utils import TSPLIBReader, tsplib_cost

# ---- LLM Configuration ----
LLM_HOST = "pre-openai-keys.alibaba-inc.com"
LLM_KEY = "aib_AIB_Marco_LLM_5f4ced"
LLM_MODEL = "gpt-4o-mini"

# ---- Search Configuration ----
MAX_ITERATIONS = 30
AUG_FACTOR = 32  # Use 32 for fast search, increase for final eval
LOG_FILE = os.path.join(os.path.dirname(__file__), "search_log.json")

# ---- Baseline result ----
BASELINE_AVG_GAP = 0.6237  # Standard 8-fold dihedral

SYSTEM_PROMPT = """You are an expert algorithm designer for combinatorial optimization.
Your task is to design a Python function `generate_augmentations` that creates diverse
geometric transformations of 2D coordinates for a TSP (Traveling Salesman Problem) solver.

The POMO solver runs independently on each augmented copy and picks the shortest tour.
More diverse transformations = better solutions.

Rules:
- Input: coords (1, num_nodes, 2) tensor in [0,1], aug_factor (int)
- Output: (aug_factor, num_nodes, 2) tensor in [0,1]
- Use PyTorch operations only (vectorized, no for-loops over nodes)
- After any transformation, re-normalize to [0,1] using min-max scaling
- You can use: rotations, reflections, scaling, shearing, etc.
- The standard POMO baseline uses 8 dihedral group transformations (4 rotations + 4 reflections)
- Your goal is to BEAT this baseline by providing more diverse augmentations

Important: numpy is available as `np`, torch is available as `torch`.
Return ONLY the Python function, no explanation."""

def call_llm(prompt, max_retries=10):
    """Call LLM API with retry logic."""
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection(LLM_HOST, timeout=60)
            payload = json.dumps({
                'model': LLM_MODEL,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 2048,
                'temperature': 0.8,
            })
            headers = {
                'Authorization': f'Bearer {LLM_KEY}',
                'Content-Type': 'application/json'
            }
            conn.request('POST', '/v1/chat/completions', payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode('utf-8'))
            conn.close()

            if 'choices' in data:
                return data['choices'][0]['message']['content']
            else:
                print(f"  [LLM] Non-200 response (attempt {attempt+1}): {str(data)[:100]}")
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"  [LLM] Error (attempt {attempt+1}): {e}")
            time.sleep(3 * (attempt + 1))
    return None


def extract_function(response_text):
    """Extract the generate_augmentations function from LLM response."""
    if response_text is None:
        return None

    # Try to find function definition
    lines = response_text.split('\n')
    func_lines = []
    in_func = False
    indent_level = None

    for line in lines:
        if 'def generate_augmentations' in line:
            in_func = True
            indent_level = len(line) - len(line.lstrip())
            func_lines.append(line[indent_level:])
        elif in_func:
            if line.strip() == '' or line.strip().startswith('#'):
                func_lines.append(line[indent_level:] if len(line) > indent_level else line)
            elif len(line) - len(line.lstrip()) > indent_level or line.strip() == '':
                func_lines.append(line[indent_level:] if len(line) > indent_level else line)
            elif line.strip().startswith('```'):
                break
            else:
                # Check if this is still inside the function
                stripped = line.lstrip()
                if stripped.startswith('def ') or stripped.startswith('class '):
                    break
                func_lines.append(line[indent_level:] if len(line) > indent_level else line)

    if not func_lines:
        # Try extracting from code block
        if '```' in response_text:
            code_blocks = response_text.split('```')
            for i, block in enumerate(code_blocks):
                if i % 2 == 1:  # inside code block
                    if block.startswith('python'):
                        block = block[6:]
                    if 'def generate_augmentations' in block:
                        return block.strip()
        return None

    return '\n'.join(func_lines)


def compile_and_test(func_code):
    """Compile function code and return callable, or None on failure."""
    full_code = f"import torch\nimport numpy as np\nimport math\n\n{func_code}"
    try:
        namespace = {}
        exec(full_code, namespace)
        func = namespace.get('generate_augmentations')
        if func is None:
            return None

        # Quick sanity test
        test_coords = torch.rand(1, 50, 2)
        result = func(test_coords, 8)
        if result is None or result.dim() != 3 or result.size(0) != 8 or result.size(2) != 2:
            return None
        if torch.isnan(result).any() or torch.isinf(result).any():
            return None
        return func
    except Exception as e:
        return None


def evaluate_function(model, device, func, aug_factor):
    """Evaluate augmentation function on validation set. Returns avg_gap."""
    gaps = []
    for root, _, files in os.walk(VAL_DATA_PATH):
        for file in sorted(files):
            if not file.endswith('.tsp'):
                continue
            full_path = os.path.join(root, file)
            name, dimension, locs, ew_type = TSPLIBReader(full_path)
            if name is None:
                continue
            optimal = tsplib_cost.get(name, None)
            if optimal is None:
                continue

            coords_orig_np = np.array(locs, dtype=np.float32)
            coords_orig = torch.from_numpy(coords_orig_np).to(device)
            node_coord = coords_orig[None, :, :]
            nodes_xy = normalize_to_unit_square(node_coord)

            try:
                score = evaluate_with_custom_aug(
                    model, device, nodes_xy, coords_orig, ew_type, func, aug_factor
                )
                gap = (score - optimal) / optimal * 100
                gaps.append(gap)
            except Exception:
                gaps.append(100.0)

    return float(np.mean(gaps)) if gaps else 100.0


def main():
    print("=" * 70)
    print("Project 2: Simplified EoH Search for Augmentation Strategy")
    print("=" * 70)
    print(f"  LLM: {LLM_MODEL} @ {LLM_HOST}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Aug factor: {AUG_FACTOR}")
    print(f"  Baseline: {BASELINE_AVG_GAP:.4f}%")
    print()

    # Setup
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    model = Model(**MODEL_PARAMS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load existing best
    from best_algorithm import generate_augmentations as current_best
    best_gap = evaluate_function(model, device, current_best, AUG_FACTOR)
    best_code = open(os.path.join(os.path.dirname(__file__), 'best_algorithm.py')).read()
    print(f"  Current best (golden-dihedral): {best_gap:.4f}%")
    print()

    # Search log
    search_log = []

    for iteration in range(MAX_ITERATIONS):
        print(f"[Iter {iteration+1}/{MAX_ITERATIONS}] ", end="", flush=True)

        # Construct prompt with feedback
        if iteration == 0:
            prompt = (
                "Design a `generate_augmentations(coords, aug_factor)` function.\n"
                "The current best approach uses 8 standard dihedral transforms + additional "
                "rotations at golden-angle spacing (~137.5°), with dihedral reflections applied "
                "to each rotation. This achieves avg_gap=0.54% with aug_factor=32.\n\n"
                "Try to improve by:\n"
                "1. Using different angle distributions (not just golden angle)\n"
                "2. Adding slight random perturbations/jitter to coordinates\n"
                "3. Anisotropic scaling before rotation\n"
                "4. Combining multiple transformation types\n"
                "5. Instance-adaptive strategies based on coordinate statistics\n\n"
                "The function must work with aug_factor=32."
            )
        else:
            # Use feedback from previous attempts
            recent = search_log[-min(3, len(search_log)):]
            feedback = "\n".join([
                f"- Attempt {s['iter']}: avg_gap={s['gap']:.4f}% "
                f"({'BETTER' if s['gap'] < best_gap else 'WORSE'})"
                for s in recent
            ])
            prompt = (
                f"Design a NEW `generate_augmentations(coords, aug_factor)` function.\n\n"
                f"Current best avg_gap: {best_gap:.4f}% (lower is better).\n"
                f"Recent attempts:\n{feedback}\n\n"
                f"The baseline uses 8 dihedral transforms + golden-angle rotations.\n"
                f"Think creatively! Try completely different approaches:\n"
                f"- Non-uniform rotation angles based on coordinate statistics\n"
                f"- Combining rotation with slight coordinate perturbation\n"
                f"- Using both rotation and shearing transforms\n"
                f"- Adaptive strategies that analyze the point distribution\n"
                f"- Different normalization approaches after transformation\n\n"
                f"The function must return (aug_factor, num_nodes, 2) tensor in [0,1]."
            )

        # Call LLM
        response = call_llm(prompt)
        if response is None:
            print("LLM call failed, skipping")
            continue

        # Extract and compile function
        func_code = extract_function(response)
        if func_code is None:
            print("failed to extract function")
            continue

        func = compile_and_test(func_code)
        if func is None:
            print("compilation/test failed")
            continue

        # Evaluate
        try:
            gap = evaluate_function(model, device, func, AUG_FACTOR)
        except Exception as e:
            print(f"eval error: {e}")
            continue

        improved = gap < best_gap - 0.001
        print(f"avg_gap={gap:.4f}% {'✓ NEW BEST!' if improved else ''}")

        search_log.append({
            'iter': iteration + 1,
            'gap': gap,
            'code': func_code,
            'timestamp': datetime.now().isoformat(),
        })

        if improved:
            best_gap = gap
            best_code = func_code
            # Save best to file
            with open(os.path.join(os.path.dirname(__file__), 'best_algorithm_candidate.py'), 'w') as f:
                f.write(f'"""LLM-discovered augmentation (gap={gap:.4f}%, iter={iteration+1})"""\n')
                f.write('import torch\nimport numpy as np\nimport math\n\n')
                f.write(func_code)
            print(f"  >> Saved to best_algorithm_candidate.py")

        # Save log periodically
        if (iteration + 1) % 5 == 0:
            with open(LOG_FILE, 'w') as f:
                json.dump(search_log, f, indent=2, default=str)

        # Rate limit protection
        time.sleep(2)

    # Final save
    with open(LOG_FILE, 'w') as f:
        json.dump(search_log, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"Search complete! Best avg_gap: {best_gap:.4f}%")
    print(f"Baseline: {BASELINE_AVG_GAP:.4f}%")
    print(f"Improvement: {(BASELINE_AVG_GAP - best_gap) / BASELINE_AVG_GAP * 100:.1f}%")
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'best_algorithm_candidate.py')):
        print(f"Best candidate saved to: best_algorithm_candidate.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
