"""Search for 10 different augmentation strategies that each improve >= 60% instances.

This script iteratively calls gpt-4o-mini to generate novel augmentation strategies,
evaluates each on the validation set, and keeps those that satisfy the requirement.
Continues until 10 valid strategies are found.

Usage:
    cd project2
    python search_10_methods.py
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
    evaluate_with_custom_aug, normalize_to_unit_square, standard_8fold_augment
)
from TSPModel import TSPModel as Model
from tsplib_utils import TSPLIBReader, tsplib_cost

# ---- LLM Configuration ----
LLM_HOST = "pre-openai-keys.alibaba-inc.com"
LLM_KEY = "aib_AIB_Marco_LLM_5f4ced"
LLM_MODEL = "gpt-4o-mini"

# ---- Search Configuration ----
TARGET_NUM_METHODS = 10
IMPROVEMENT_THRESHOLD = 0.6  # >= 60% instances improved
AUG_FACTOR = 256  # Use min(N*8, 800) in final, but 256 for search speed
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "found_methods.json")
CHECKPOINT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "methods")

# ---- Prompts ----
SYSTEM_PROMPT = """You are an expert algorithm designer for combinatorial optimization.
You design Python functions that generate geometric transformations of 2D coordinates 
for augmenting a neural TSP solver at test time.

Rules:
- Function signature: generate_augmentations(coords: torch.Tensor, aug_factor: int = 256) -> torch.Tensor
- Input: coords shape (1, num_nodes, 2), values in [0,1]
- Output: shape (aug_factor, num_nodes, 2), values in [0,1]
- Use only torch and numpy (import torch, import numpy as np)
- After any transformation, re-normalize to [0,1] using min-max scaling
- Must be deterministic (no random without fixed seeds)
- Use vectorized operations (for-loops over aug copies are OK, not over nodes)
- Always start with 8 standard dihedral transforms as the first 8 copies
- The remaining copies should be rotations/transforms that provide diverse viewpoints

The standard dihedral 8 transforms are:
(x,y), (1-x,y), (x,1-y), (1-x,1-y), (y,x), (1-y,x), (y,1-x), (1-y,1-x)

Return ONLY the Python function code, no explanations, no markdown code blocks."""


def call_llm(prompt, temperature=0.9, max_retries=15):
    """Call LLM API with retry logic."""
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection(LLM_HOST, timeout=90)
            payload = json.dumps({
                'model': LLM_MODEL,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 3000,
                'temperature': temperature,
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
                msg = data.get('message', data.get('error', {}).get('message', str(data)))
                print(f"    [LLM] API error (attempt {attempt+1}): {msg[:80]}")
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"    [LLM] Exception (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return None


def extract_function(response_text):
    """Extract the generate_augmentations function from LLM response."""
    if response_text is None:
        return None

    # Remove markdown code blocks if present
    text = response_text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first and last ``` lines
        start = 1
        end = len(lines)
        for i, line in enumerate(lines):
            if i > 0 and line.strip().startswith('```'):
                end = i
                break
        text = '\n'.join(lines[start:end])

    # Find the function definition
    lines = text.split('\n')
    func_lines = []
    in_func = False
    base_indent = 0

    for line in lines:
        if 'def generate_augmentations' in line:
            in_func = True
            base_indent = len(line) - len(line.lstrip())
            func_lines.append(line[base_indent:])
        elif in_func:
            if line.strip() == '':
                func_lines.append('')
            elif line.strip().startswith('#'):
                func_lines.append(line[base_indent:] if len(line) > base_indent else line)
            else:
                current_indent = len(line) - len(line.lstrip())
                if current_indent > base_indent or line.strip() == '':
                    func_lines.append(line[base_indent:] if len(line) > base_indent else line)
                elif line.strip().startswith('def ') or line.strip().startswith('class '):
                    break
                else:
                    func_lines.append(line[base_indent:] if len(line) > base_indent else line)

    if not func_lines:
        # Maybe the whole response is the function
        if 'def generate_augmentations' in text:
            return text
        return None

    return '\n'.join(func_lines)


def compile_function(func_code):
    """Compile function code and return callable, or None on failure."""
    # Add helper functions that might be needed
    full_code = "import torch\nimport numpy as np\nimport math\n\n" + func_code
    try:
        namespace = {}
        exec(full_code, namespace)
        func = namespace.get('generate_augmentations')
        if func is None:
            return None, "Function not found"

        # Sanity test with small input
        test_coords = torch.rand(1, 50, 2)
        result = func(test_coords, 16)
        if result is None:
            return None, "Function returned None"
        if result.dim() != 3:
            return None, f"Wrong dimensions: {result.dim()}"
        if result.size(0) != 16:
            return None, f"Wrong aug count: {result.size(0)} != 16"
        if result.size(2) != 2:
            return None, f"Wrong coord dim: {result.size(2)}"
        if torch.isnan(result).any() or torch.isinf(result).any():
            return None, "NaN/Inf in output"
        if result.min() < -0.01 or result.max() > 1.01:
            return None, f"Values out of range: [{result.min():.3f}, {result.max():.3f}]"

        return func, "OK"
    except Exception as e:
        return None, str(e)[:100]


def evaluate_method(model, device, func, aug_factor=256):
    """Evaluate augmentation function. Returns (avg_gap, per_instance_gaps, improved_count)."""
    instances = []
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
            instances.append((name, dimension, locs, ew_type, optimal))

    baseline_gaps = []
    method_gaps = []

    for name, dimension, locs, ew_type, optimal in instances:
        coords_orig = torch.from_numpy(np.array(locs, dtype=np.float32)).to(device)
        node_coord = coords_orig[None, :, :]
        nodes_xy = normalize_to_unit_square(node_coord)

        # Baseline: standard 8-fold
        score_base = evaluate_with_custom_aug(
            model, device, nodes_xy, coords_orig, ew_type, standard_8fold_augment, 8
        )
        gap_base = (score_base - optimal) / optimal * 100
        baseline_gaps.append(gap_base)

        # Method under test
        actual_aug = min(dimension * 8, aug_factor)
        score_method = evaluate_with_custom_aug(
            model, device, nodes_xy, coords_orig, ew_type, func, actual_aug
        )
        gap_method = (score_method - optimal) / optimal * 100
        method_gaps.append(gap_method)

    # Count improvements
    improved = sum(1 for b, m in zip(baseline_gaps, method_gaps) if m < b - 1e-6)
    total = len(baseline_gaps)

    return {
        'avg_gap': float(np.mean(method_gaps)),
        'baseline_avg_gap': float(np.mean(baseline_gaps)),
        'per_instance': {name: {'baseline': bg, 'method': mg}
                         for (name, _, _, _, _), bg, mg in zip(instances, baseline_gaps, method_gaps)},
        'improved_count': improved,
        'total_count': total,
        'improvement_ratio': improved / total if total > 0 else 0,
    }


def generate_prompt(found_methods, failed_attempts):
    """Generate a diverse prompt based on what we've already found."""
    if not found_methods and not failed_attempts:
        return (
            "Design a novel `generate_augmentations(coords, aug_factor)` function.\n\n"
            "Requirements:\n"
            "- Start with 8 standard dihedral transforms\n"
            "- Add extra rotations/transforms for the remaining aug_factor - 8 copies\n"
            "- Use a UNIQUE angle selection strategy (not simple evenly-spaced)\n\n"
            "Ideas to explore:\n"
            "- Halton quasi-random sequence for angles\n"
            "- Van der Corput sequence\n"
            "- Golden ratio based spacing\n"
            "- Prime-based angle distribution\n"
            "- Fibonacci lattice on the circle\n"
            "- Combining rotation with mild shearing\n\n"
            "aug_factor will typically be 256. Return only the function code."
        )

    found_names = [m['name'] for m in found_methods]
    failed_ideas = [f['idea'][:50] for f in failed_attempts[-5:]] if failed_attempts else []

    prompt = f"Design a NOVEL `generate_augmentations(coords, aug_factor)` function.\n\n"
    prompt += f"I already have {len(found_methods)} working methods:\n"
    for m in found_methods:
        prompt += f"  - {m['name']}: {m['idea'][:60]}\n"
    prompt += f"\nI need something DIFFERENT from all of the above.\n"

    if failed_ideas:
        prompt += f"\nRecent failed attempts (don't repeat these):\n"
        for idea in failed_ideas:
            prompt += f"  - {idea}\n"

    prompt += (
        "\nTry one of these NOVEL approaches:\n"
        "- Sobol sequence angles with base-3 Halton for diversity\n"
        "- Rotation angles from digits of pi or e\n"
        "- Angles based on prime number reciprocals (1/2, 1/3, 1/5, 1/7...)\n"
        "- Alternating rotation + slight anisotropic scale (stretch x or y by 0.95-1.05 before normalizing)\n"
        "- Rotation with progressive angular momentum (accelerating angle increments)\n"
        "- Using both base-2 and base-3 Halton sequences interleaved\n"
        "- Sinusoidal angle modulation\n"
        "- Rotation + transpose combinations (rotate, then swap x/y for some copies)\n"
        "- Multi-resolution: first half at fine angles, second half at coarse angles\n"
        "- Power-law distributed angles (more copies near 0/90/180/270, fewer in between)\n"
        "- Jittered regular grid: evenly-spaced angles + small deterministic offsets\n"
        "- Weyl sequence: angles = i * sqrt(2) mod 1 * 2*pi\n"
        "\nReturn ONLY the Python function. aug_factor defaults to 256."
    )
    return prompt


def main():
    print("=" * 70)
    print("Searching for 10 Valid Augmentation Strategies")
    print("=" * 70)
    print(f"  LLM: {LLM_MODEL} @ {LLM_HOST}")
    print(f"  Requirement: improve >= {IMPROVEMENT_THRESHOLD*100:.0f}% instances")
    print(f"  Aug factor: adaptive min(N*8, {AUG_FACTOR})")
    print(f"  Target: {TARGET_NUM_METHODS} valid methods")
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

    # Track found methods and failures
    found_methods = []
    failed_attempts = []
    iteration = 0
    os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

    # Also try some hand-crafted strategies first
    handcrafted = [
        {
            'name': 'Halton Base-2 Rotations',
            'idea': 'Use Halton quasi-random sequence (base 2) for rotation angles, providing low-discrepancy angular coverage',
            'novelty': 'Halton sequences are quasi-random with provably low discrepancy, ensuring angles are more uniformly distributed than golden angle or evenly-spaced approaches',
            'code': '''def generate_augmentations(coords, aug_factor=256):
    import torch
    import numpy as np
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)
    def halton(i, base=2):
        f, r = 1.0, 0.0
        while i > 0:
            f = f / base
            r = r + f * (i % base)
            i = i // base
        return r
    extras = []
    for i in range(aug_factor - 8):
        angle = halton(i + 1) * 2 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:,:,0]*c - centered[:,:,1]*s
        yr = centered[:,:,0]*s + centered[:,:,1]*c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx-mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)'''
        },
        {
            'name': 'Golden-Dihedral Expansion',
            'idea': 'Golden angle rotations with dihedral reflections applied to each rotation for 4x expansion per angle',
            'novelty': 'Combines golden angle irrational spacing with dihedral group expansion, producing 4 variants per unique rotation angle',
            'code': '''def generate_augmentations(coords, aug_factor=256):
    import torch
    import numpy as np
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)
    golden_angle = np.pi * (3 - np.sqrt(5))
    extras = []
    num_remaining = aug_factor - 8
    num_angles = (num_remaining + 3) // 4
    for i in range(num_angles):
        angle = (i + 1) * golden_angle
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:,:,0]*c - centered[:,:,1]*s
        yr = centered[:,:,0]*s + centered[:,:,1]*c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx-mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        nr = (rotated - mn) / sc
        rx, ry = nr[:,:,[0]], nr[:,:,[1]]
        extras.append(nr)
        if len(extras) >= num_remaining: break
        extras.append(torch.cat((1-rx, ry), dim=2))
        if len(extras) >= num_remaining: break
        extras.append(torch.cat((rx, 1-ry), dim=2))
        if len(extras) >= num_remaining: break
        extras.append(torch.cat((1-rx, 1-ry), dim=2))
        if len(extras) >= num_remaining: break
    return torch.cat(dihedral + extras[:num_remaining], dim=0)'''
        },
        {
            'name': 'Weyl Sequence Rotations',
            'idea': 'Use Weyl/irrational rotation sequence (i*sqrt(2) mod 1) for equidistributed angles',
            'novelty': 'Weyl theorem guarantees equidistribution for irrational multiples, providing theoretically optimal angle uniformity',
            'code': '''def generate_augmentations(coords, aug_factor=256):
    import torch
    import numpy as np
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)
    sqrt2 = np.sqrt(2)
    extras = []
    for i in range(aug_factor - 8):
        angle = ((i + 1) * sqrt2 % 1.0) * 2 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:,:,0]*c - centered[:,:,1]*s
        yr = centered[:,:,0]*s + centered[:,:,1]*c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx-mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)'''
        },
        {
            'name': 'Fibonacci Spiral Rotations',
            'idea': 'Rotation angles from Fibonacci spiral (i / golden_ratio * 2pi), different from golden angle',
            'novelty': 'Uses division by golden ratio rather than multiplication, producing a different quasi-uniform angular distribution',
            'code': '''def generate_augmentations(coords, aug_factor=256):
    import torch
    import numpy as np
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)
    phi = (1 + np.sqrt(5)) / 2
    extras = []
    for i in range(aug_factor - 8):
        angle = 2 * np.pi * i / phi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:,:,0]*c - centered[:,:,1]*s
        yr = centered[:,:,0]*s + centered[:,:,1]*c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx-mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)'''
        },
        {
            'name': 'Halton Base-3 Rotations',
            'idea': 'Halton sequence with base 3 for different quasi-random angular distribution than base 2',
            'novelty': 'Base-3 Halton produces fundamentally different low-discrepancy points than base-2, offering complementary angular coverage',
            'code': '''def generate_augmentations(coords, aug_factor=256):
    import torch
    import numpy as np
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)
    def halton(i, base=3):
        f, r = 1.0, 0.0
        while i > 0:
            f = f / base
            r = r + f * (i % base)
            i = i // base
        return r
    extras = []
    for i in range(aug_factor - 8):
        angle = halton(i + 1, base=3) * 2 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:,:,0]*c - centered[:,:,1]*s
        yr = centered[:,:,0]*s + centered[:,:,1]*c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx-mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)'''
        },
    ]

    # First evaluate handcrafted strategies
    print("Phase 1: Evaluating handcrafted strategies...")
    print("-" * 70)

    for hc in handcrafted:
        if len(found_methods) >= TARGET_NUM_METHODS:
            break
        iteration += 1
        print(f"[{iteration}] Testing: {hc['name']}... ", end="", flush=True)

        full_code = "import torch\nimport numpy as np\nimport math\n\n" + hc['code']
        namespace = {}
        try:
            exec(full_code, namespace)
            func = namespace['generate_augmentations']
        except Exception as e:
            print(f"compile error: {e}")
            continue

        try:
            result = evaluate_method(model, device, func, AUG_FACTOR)
        except Exception as e:
            print(f"eval error: {e}")
            continue

        ratio = result['improvement_ratio']
        avg_gap = result['avg_gap']
        improved = result['improved_count']
        total = result['total_count']

        if ratio >= IMPROVEMENT_THRESHOLD:
            print(f"✓ PASS! gap={avg_gap:.4f}%, improved={improved}/{total} ({ratio*100:.0f}%)")
            method_info = {
                'name': hc['name'],
                'idea': hc['idea'],
                'novelty': hc['novelty'],
                'code': hc['code'],
                'avg_gap': avg_gap,
                'baseline_avg_gap': result['baseline_avg_gap'],
                'improved_count': improved,
                'total_count': total,
                'improvement_ratio': ratio,
                'per_instance': result['per_instance'],
                'aug_factor': AUG_FACTOR,
                'source': 'handcrafted',
                'checkpoint_path': CHECKPOINT_PATH,
            }
            found_methods.append(method_info)

            # Save method code
            method_file = os.path.join(CHECKPOINT_SAVE_DIR, f"method_{len(found_methods):02d}.py")
            with open(method_file, 'w') as f:
                f.write(f'"""Method {len(found_methods)}: {hc["name"]}\n{hc["idea"]}\n"""\n')
                f.write("import torch\nimport numpy as np\nimport math\n\n")
                f.write(hc['code'])
            method_info['saved_path'] = method_file
        else:
            print(f"✗ FAIL. gap={avg_gap:.4f}%, improved={improved}/{total} ({ratio*100:.0f}%)")
            failed_attempts.append({'idea': hc['idea'], 'gap': avg_gap, 'ratio': ratio})

    # Phase 2: LLM-generated strategies
    if len(found_methods) < TARGET_NUM_METHODS:
        print(f"\nPhase 2: LLM search (need {TARGET_NUM_METHODS - len(found_methods)} more)...")
        print("-" * 70)

    while len(found_methods) < TARGET_NUM_METHODS:
        iteration += 1
        print(f"[{iteration}] Calling LLM... ", end="", flush=True)

        prompt = generate_prompt(found_methods, failed_attempts)
        response = call_llm(prompt, temperature=0.9 + 0.05 * (iteration % 5))

        if response is None:
            print("LLM call failed")
            time.sleep(5)
            continue

        func_code = extract_function(response)
        if func_code is None:
            print("extraction failed")
            failed_attempts.append({'idea': 'extraction_failed', 'gap': 999, 'ratio': 0})
            time.sleep(2)
            continue

        func, msg = compile_function(func_code)
        if func is None:
            print(f"compile error: {msg}")
            failed_attempts.append({'idea': f'compile_error: {msg}', 'gap': 999, 'ratio': 0})
            time.sleep(2)
            continue

        print("evaluating... ", end="", flush=True)
        try:
            result = evaluate_method(model, device, func, AUG_FACTOR)
        except Exception as e:
            print(f"eval error: {str(e)[:50]}")
            failed_attempts.append({'idea': f'eval_error: {str(e)[:50]}', 'gap': 999, 'ratio': 0})
            time.sleep(2)
            continue

        ratio = result['improvement_ratio']
        avg_gap = result['avg_gap']
        improved = result['improved_count']
        total = result['total_count']

        if ratio >= IMPROVEMENT_THRESHOLD:
            # Generate a name/description for this method
            name_prompt = (
                f"Give a short creative name (3-5 words) for this augmentation strategy:\n"
                f"```\n{func_code[:300]}\n```\n"
                f"Return ONLY the name, nothing else."
            )
            name_resp = call_llm(name_prompt, temperature=0.5, max_retries=3)
            method_name = name_resp.strip().strip('"\'') if name_resp else f"LLM Strategy {len(found_methods)+1}"

            # Get novelty description
            novelty_prompt = (
                f"In one sentence, what makes this augmentation strategy novel compared to standard rotations?\n"
                f"```\n{func_code[:400]}\n```\n"
                f"Return ONLY the sentence."
            )
            novelty_resp = call_llm(novelty_prompt, temperature=0.3, max_retries=3)
            novelty = novelty_resp.strip() if novelty_resp else "LLM-discovered novel transformation strategy"

            print(f"✓ PASS! gap={avg_gap:.4f}%, improved={improved}/{total} ({ratio*100:.0f}%) - {method_name}")

            method_info = {
                'name': method_name,
                'idea': f"LLM-generated strategy achieving {avg_gap:.4f}% avg gap",
                'novelty': novelty,
                'code': func_code,
                'avg_gap': avg_gap,
                'baseline_avg_gap': result['baseline_avg_gap'],
                'improved_count': improved,
                'total_count': total,
                'improvement_ratio': ratio,
                'per_instance': result['per_instance'],
                'aug_factor': AUG_FACTOR,
                'source': 'llm_generated',
                'checkpoint_path': CHECKPOINT_PATH,
            }
            found_methods.append(method_info)

            # Save method code
            method_file = os.path.join(CHECKPOINT_SAVE_DIR, f"method_{len(found_methods):02d}.py")
            with open(method_file, 'w') as f:
                f.write(f'"""Method {len(found_methods)}: {method_name}\n{novelty}\n"""\n')
                f.write("import torch\nimport numpy as np\nimport math\n\n")
                f.write(func_code)
            method_info['saved_path'] = method_file

            # Save intermediate results
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(found_methods, f, indent=2, default=str)
        else:
            print(f"✗ gap={avg_gap:.4f}%, improved={improved}/{total} ({ratio*100:.0f}%)")
            failed_attempts.append({'idea': func_code[:100], 'gap': avg_gap, 'ratio': ratio})

        time.sleep(3)  # Rate limit protection

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(found_methods, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"DONE! Found {len(found_methods)} valid methods.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Method files saved in: {CHECKPOINT_SAVE_DIR}/")
    print()
    print(f"{'#':<3} {'Name':<30} {'Avg Gap':>8} {'Improved':>10} {'Source':>12}")
    print("-" * 70)
    for i, m in enumerate(found_methods):
        print(f"{i+1:<3} {m['name']:<30} {m['avg_gap']:>7.4f}% "
              f"{m['improved_count']}/{m['total_count']:>6} {m['source']:>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
