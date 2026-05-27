"""Project 2 main entry: Use EoH (Evolution of Heuristics) to search for an
optimal augmentation strategy for the POMO TSP solver.

Usage:
    cd project2
    python run_eoh.py

This script uses gpt-4o-mini via the configured API to evolve augmentation functions.
The best function found will be logged and can be extracted from the logs directory.
"""
from __future__ import annotations

import os
import sys
import inspect
from datetime import datetime

import pytz

# ---- Path setup ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM4AD_DIR = os.path.join(PROJECT_ROOT, "src", "LLM4AD")

sys.path.insert(0, os.path.dirname(__file__))  # for evaluation.py, template.py
sys.path.insert(0, LLM4AD_DIR)

import llm4ad
from llm4ad.task import import_all_evaluation_classes
from llm4ad.method import import_all_method_classes_from_subfolders
from llm4ad.tools.llm import import_all_llm_classes_from_subfolders

# Import all LLM4AD components
import_all_evaluation_classes(os.path.join(LLM4AD_DIR, "llm4ad", "task"))
import_all_method_classes_from_subfolders(os.path.join(LLM4AD_DIR, "llm4ad", "method"))
import_all_llm_classes_from_subfolders(os.path.join(LLM4AD_DIR, "llm4ad", "tools", "llm"))

# Register our custom evaluation class
from evaluation import POMOBiasEvaluation

# Dynamically import all usable classes from llm4ad
for module in [llm4ad.tools.llm, llm4ad.tools.profiler, llm4ad.task, llm4ad.method]:
    globals().update(
        {name: obj for name, obj in vars(module).items() if inspect.isclass(obj)}
    )

# Also register our evaluation class
globals()["POMOBiasEvaluation"] = POMOBiasEvaluation


def main(llm: dict, method: dict, evaluation: dict, profiler: dict):
    """Execute the EoH optimization process."""
    profiler_case = globals()[profiler["name"]]
    llm_case = globals()[llm["name"]]
    method_case = globals()[method["name"]]
    eval_case = globals()[evaluation["name"]]

    profiler_inst = profiler_case(
        evaluation_name=evaluation["name"],
        method_name=method["name"],
        log_dir=profiler["log_dir"],
        log_style="complex",
        create_random_path=False,
        final_log_dir=profiler["log_dir"],
    )

    llm_params = {k: v for k, v in llm.items() if k != "name"}
    method_params = {k: v for k, v in method.items() if k != "name"}
    evaluation_params = {k: v for k, v in evaluation.items() if k != "name"}

    llm_inst = llm_case(**llm_params)
    eval_inst = eval_case(**evaluation_params)
    method_inst = method_case(
        llm=llm_inst, profiler=profiler_inst, evaluation=eval_inst, **method_params
    )
    method_inst.run()


if __name__ == "__main__":
    # ============= Configuration =============
    llm = {
        "name": "HttpsApi",
        "host": "pre-openai-keys.alibaba-inc.com",
        "key": "aib_AIB_Marco_LLM_5f4ced",
        "model": "gpt-4o-mini",
    }

    method = {
        "name": "EoH",
        "max_sample_nums": 200,        # Total functions to evaluate
        "max_generations": 20,         # Number of generations
        "pop_size": 5,                 # Population size
        "num_samplers": 1,             # Sequential LLM queries (avoid rate limit)
        "num_evaluators": 1,           # Sequential eval (GPU bound)
        "debug_mode": False,           # Don't crash on API errors, just retry
        "selection": 2,
    }

    evaluation = {
        "name": "POMOBiasEvaluation",
        "timeout_seconds": 300,        # 5 min per evaluation
    }

    # Log directory
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    log_folder = os.path.join(
        PROJECT_ROOT,
        "project2",
        "logs",
        process_start_time.strftime("%Y%m%d_%H%M%S") + "_POMOBias_EoH"
    )
    os.makedirs(log_folder, exist_ok=True)

    profiler = {
        "name": "ProfilerBase",
        "log_dir": log_folder,
    }

    print("=" * 60)
    print("Project 2: LLM-Designed Augmentation Strategy for POMO TSP")
    print("=" * 60)
    print(f"  LLM Model: {llm['model']}")
    print(f"  Method: EoH (pop={method['pop_size']}, gens={method['max_generations']})")
    print(f"  Max samples: {method['max_sample_nums']}")
    print(f"  Log folder: {log_folder}")
    print("=" * 60)
    print()

    main(llm=llm, method=method, evaluation=evaluation, profiler=profiler)
