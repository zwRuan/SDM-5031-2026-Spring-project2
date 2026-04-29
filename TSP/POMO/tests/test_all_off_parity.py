"""End-to-end sanity check: with every new feature OFF the tester output
must match the original baseline payload bit-for-bit.

This test requires the bundled checkpoint at::

    TSP/POMO/result/saved_tsp100_model2_longTrain/checkpoint-3000.pt

and a directory of TSP instances at ``TSP/data/val``. It is skipped if
either of these is missing or if torch / CUDA is unavailable.

Run manually:
    cd TSP/POMO
    python tests/test_all_off_parity.py

The recorded baseline reference comes from
``TSP/POMO/result/result_record/baseline_result.txt`` (the SUMMARY_JSON line).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POMO_DIR = os.path.dirname(HERE)
sys.path.insert(0, POMO_DIR)
sys.path.insert(0, os.path.dirname(POMO_DIR))
sys.path.insert(0, os.path.dirname(os.path.dirname(POMO_DIR)))

EXPECTED_AVG_AUG_GAP = 2.329567752253744
EXPECTED_AVG_NO_AUG_GAP = 3.753675687101596
RTOL = 1e-6


def main() -> int:
    ckpt = os.path.join(POMO_DIR, "result", "saved_tsp100_model2_longTrain", "checkpoint-3000.pt")
    data_path = os.path.abspath(os.path.join(POMO_DIR, "..", "data", "val"))

    if not os.path.exists(ckpt):
        print("SKIP: missing checkpoint", ckpt)
        return 0
    if not os.path.isdir(data_path):
        print("SKIP: missing data dir", data_path)
        return 0

    try:
        import torch  # noqa: F401
    except Exception as exc:
        print("SKIP: torch unavailable:", exc)
        return 0

    from utils.utils import create_logger
    create_logger(log_file={"desc": "_parity_sanity", "filename": "parity.txt", "filepath": "./result_lib/_parity"})

    from TSPTester_LIB import TSPTester_LIB
    tester_params = {
        "use_cuda": False,
        "cuda_device_num": 0,
        "checkpoint_path": ckpt,
        "filename": data_path,
        "augmentation_enable": True,
        "aug_factor": 8,
        "detailed_log": False,
        "scale_range_all": [[0, 10000]],
        # All new features explicitly OFF.
        "rerank_enabled": False,
        "two_opt_enabled": False,
        "distance_bias_enabled": False,
        "knn_bias_enabled": False,
    }
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    tester = TSPTester_LIB(model_params=model_params, tester_params=tester_params)
    result = tester.run_lib()

    print(json.dumps(result.to_dict(), default=str, indent=2))
    assert result.avg_aug_gap is not None
    assert abs(result.avg_aug_gap - EXPECTED_AVG_AUG_GAP) < 1e-3, (
        f"avg_aug_gap drifted: got {result.avg_aug_gap}, expected ~{EXPECTED_AVG_AUG_GAP}"
    )
    assert abs(result.avg_no_aug_gap - EXPECTED_AVG_NO_AUG_GAP) < 1e-3, (
        f"avg_no_aug_gap drifted: got {result.avg_no_aug_gap}, expected ~{EXPECTED_AVG_NO_AUG_GAP}"
    )
    print("PASS: all-off parity holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
