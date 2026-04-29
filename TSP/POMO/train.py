##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
DEFAULT_USE_CUDA = not DEBUG_MODE
DEFAULT_CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSProblemDef import get_default_msc_config
from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# baseline parameters (zero-arg call preserves legacy baseline behaviour)

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [3001,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': DEFAULT_USE_CUDA,
    'cuda_device_num': DEFAULT_CUDA_DEVICE_NUM,
    'epochs': 3100,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,
    },
    # Mixed Structured Curriculum (training-data generator).  Default ON; set
    # ``msc_cfg['enabled'] = False`` (or pass ``--msc_enabled false``) to fall
    # back to the legacy uniform sampler bit-for-bit.
    'msc_cfg': get_default_msc_config(),
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n100__3000epoch',
        'filename': 'log.txt'
    }
}


##########################################################################################
# CLI

def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a POMO TSP model. With zero args, mirrors the baseline recipe.",
    )
    # Core training overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train_episodes", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use_cuda", type=str2bool, default=DEFAULT_USE_CUDA)
    p.add_argument("--cuda_device_num", type=int, default=DEFAULT_CUDA_DEVICE_NUM)
    p.add_argument("--problem_size", type=int, default=None)
    p.add_argument("--pomo_size", type=int, default=None)
    p.add_argument("--desc", default=None, help="Log file description tag.")

    # Finetune path
    p.add_argument(
        "--finetune_from",
        default=None,
        help="Path to a checkpoint to load weights from (optimizer fresh).",
    )
    p.add_argument("--grad_clip_max_norm", type=float, default=None)

    # M2 distance / kNN bias (training-side)
    p.add_argument("--distance_bias_enabled", type=str2bool, default=False)
    p.add_argument("--distance_bias_scale", type=float, default=1.0)
    p.add_argument("--distance_bias_mode", choices=["logit", "attn"], default="logit")
    p.add_argument("--distance_norm_mode", choices=["none", "mean", "max", "std"], default="mean")
    p.add_argument("--knn_bias_enabled", type=str2bool, default=False)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--knn_bias_value", type=float, default=0.5)

    # M3 leader reward
    p.add_argument("--leader_reward_enabled", type=str2bool, default=False)
    p.add_argument("--leader_mode", choices=["bonus_adv", "aux_imitation"], default="bonus_adv")
    p.add_argument("--leader_gamma", type=float, default=0.5)
    p.add_argument("--leader_aux_weight", type=float, default=0.1)

    # Mixed Structured Curriculum (MSC) data generation.
    # Defaults preserve the recommended setup from the spec; flags below let
    # ablations toggle the curriculum, switch to fixed mixing ratios, or
    # disable MSC entirely (legacy uniform).
    p.add_argument("--msc_enabled", type=str2bool, default=True,
                   help="Enable Mixed Structured Curriculum data generation.")
    p.add_argument("--msc_use_curriculum", type=str2bool, default=True,
                   help="If false, use msc_fixed_ratios instead of stage_ratios.")
    p.add_argument("--msc_stage_boundaries", type=str, default=None,
                   help='Comma-separated stage cutoffs in [0,1], e.g. "0.3,0.7".')
    p.add_argument("--msc_fixed_ratios", type=str, default=None,
                   help='Comma-separated ratios "uniform,clustered_uniform,gaussian_mixture".')
    p.add_argument("--msc_log_every", type=int, default=None,
                   help="Emit an MSC debug line every N batches (0 disables).")
    return p


def _apply_cli_overrides(args):
    if args.epochs is not None:
        trainer_params['epochs'] = args.epochs
    if args.train_episodes is not None:
        trainer_params['train_episodes'] = args.train_episodes
    if args.train_batch_size is not None:
        trainer_params['train_batch_size'] = args.train_batch_size
    if args.lr is not None:
        optimizer_params['optimizer']['lr'] = args.lr
    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
    if args.pomo_size is not None:
        env_params['pomo_size'] = args.pomo_size
    trainer_params['use_cuda'] = args.use_cuda
    trainer_params['cuda_device_num'] = args.cuda_device_num
    if args.desc is not None:
        logger_params['log_file']['desc'] = args.desc

    if args.finetune_from is not None:
        trainer_params['finetune_from'] = os.path.abspath(args.finetune_from)

    if args.grad_clip_max_norm is not None:
        trainer_params['grad_clip_max_norm'] = args.grad_clip_max_norm

    if args.distance_bias_enabled or args.knn_bias_enabled:
        trainer_params['distance_bias_cfg'] = {
            'distance_bias_enabled': args.distance_bias_enabled,
            'distance_bias_scale': args.distance_bias_scale,
            'distance_bias_mode': args.distance_bias_mode,
            'distance_norm_mode': args.distance_norm_mode,
            'knn_bias_enabled': args.knn_bias_enabled,
            'knn_k': args.knn_k,
            'knn_bias_value': args.knn_bias_value,
        }

    if args.leader_reward_enabled:
        trainer_params['leader_cfg'] = {
            'leader_reward_enabled': True,
            'leader_mode': args.leader_mode,
            'leader_gamma': args.leader_gamma,
            'leader_aux_weight': args.leader_aux_weight,
        }

    _apply_msc_overrides(args)


def _parse_csv_floats(text):
    return [float(t) for t in text.split(',') if t.strip()]


def _apply_msc_overrides(args):
    """Apply CLI flags onto trainer_params['msc_cfg'] in-place."""
    msc_cfg = trainer_params.setdefault('msc_cfg', get_default_msc_config())
    msc_cfg['enabled'] = bool(args.msc_enabled)
    msc_cfg['use_curriculum'] = bool(args.msc_use_curriculum)

    if args.msc_stage_boundaries is not None:
        bounds = _parse_csv_floats(args.msc_stage_boundaries)
        if len(bounds) < 1:
            raise ValueError("--msc_stage_boundaries needs >=1 float (e.g. '0.3,0.7').")
        msc_cfg.setdefault('curriculum', {})['stage_boundaries'] = bounds

    if args.msc_fixed_ratios is not None:
        vals = _parse_csv_floats(args.msc_fixed_ratios)
        if len(vals) != 3:
            raise ValueError(
                "--msc_fixed_ratios needs exactly 3 floats: uniform,clustered_uniform,gaussian_mixture"
            )
        msc_cfg.setdefault('curriculum', {})['fixed_ratios'] = {
            'uniform': vals[0],
            'clustered_uniform': vals[1],
            'gaussian_mixture': vals[2],
        }

    if args.msc_log_every is not None:
        msc_cfg['log_every'] = int(args.msc_log_every)


##########################################################################################
# main

def main():
    args = build_parser().parse_args()
    _apply_cli_overrides(args)

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(
        trainer_params['use_cuda'], trainer_params['cuda_device_num']))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
