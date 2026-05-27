#!/usr/bin/env python3
"""Run the full leaf-contour processing pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    '0_rename.py',
    '1_get_leaf_mask.py',
    '2_click_keypoints.py',
    '3_rotate_img_and_mask.py',
    '4_find_contour.py',
    '5_solve_pde.py',
    '6_pca_inner.py',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run the leaf contour processing pipeline.')
    parser.add_argument(
        '--data-dir',
        default='../sample_leaf_data',
        help='Image folder path.',
    )
    parser.add_argument(
        '--sam-checkpoint',
        default=str(SCRIPT_DIR / 'sam2/checkpoints/sam2.1_hiera_large.pt'),
        help='SAM2 checkpoint path (step 1).',
    )
    parser.add_argument(
        '--sam-config',
        default='configs/sam2.1/sam2.1_hiera_l.yaml',
        help='SAM2 model config (step 1).',
    )
    parser.add_argument(
        '--bound-type',
        default='square',
        choices=['square', 'circle'],
        help='Boundary type for PDE solve (step 5).',
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        help='PDE solution resolution (step 5).',
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retrain PDE solutions even if cached (step 5).',
    )
    return parser.parse_args()


def run_step(script_name, cmd):
    print(f'\n=== Running {script_name} ===')
    print(' '.join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    python = sys.executable

    step_cmds = [
        [python, '0_rename.py', '--data-dir', data_dir],
        [
            python, '1_get_leaf_mask.py',
            '--data-dir', data_dir,
            '--sam-checkpoint', args.sam_checkpoint,
            '--sam-config', args.sam_config,
        ],
        [python, '2_click_keypoints.py', '--data-dir', data_dir],
        [python, '3_rotate_img_and_mask.py', '--data-dir', data_dir],
        [python, '4_find_contour.py', '--data-dir', data_dir],
        [
            python, '5_solve_pde.py',
            '--data-dir', data_dir,
            '--bound-type', args.bound_type,
            '--resolution', str(args.resolution),
        ] + (['--retrain'] if args.retrain else []),
        [
            python, '6_pca_inner.py',
            '--data-dir', data_dir,
            '--bound-type', args.bound_type,
        ],
    ]

    for script_name, cmd in zip(STEPS, step_cmds):
        run_step(script_name, cmd)

    print('\nPipeline finished.')


if __name__ == '__main__':
    main()
