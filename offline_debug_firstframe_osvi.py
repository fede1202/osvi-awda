#!/usr/bin/env python3
"""Offline OSVI-AWDA prediction + *exact training debug overlay renderer*.

Uses the dataset's ground-truth waypoint and projection builders; uses ONLY
robot PKL frame 0 for model conditioning (duplicated to two model frames).
It does not run the robot or train the model.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from hem.models.inverse_module import InverseImitation
from mtlfd_adaptation import debug_utils
import mtlfd_adaptation.mtlfd_dataset as dataset_module
from mtlfd_adaptation.mtlfd_dataset import MTLFDAgentTeacherDataset
from mtlfd_adaptation.trajectory_bridge import load_traj


def load_checkpoint(path, config, device):
    model = InverseImitation(**config['policy'])
    try:
        ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # older PyTorch without weights_only argument
        ckpt = torch.load(str(path), map_location='cpu')
    state = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Training YAML belonging to checkpoint')
    p.add_argument('--checkpoint', required=True, help='model_save-N.pt')
    p.add_argument('--robot-root', required=True, help='real_eye_in_hand_ur5e_pick_place dir')
    p.add_argument('--demo-root', required=True, help='human_rgb_pick_place dir')
    p.add_argument('--output', required=True)
    p.add_argument('--max-files', type=int, default=10, help='0 for all files')
    p.add_argument('--task-ids', type=int, nargs='*', default=None)
    p.add_argument('--demo-index', type=int, default=0, help='index of sorted same-task demo')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Execute on an allocated GPU compute node.')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_cfg = dict(cfg['dataset'])
    root = Path(args.robot_root).resolve()
    demo_root = Path(args.demo_root).resolve()
    robot_files = sorted(root.glob('task_*/*.pkl'))
    if args.task_ids is not None:
        selected_tasks = set(args.task_ids)
        robot_files = [f for f in robot_files if int(f.parent.name.split('_')[-1]) in selected_tasks]
    if not robot_files:
        raise RuntimeError('No robot PKLs found in {}'.format(root))
    if args.max_files > 0:
        robot_files = robot_files[:args.max_files]

    # Instantiate the *same dataset helper* used during training, but with the
    # specified ROBOT agent and without stochastic or human-only augmentations.
    # Dataset helper computes GT (50 world XYZ/grasp points) and a calibrated
    # per-sample projection; manually interpreting raw action XYZ would be WRONG.
    discovered = sorted({int(f.parent.name.split('_')[-1]) for f in robot_files})
    ds_cfg.update({
        "root_dir": "/mnt/beegfs/f.demaio28/osvi_awda_finetune/data/opt_dataset",
        "task_name": "pick_place",
        "agent_name": "human_osvi_front_grasp_final",
        "demo_name": "human_rgb",
        "train_tasks": discovered,
        "test_tasks": [],
        "color_jitter": None,
        "rand_crop": None,
        "rand_translate": [0, 0],
        "rand_flip": False,
        "person_aug": False,
        "joystick_aug": False,
        "is_real": True,
    })
    ds = MTLFDAgentTeacherDataset(mode='train', **ds_cfg)
    model = load_checkpoint(args.checkpoint, cfg, device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Source training dataset uses AGENT_SETTLE_STEPS=1. Explicitly override
    # ONLY for this offline request to condition on the literal *first* frame.
    old_settle = dataset_module.AGENT_SETTLE_STEPS
    dataset_module.AGENT_SETTLE_STEPS = 0
    try:
        for i, robot_path in enumerate(robot_files):
            task = robot_path.parent.name
            demos = sorted((demo_root / task).glob('*.pkl'))
            if not demos:
                raise FileNotFoundError('No human demo for {} in {}'.format(task, demo_root))
            if args.demo_index >= len(demos):
                raise IndexError('demo index {} >= {}'.format(args.demo_index, len(demos)))
            demo_path = demos[args.demo_index]
            robot_traj, _ = load_traj(str(robot_path), is_real=True)
            demo_traj, _ = load_traj(str(demo_path))
            context = ds._make_context(demo_traj)
            agent = ds._make_agent_sample(robot_traj)

            # Make the model consume EXACTLY the first robot frame, twice.
            # Its 2nd image would otherwise be frame 1 (T_pair=1).
            agent['images'][1:] = agent['images'][0:1]
            ctx = torch.from_numpy(context['video'][None]).to(device)
            images = torch.from_numpy(agent['images'][None]).to(device)
            states = torch.from_numpy(agent['states'][None]).to(device)
            ents = torch.tensor([int(agent['head_label'])], dtype=torch.long, device=device)
            with torch.no_grad():
                prediction = model(states, images, ctx, ret_dist=False, ents=ents)
            pred = prediction['waypoints'][0, -5:].detach().cpu().numpy()
            gt = agent['traj_points']
            projection = agent['projection_matrix']
            sample_dir = out_dir / '{}_{}'.format(task, robot_path.stem)
            sample_dir.mkdir(parents=True, exist_ok=True)
            overlay = sample_dir / 'waypoints.png'

            # EXACTLY the function + input convention used by
            # mtlfd_adaptation/train_mtlfd.py::make_debug_forward.
            debug_utils.save_waypoint_overlay(
                str(overlay), agent['images'][0], pred, gt, projection,
                image_waypoints=cfg.get('image_waypoints', False))
            debug_utils.save_frame_grid(context['video'], str(sample_dir / 'demo_context.png'))
            debug_utils.save_frame_grid(agent['images'], str(sample_dir / 'agent_frames.png'))
            np.savetxt(str(sample_dir / 'pred_waypoints.csv'), pred, delimiter=',')
            np.savetxt(str(sample_dir / 'gt_traj_points.csv'), gt, delimiter=',')
            np.savetxt(str(sample_dir / 'projection_matrix.csv'), projection, delimiter=',')
            rows.append(dict(task=task, robot_pkl=str(robot_path), demo_pkl=str(demo_path), overlay=str(overlay)))
            print('[{}/{}] {}'.format(i + 1, len(robot_files), overlay), flush=True)
    finally:
        dataset_module.AGENT_SETTLE_STEPS = old_settle
    with open(str(out_dir / 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print('Done: {} overlays'.format(len(rows)))


if __name__ == '__main__':
    main()
