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
    p.add_argument('--training-augmentations', action='store_true',
                   help='Apply agent augmentations exactly as configured in training YAML')
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
        "agent_name": "real_eye_in_hand_ur5e",
        "demo_name": "human_rgb",
        "train_tasks": discovered,
        "test_tasks": [],
        "is_real": True,
    })
    ds = MTLFDAgentTeacherDataset(mode='train', **ds_cfg)
    model = load_checkpoint(args.checkpoint, cfg, device)
    if args.training_augmentations:
        print('AGENT training augmentations:', {k: cfg['dataset'].get(k) for k in
              ('joystick_aug', 'joystick_source_roi', 'joystick_target_roi',
               'person_aug', 'person_aug_prob', 'color_jitter', 'rand_crop',
               'rand_translate', 'rand_flip')}, flush=True)
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
            # The demo must remain unaugmented for this controlled test.
            # Sample the same context for every scene, independently of agent RNG.
            np.random.seed(args.seed)
            random.seed(args.seed)
            context = ds._make_context(demo_traj)

            # Training augmentations operate on AGENT ONLY. _make_agent_sample
            # applies joystick removal BEFORE crop, person recolor AFTER resize,
            # then OSVI randomize_video. It also adjusts projection_matrix.
            # Use exactly the YAML settings, including ROI and probability.
            if args.training_augmentations:
                original = {
                    key: getattr(ds, key) for key in (
                        'joystick_aug', 'joystick_source_roi', 'joystick_target_roi',
                        'person_aug', 'person_aug_prob', 'color_jitter',
                        'rand_crop', 'rand_translate', 'rand_flip',
                    )
                }
                training_ds = cfg['dataset']
                ds.joystick_aug = bool(training_ds.get('joystick_aug', False))
                ds.joystick_source_roi = tuple(training_ds.get(
                    'joystick_source_roi', ds.joystick_source_roi))
                ds.joystick_target_roi = tuple(training_ds.get(
                    'joystick_target_roi', ds.joystick_target_roi))
                ds.person_aug = bool(training_ds.get('person_aug', False))
                ds.person_aug_prob = float(training_ds.get('person_aug_prob', ds.person_aug_prob))
                ds.color_jitter = training_ds.get('color_jitter')
                ds.rand_crop = training_ds.get('rand_crop')
                ds.rand_translate = np.array(training_ds.get('rand_translate') or [0, 0])
                ds.rand_flip = bool(training_ds.get('rand_flip', False))
                np.random.seed(args.seed + i)
                random.seed(args.seed + i)
                try:
                    agent = ds._make_agent_sample(robot_traj)
                finally:
                    for key, value in original.items():
                        setattr(ds, key, value)
            else:
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
