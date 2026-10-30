#!/usr/bin/env python3
"""Inspect a real OSVI-AWDA human-only DataLoader batch without training."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mtlfd_adaptation.mtlfd_dataset import MTLFDAgentTeacherDataset
from hem.models.trainer import PerTaskBatchSampler


def to_rgb_uint8(chw):
    """randomize_video(normalize=True) -> ImageNet, otherwise /255."""
    a = np.asarray(chw, dtype=np.float32).transpose(1, 2, 0)
    # This dataset calls randomize_video(..., normalize=True), so undo ImageNet.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    a = (a * std + mean) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def label_image(img, label):
    out = cv2.resize(img, (360, 200), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(out, (0, 0), (360, 26), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def describe(name, x):
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    msg = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.number):
        msg.update(min=float(arr.min()), max=float(arr.max()),
                   finite=bool(np.isfinite(arr).all()))
    print(name, msg, flush=True)
    return msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds = MTLFDAgentTeacherDataset(**cfg['dataset'], mode=args.mode)
    if not len(ds):
        raise RuntimeError('Dataset vuoto')
    print('Pairs:', len(ds), 'agent PKL:', len(ds.agent_files),
          'demo PKL:', len(ds.demo_files), flush=True)
    print('Custom aug:', 'person', ds.person_aug, 'p', ds.person_aug_prob,
          'joystick', ds.joystick_aug, 'mode', ds.mode, flush=True)
    # Same batch sampler used by OSVI-AWDA during training
    samples_per_task = cfg["samples_per_task"]

    batch_sampler = PerTaskBatchSampler(
        ds,
        samples_per_task=samples_per_task,
    )

    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
    )

    print("\n=== REAL OSVI-AWDA TRAINING BATCH ===")
    print("Samples per task:", samples_per_task)
    print("Number of tasks:", len(batch_sampler.tasks))
    print(
        "Effective batch size:",
        samples_per_task * len(batch_sampler.tasks)
    )

    print("Training tasks:", batch_sampler.tasks)
    context, traj = next(iter(loader))

    from collections import Counter

    task_names = traj["setting_name"]

    print("\n=== BATCH TASK DISTRIBUTION ===")

    counts = Counter(task_names)

    for task, count in sorted(counts.items()):
        print(f"{task}: {count} samples")

    print("Total samples:", len(task_names))

    assert len(task_names) == 24
    assert len(counts) == 12
    assert all(count == 2 for count in counts.values())

    print("\nBatch composition verified!")

    print('\n=== REAL BATCH ===')
    report = {}
    for name, value in context.items():
        if torch.is_tensor(value):
            report['context.' + name] = describe('context.' + name, value)
        else:
            print('context.' + name, value)
    for name, value in traj.items():
        if torch.is_tensor(value):
            report['traj.' + name] = describe('traj.' + name, value)
        else:
            print('traj.' + name, value)
    report['dataset_pairs'] = len(ds)
    report['dataset_agent_files'] = len(ds.agent_files)
    report['dataset_demo_files'] = len(ds.demo_files)
    report['mode'] = args.mode
    report['samples_per_task'] = samples_per_task
    report['effective_batch_size'] = len(task_names)
    report['task_distribution'] = dict(counts)
    report['agent_task_names'] = list(task_names)
    report['context_files'] = list(context['fname'])
    with open(out / 'batch_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    B, Tctx, C, H, W = context['video'].shape
    Ba, Tagent, Ca, Ha, Wa = traj['images'].shape
    assert B == Ba and C == Ca == 3
    assert Tctx == cfg['dataset']['T_context']
    assert Tagent == cfg['dataset']['T_pair'] + 1
    assert traj['traj_points'].shape == (B, 50, 4)
    assert traj['projection_matrix'].shape == (B, 4, 4)

    for b in range(B):
        folder = out / f'sample_{b:02d}'
        folder.mkdir(exist_ok=True)
        ctx_imgs = [to_rgb_uint8(context['video'][b, t].numpy()) for t in range(Tctx)]
        ag_imgs = [to_rgb_uint8(traj['images'][b, t].numpy()) for t in range(Tagent)]
        for t, im in enumerate(ctx_imgs):
            save_rgb(folder / f'context_{t:02d}.png', im)
        for t, im in enumerate(ag_imgs):
            save_rgb(folder / f'agent_{t:02d}.png', im)
        ctx_grid = np.concatenate([np.concatenate([label_image(ctx_imgs[t], f'CONTEXT {t}')
                                                   for t in range(start, min(start + 5, Tctx))], axis=1)
                                   for start in range(0, Tctx, 5)], axis=0)
        ag_grid = np.concatenate([label_image(im, f'AGENT {t}') for t, im in enumerate(ag_imgs)], axis=1)
        save_rgb(folder / 'context_grid.png', ctx_grid)
        save_rgb(folder / 'agent_grid.png', ag_grid)
        np.savetxt(folder / 'traj_points.csv', traj['traj_points'][b].numpy(), delimiter=',',
                   header='x,y,z,grasp_attribute', comments='')
        np.savetxt(folder / 'projection_matrix.csv', traj['projection_matrix'][b].numpy(), delimiter=',')
        np.savetxt(folder / 'grasp_point.csv', traj['grasp_point'][b].numpy()[None], delimiter=',',
                   header='x,y,z', comments='')
        print(f'sample {b}: saved {Tctx} context, {Tagent} agent, waypoints to {folder}')

    print('\nBATCH VALIDATO. Output:', out)


if __name__ == '__main__':
    main()
