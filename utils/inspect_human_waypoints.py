
import argparse
import numpy as np

from mtlfd_adaptation.trajectory_bridge import load_traj


def describe_value(value):
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"

    try:
        arr = np.asarray(value)
        if arr.dtype == object:
            return f"type={type(value).__name__}"

        if arr.ndim >= 2:
            return f"shape={arr.shape}, dtype={arr.dtype}"

        return f"shape={arr.shape}, value={arr.tolist()}"
    except Exception:
        return repr(value)


def inspect_dict(name, data):
    print(f"\n--- {name} ---")
    if not isinstance(data, dict):
        print(describe_value(data))
        return

    for key, value in data.items():
        print(f"{key}: {describe_value(value)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--is-real", action="store_true")
    args = parser.parse_args()

    traj, command = load_traj(
        args.pkl, is_real=args.is_real
    )

    n = len(traj)
    print("=" * 65)
    print("HUMAN TRAJECTORY INSPECTION")
    print("=" * 65)
    print("File:", args.pkl)
    print("Steps:", n)
    print("Command:", command)

    # Inspect first, second and last steps
    for t in sorted(set([0, min(1, n-1), n-1])):
        step = traj.get(t)
        print(f"\n========== STEP {t} ==========")
        print("Top-level keys:", list(step.keys()))

        inspect_dict("OBS", step.get("obs", {}))
        inspect_dict("ACTION", step.get("action"))
        if "info" in step:
            inspect_dict("INFO", step["info"])

    # All actions
    print("\n" + "=" * 65)
    print("ACTION TIMELINE")
    print("=" * 65)

    actions = []
    for t in range(n):
        action = traj.get(t).get("action")
        if action is None:
            actions.append(None)
        else:
            try:
                actions.append(
                    np.asarray(action, dtype=float).reshape(-1)
                )
            except (ValueError, TypeError):
                actions.append(None)

    for t, action in enumerate(actions):
        if t < 15 or t >= n - 5:
            print(
                f"Step {t:03d}: "
                f"{None if action is None else action.tolist()}"
            )

    valid = [
        (t, a) for t, a in enumerate(actions)
        if a is not None and len(a) > 0
    ]

    print("\nValid actions:", len(valid), "/", n)

    if valid:
        sizes = sorted(set(len(a) for _, a in valid))
        print("Action dimensions:", sizes)

        if len(sizes) == 1:
            action_array = np.stack([a for _, a in valid])
            print("Action min:", action_array.min(axis=0))
            print("Action max:", action_array.max(axis=0))

            # Current OSVI-AWDA grasp rule
            last_values = action_array[:, -1]

            print("\nLast action component:")
            print("Unique values:", np.unique(last_values))
            print("Values > 0.01:", np.sum(last_values > 0.01))

            transitions = np.where(
                np.diff((last_values > 0.01).astype(int)) != 0
            )[0]

            print("Grasp transitions (valid-action indices):",
                  transitions.tolist())
            print("Transition pairs (trajectory indices):", [
                (valid[i][0], valid[i+1][0])
                for i in transitions
            ])

    # Candidate 3D position fields
    print("\n" + "=" * 65)
    print("POSITION FIELDS")
    print("=" * 65)

    first_obs = traj.get(0).get("obs", {})
    candidates = [
        k for k in first_obs.keys()
        if any(term in k.lower() for term in [
            "pos", "hand", "eef", "ee_",
            "wrist", "pose", "joint", "state"
        ])
    ]

    print("Candidate keys:", candidates)

    for key in candidates:
        vectors = []
        valid_field = True

        for t in range(n):
            value = traj.get(t)["obs"].get(key)

            if value is None:
                valid_field = False
                break

            try:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1:
                    valid_field = False
                    break
                vectors.append(arr)
            except (ValueError, TypeError):
                valid_field = False
                break

        if not valid_field:
            print(f"\n{key}: not a numeric 1D field")
            continue

        if len(set(v.shape for v in vectors)) != 1:
            print(f"\n{key}: inconsistent dimensions")
            continue

        values = np.stack(vectors)

        print(f"\n{key}:")
        print("Shape:", values.shape)
        print("First:", values[0])
        print("Last:", values[-1])
        print("Min:", values.min(axis=0))
        print("Max:", values.max(axis=0))
        print("All finite:", bool(np.isfinite(values).all()))

        if values.shape[1] >= 3:
            xyz = values[:, :3]
            movements = np.linalg.norm(
                np.diff(xyz, axis=0), axis=1
            )
            print("Mean step distance:", movements.mean())
            print("Max step distance:", movements.max())

    # First-frame offset
    print("\n" + "=" * 65)
    print("FIRST FRAME CHECK")
    print("=" * 65)

    if n >= 2:
        for key in ("eef_pos", "ee_aa"):
            obs0 = traj.get(0)["obs"]
            obs1 = traj.get(1)["obs"]

            if key in obs0 and key in obs1:
                a = np.asarray(obs0[key]).reshape(-1)[:3]
                b = np.asarray(obs1[key]).reshape(-1)[:3]

                print(f"{key} frame 0:", a)
                print(f"{key} frame 1:", b)
                print(f"{key} difference:", b - a)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
