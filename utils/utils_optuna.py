from optuna.trial import TrialState
import os
import json
def save_callback(study, trial, name, folder):
    # Only act on successful, completed trials
    if trial.state != TrialState.COMPLETE:
        return

    # Track best value across calls
    if not hasattr(save_callback, "best_value"):
        save_callback.best_value = float("inf")

    current_best = float(study.best_value)
    if current_best >= save_callback.best_value:
        return

    # Update tracker
    save_callback.best_value = current_best

    # Write to a user-writable location (avoid PermissionError on /jax_res)
    out_dir = folder
    os.makedirs(out_dir, exist_ok=True)

    best = study.best_trial
    payload = dict(best.params)
    payload["objective value"] = current_best
    payload["trial number"] = int(best.number)
    # Optional convenience: pack kernel params if present
    ls = float(payload.get("length_scale", 0.0))
    wm = float(payload.get("weight_matern", 0.0))
    payload["parameters"] = [ls, wm]

    out_path = os.path.join(out_dir, f"{name}_current_best_params.json")
    print("Objective improved, writing parameters to :", out_path)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)