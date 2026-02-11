from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast


def discover_checkpoints(outputs_dir: Path) -> list[Path]:
    return sorted(outputs_dir.glob("*/finetune/last.ckpt"))


def run_eval(
    eval_script: Path,
    checkpoint: Path,
    csv_path: Path,
    device: str,
    save_predictions_csv: bool,
) -> tuple[int, float, str]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(checkpoint),
        "--csv",
        str(csv_path),
        "--device",
        device,
    ]
    if save_predictions_csv:
        cmd.append("--save_predictions_csv")

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, elapsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch eval for CORA finetune checkpoints")
    _ = parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"), help="Path to CORA outputs directory")
    _ = parser.add_argument("--csv", type=Path, default=Path("data/quic360/test.csv"), help="Test CSV path")
    _ = parser.add_argument("--device", type=str, default="cuda", help="Device for eval.py")
    _ = parser.add_argument("--save_predictions_csv", action="store_true", help="Forward flag to eval.py")
    _ = parser.add_argument("--dry_run", action="store_true", help="Print targets without running eval")
    _ = parser.add_argument(
        "--summary_file",
        type=Path,
        default=Path("outputs") / "finetune_eval_summary.json",
        help="Where to save run summary JSON",
    )
    args = parser.parse_args()

    outputs_dir_arg = cast(Path, args.outputs_dir)
    csv_arg = cast(Path, args.csv)
    summary_file_arg = cast(Path, args.summary_file)
    device = cast(str, args.device)
    save_predictions_csv = cast(bool, args.save_predictions_csv)
    dry_run = cast(bool, args.dry_run)

    root_dir = Path(__file__).resolve().parents[1]
    eval_script = root_dir / "scripts" / "eval.py"
    outputs_dir = outputs_dir_arg if outputs_dir_arg.is_absolute() else root_dir / outputs_dir_arg
    csv_path = csv_arg if csv_arg.is_absolute() else root_dir / csv_arg
    summary_file = summary_file_arg if summary_file_arg.is_absolute() else root_dir / summary_file_arg

    if not eval_script.exists():
        print(f"[ERROR] eval script not found: {eval_script}")
        return 1
    if not outputs_dir.exists():
        print(f"[ERROR] outputs_dir not found: {outputs_dir}")
        return 1
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 1

    checkpoints = discover_checkpoints(outputs_dir)
    if not checkpoints:
        print(f"[ERROR] no finetune checkpoints found in: {outputs_dir}")
        return 1

    print(f"Found {len(checkpoints)} finetune checkpoint(s)")
    for ckpt in checkpoints:
        print(f" - {ckpt}")

    if dry_run:
        return 0

    results: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(csv_path),
        "device": device,
        "results": results,
    }

    has_failure = False
    for checkpoint in checkpoints:
        print(f"\n[RUN] {checkpoint}")
        rc, elapsed_sec, output = run_eval(
            eval_script=eval_script,
            checkpoint=checkpoint,
            csv_path=csv_path,
            device=device,
            save_predictions_csv=save_predictions_csv,
        )
        record: dict[str, object] = {
            "checkpoint": str(checkpoint),
            "returncode": rc,
            "elapsed_sec": round(elapsed_sec, 2),
            "status": "ok" if rc == 0 else "failed",
            "log_tail": output[-2000:],
        }
        results.append(record)
        if rc == 0:
            print(f"[OK] {checkpoint} ({elapsed_sec:.1f}s)")
        else:
            has_failure = True
            print(f"[FAIL] {checkpoint} ({elapsed_sec:.1f}s)")

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    _ = summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_file}")

    return 2 if has_failure else 0


if __name__ == "__main__":
    code = main()
    raise SystemExit(code)
