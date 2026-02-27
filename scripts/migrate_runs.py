#!/usr/bin/env python3
"""Migrate existing run directories to new structure."""
import argparse
import os
import json
import shutil
from datetime import datetime
from pathlib import Path


def scan_runs(source_dir: str) -> list:
    """Scan runs directory and classify experiments."""
    runs_path = Path(source_dir)
    if not runs_path.exists():
        print(f"No runs directory found at {source_dir}")
        return []
    
    experiments = []
    for item in sorted(runs_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check for stage_state.json (CORA) or adapter_config.json (baseline)
            is_cora = any(item.glob("**/stage_state.json")) or any(item.glob("**/*vision*.ckpt"))
            is_baseline = any(item.glob("**/adapter_config.json"))
            mod_time = datetime.fromtimestamp(item.stat().st_mtime)
            experiments.append({
                "name": item.name,
                "path": str(item),
                "type": "cora" if is_cora else ("baseline" if is_baseline else "unknown"),
                "date": mod_time.strftime("%Y%m%d"),
                "mod_time": mod_time.isoformat(),
            })
    return experiments


def plan_migration(experiments: list, dest_dir: str) -> list:
    """Plan migration moves."""
    plan = []
    date_counters = {}
    for exp in experiments:
        if exp["type"] == "cora":
            date = exp["date"]
            key = f"{exp['name']}_{date}"
            date_counters[key] = date_counters.get(key, 0) + 1
            num = date_counters[key]
            new_path = os.path.join(dest_dir, exp["name"], f"{date}_{num:03d}")
        elif exp["type"] == "baseline":
            new_path = os.path.join(dest_dir, "baseline", exp["name"])
        else:
            new_path = os.path.join(dest_dir, "unclassified", exp["name"])
        plan.append({"source": exp["path"], "dest": new_path, "type": exp["type"]})
    return plan


def main():
    parser = argparse.ArgumentParser(description="Migrate run directories to new structure")
    parser.add_argument("--source", type=str, default="runs", help="Source runs directory")
    parser.add_argument("--dest", type=str, default="runs", help="Destination runs directory")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    experiments = scan_runs(args.source)
    if not experiments:
        print("No experiments found to migrate.")
        return

    plan = plan_migration(experiments, args.dest)
    
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Migration Plan:")
    print(f"{'='*60}")
    for move in plan:
        status = "[SKIP]" if move["source"] == move["dest"] else "[MOVE]"
        print(f"  {status} {move['source']}")
        print(f"      → {move['dest']}  ({move['type']})")
    print(f"{'='*60}")
    print(f"Total: {len(plan)} experiments")

    if args.dry_run:
        print("\nDRY RUN complete. No files were moved.")
        return

    if not args.yes:
        response = input("\nProceed with migration? [y/N] ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return

    for move in plan:
        if move["source"] != move["dest"]:
            os.makedirs(os.path.dirname(move["dest"]), exist_ok=True)
            shutil.move(move["source"], move["dest"])
            print(f"  Moved: {move['source']} → {move['dest']}")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
