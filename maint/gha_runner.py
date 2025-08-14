#!/usr/bin/env python3
"""
gha_runner.py

A flexible wrapper around GitHub CLI (`gh`) to manage conda-only CI builds for llvmlite and numba:
 1. Trigger llvmdev build workflow on numba/llvmlite
 2. Trigger llvmlite conda-builder workflow on numba/llvmlite
 3. Trigger numba conda-builder workflows on numba/numba for all platforms
 4. Monitor each run to successful completion
 5. Download conda artifacts

Usage example:
  export GH_TOKEN=<your_token>
  ./gha_runner.py \
    --llvmlite-branch pr-1240-llvmlite \
    --numba-branch pr-1240-numba \
    --steps llvmdev,llvmlite_conda,numba_conda,download_llvmlite_conda,download_numba_conda \
    --reuse-run llvmdev:12345
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Set, List, Optional
from datetime import datetime, timezone

# Repository constants
LLVMLITE_REPO = "swap357/llvmlite"
NUMBA_REPO    = "swap357/numba"
DEFAULT_BRANCH = "main"

# Supported steps
ALL_STEPS = {
    "llvmdev",
    "llvmlite_conda",
    "llvmlite_wheels",
    "numba_conda",
    "numba_wheels",
    "download_llvmlite_conda",
    "download_llvmlite_wheels",
    "download_numba_conda",
    "download_numba_wheels",
}
PLATFORMS = ["osx-64", "osx-arm64", "win-64", "linux-aarch64", "linux-64"]

STATE_FILE = Path(__file__).parent / ".ci_state.json"
DRY_RUN = False


def _parse_github_time_iso8601(timestr: str) -> datetime:
    """Parse GitHub ISO8601 time strings into aware datetimes.

    GitHub returns times with a trailing 'Z'. Convert to +00:00 for fromisoformat.
    """
    return datetime.fromisoformat(timestr.replace("Z", "+00:00"))


def _gh_check_output(args: List[str]) -> str:
    """Run a gh command returning stdout, with basic retry/backoff."""
    if DRY_RUN:
        print(" ".join(args))
        return ""
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            return subprocess.check_output(args, text=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            last_exc = exc
            backoff_seconds = 2 ** attempt
            logging.warning("Command failed (%s). Retrying in %ss: %s", type(exc).__name__, backoff_seconds, " ".join(args))
            time.sleep(backoff_seconds)
    # If we get here all retries failed
    assert last_exc is not None
    raise last_exc


def _gh_run(args: List[str]) -> None:
    """Run a gh command for side-effects, with basic retry/backoff."""
    if DRY_RUN:
        print(" ".join(args))
        return
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            subprocess.run(args, check=True)
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            last_exc = exc
            backoff_seconds = 2 ** attempt
            logging.warning("Command failed (%s). Retrying in %ss: %s", type(exc).__name__, backoff_seconds, " ".join(args))
            time.sleep(backoff_seconds)
    assert last_exc is not None
    raise last_exc


def _is_gh_authenticated() -> bool:
    """Check if gh is already authenticated locally."""
    try:
        subprocess.run(["gh", "auth", "status"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def load_state() -> Dict[str, Dict[str, Any]]:
    """Load build state from disk, or return empty state."""
    if STATE_FILE.exists():
        logging.info("Reading state from %s", STATE_FILE)
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    logging.info("No state file found at %s; starting without prior state", STATE_FILE)
    return {}


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    """Persist build state to disk."""
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    logging.info("Saved state to %s", STATE_FILE)


def get_workflow_conclusion(workflow_run_id: str, repo: str) -> str:
    """Retrieve the conclusion of a completed GitHub Actions run."""
    output = _gh_check_output([
        "gh", "run", "view", workflow_run_id,
        "--repo", repo,
        "--json", "conclusion"
    ])
    data = json.loads(output or "{}")
    return data.get("conclusion", "")


def dispatch_or_reuse(
    workflow_filename: str,
    workflow_inputs: Dict[str, str],
    step_key: str,
    state: Dict[str, Dict[str, Any]],
    repo: str,
    branch: str
) -> int:
    """
    Reuse an existing successful or in-progress run, or dispatch a new one.
    Returns the workflow run ID.
    """
    state_entry     = state.get(step_key, {})
    existing_run_id = state_entry.get("run_id")
    completed       = state_entry.get("completed", False)
    last_conclusion = state_entry.get("conclusion", "")

    if existing_run_id and completed and last_conclusion == "success":
        logging.info("Reusing successful %s run %s on %s", step_key, existing_run_id, repo)
        return existing_run_id

    if existing_run_id and not completed:
        logging.info("Resuming in-progress %s run %s on %s", step_key, existing_run_id, repo)
        return existing_run_id

    # Dispatch a new run
    cmd = ["gh", "workflow", "run", workflow_filename, "--repo", repo, "--ref", branch]
    for name, value in workflow_inputs.items():
        cmd.extend(["-f", f"{name}={value}"])
    if DRY_RUN:
        # Only print the actual gh command in dry-run
        pass
    else:
        logging.info("Dispatching %s for %s on %s with inputs %s", workflow_filename, step_key, repo, workflow_inputs)
    # Take timestamp before dispatch for deterministic run capture
    pre_dispatch_time = datetime.now(timezone.utc)
    dispatch_output = _gh_check_output(cmd)

    # Parse run ID from dispatch URL
    match_obj = re.search(r"/actions/runs/(\d+)", dispatch_output)
    if match_obj:
        new_run_id = int(match_obj.group(1))
    else:
        # Deterministic polling for the freshly dispatched run
        if DRY_RUN:
            # In dry-run mode, we don't poll or parse outputs
            new_run_id = 0
        else:
            end_time = time.time() + 60  # up to 60s
            statuses = {"queued", "in_progress"}
            new_run_id = None
            while time.time() < end_time:
                list_output = _gh_check_output([
                    "gh", "run", "list",
                    "--repo", repo,
                    "--workflow", workflow_filename,
                    "--branch", branch,
                    "--limit", "50",
                    "--json", "databaseId,headBranch,workflowName,status,createdAt"
                ])
                try:
                    runs = json.loads(list_output or "[]")
                except json.JSONDecodeError:
                    runs = []
                # Filter by status and createdAt >= since
                candidates = []
                for run in runs:
                    try:
                        created = _parse_github_time_iso8601(run.get("createdAt", ""))
                    except (ValueError, TypeError):
                        continue
                    if run.get("status") in statuses and created >= pre_dispatch_time:
                        candidates.append((created, int(run.get("databaseId"))))
                if candidates:
                    # Choose the earliest created run after dispatch
                    candidates.sort(key=lambda x: x[0])
                    new_run_id = candidates[0][1]
                    break
                time.sleep(2)
        if new_run_id is None:
            logging.error("Unable to deterministically locate new run for %s on %s.", step_key, repo)
            sys.exit(1)

    if DRY_RUN:
        # Don't write or log anything extra in dry-run
        return new_run_id

    state[step_key] = {"run_id": new_run_id, "completed": False}
    save_state(state)
    logging.info("Recorded new %s run %s on %s", step_key, new_run_id, repo)
    return new_run_id


def wait_for_success(
    step_key: str,
    state: Dict[str, Dict[str, Any]],
    repo: str
) -> None:
    """
    Block until the specified workflow run completes successfully.
    """
    state_entry     = state.get(step_key, {})
    workflow_run_id = state_entry.get("run_id")

    if not workflow_run_id:
        if not DRY_RUN:
            logging.warning("No run_id for %s, skipping wait.", step_key)
        return
    if state_entry.get("completed") and state_entry.get("conclusion") == "success":
        return

    try:
        _gh_run(["gh", "run", "watch", str(workflow_run_id), "--repo", repo])
    except KeyboardInterrupt:
        logging.info("Interrupted by user during watch; exiting gracefully.")
        sys.exit(0)

    conclusion = get_workflow_conclusion(str(workflow_run_id), repo)
    if conclusion != "success":
        logging.error(
            "Run %s for %s on %s ended with '%s'. Aborting.",
            workflow_run_id, step_key, repo, conclusion
        )
        sys.exit(1)

    state_entry["completed"]  = True
    state_entry["conclusion"] = conclusion
    save_state(state)


def download_artifacts(
    step_key: str,
    state: Dict[str, Dict[str, Any]],
    repo: str,
    destination: Path
) -> None:
    """
    Download artifacts from a successful workflow run.
    """
    state_entry     = state.get(step_key, {})
    workflow_run_id = state_entry.get("run_id")

    if not workflow_run_id or state_entry.get("conclusion") != "success":
        logging.error("Cannot download %s: no successful run on %s.", step_key, repo)
        sys.exit(1)

    destination.mkdir(parents=True, exist_ok=True)
    if not DRY_RUN:
        logging.info(
            "Downloading %s artifacts from run %s on %s to %s",
            step_key, workflow_run_id, repo, destination
        )
    _gh_run([
        "gh", "run", "download", str(workflow_run_id),
        "--repo", repo,
        "--dir", str(destination)
    ])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Manage conda-only CI workflows for llvmlite & numba"
    )
    parser.add_argument(
        "--llvmlite-branch", default=DEFAULT_BRANCH,
        help="Git branch or tag for llvmlite workflows"
    )
    parser.add_argument(
        "--numba-branch", default=DEFAULT_BRANCH,
        help="Git branch or tag for numba workflows"
    )
    parser.add_argument(
        "--state-path",
        default=str((Path(__file__).parent / ".ci_state.json").resolve()),
        help="Path to state file (.ci_state.json)"
    )
    parser.add_argument(
        "--steps", default="all",
        help=",".join(sorted(ALL_STEPS)) + ",all"
    )
    parser.add_argument(
        "--reuse-run", action="append",
        help="STEP:RUN_ID to seed the state manually"
    )
    parser.add_argument(
        "--reset-state", action="store_true",
        help="Delete existing .ci_state.json before executing"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print intended gh commands without executing"
    )
    args = parser.parse_args()

    # Configure state path and dry-run without using 'global'
    setattr(sys.modules[__name__], "STATE_FILE", Path(args.state_path).expanduser().resolve())
    setattr(sys.modules[__name__], "DRY_RUN", args.dry_run)

    # If requested, only reset state and exit immediately
    if args.reset_state:
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()
                logging.info("Removed existing state file: %s", STATE_FILE)
            except OSError as exc:
                logging.error("Failed to remove state file %s: %s", STATE_FILE, exc)
                sys.exit(1)
        else:
            logging.info("No state file found at %s; nothing to reset", STATE_FILE)
        sys.exit(0)

    if not DRY_RUN and not os.getenv("GH_TOKEN") and not _is_gh_authenticated():
        logging.error("Authentication required: set GH_TOKEN or login via 'gh auth login'.")
        sys.exit(1)

    requested_steps: Set[str] = {s.strip() for s in args.steps.split(',')}
    # Warn on unknown steps
    raw_requested_steps: Set[str] = {s.strip() for s in args.steps.split(',')}
    invalid_steps = {s for s in raw_requested_steps if s and s != "all" and s not in ALL_STEPS}
    if invalid_steps:
        logging.warning("Ignoring unknown steps: %s. Valid steps: %s", ", ".join(sorted(invalid_steps)), ", ".join(sorted(ALL_STEPS)))

    if "all" in requested_steps:
        requested_steps = ALL_STEPS.copy()

    state = load_state()

    # Seed manual runs
    if args.reuse_run:
        for seed in args.reuse_run:
            try:
                seed_step, seed_id = seed.split(":")
                state[seed_step] = {"run_id": int(seed_id), "completed": False}
                logging.info("Seeded %s with run %s", seed_step, seed_id)
            except ValueError:
                logging.error("Invalid --reuse-run format: %s", seed)
                sys.exit(1)
        save_state(state)

    # Step: llvmdev on LLVMLITE_REPO
    if "llvmdev" in requested_steps:
        dispatch_or_reuse(
            "llvmdev_build.yml",
            {"platform": "all", "recipe": "all"},
            "llvmdev",
            state,
            LLVMLITE_REPO,
            args.llvmlite_branch
        )
        if not DRY_RUN:
            wait_for_success("llvmdev", state, LLVMLITE_REPO)

    # Step: llvmlite_conda on LLVMLITE_REPO
    if "llvmlite_conda" in requested_steps:
        ll_inputs: Dict[str, str] = {"platform": "all"}
        llvm_entry = state.get("llvmdev", {})
        if llvm_entry.get("run_id") and llvm_entry.get("conclusion") == "success":
            ll_inputs["llvmdev_run_id"] = str(llvm_entry["run_id"])
            logging.info("Using local llvmdev run for llvmlite_conda")
        dispatch_or_reuse(
            "llvmlite_conda_builder.yml",
            ll_inputs,
            "llvmlite_conda",
            state,
            LLVMLITE_REPO,
            args.llvmlite_branch
        )
        if not DRY_RUN:
            wait_for_success("llvmlite_conda", state, LLVMLITE_REPO)


    # Step: llvmlite_wheels on LLVMLITE_REPO
    if "llvmlite_wheels" in requested_steps:
        # Prepare inputs using the llvmdev run (needed for local artifact reuse)
        wheel_inputs: Dict[str, str] = {}
        llvm_entry = state.get("llvmdev", {})
        if llvm_entry.get("run_id") and llvm_entry.get("conclusion") == "success":
            wheel_inputs["llvmdev_run_id"] = str(llvm_entry["run_id"])
            logging.info("Using local llvmdev run for llvmlite_wheels: %s", llvm_entry["run_id"])

        # Dispatch all platform-specific llvmlite wheel workflows
        wheel_steps: List[str] = []
        for platform_name in PLATFORMS:
            workflow_file = f"llvmlite_{platform_name}_wheel_builder.yml"
            step_key = f"llvmlite_wheels_{platform_name}"
            dispatch_or_reuse(
                workflow_file,
                wheel_inputs,
                step_key,
                state,
                LLVMLITE_REPO,
                args.llvmlite_branch
            )
            wheel_steps.append(step_key)

        # Monitor each to completion
        if not DRY_RUN:
            for step_key in wheel_steps:
                wait_for_success(step_key, state, LLVMLITE_REPO)


    # Step: numba_conda on NUMBA_REPO
    if "numba_conda" in requested_steps:
        # Prepare inputs using the llvmlite_conda run
        numba_inputs: Dict[str, str] = {}
        llconda_entry = state.get("llvmlite_conda", {})
        if llconda_entry.get("run_id") and llconda_entry.get("conclusion") == "success":
            numba_inputs["llvmlite_run_id"] = str(llconda_entry["run_id"])
            logging.info("Using llvmlite_conda run for numba_conda: %s", llconda_entry["run_id"])

        # Dispatch all platform-specific numba conda workflows
        dispatched_steps: List[str] = []
        for platform_name in PLATFORMS:
            # Map linux-aarch64 to linux-arm64
            workflow_platform = "linux-arm64" if platform_name == "linux-aarch64" else platform_name
            # Windows builder vs other platforms
            if platform_name == "win-64":
                workflow_file = f"numba_{workflow_platform}_builder.yml"
            else:
                workflow_file = f"numba_{workflow_platform}_conda_builder.yml"
            step_key = f"numba_conda_{platform_name}"
            dispatch_or_reuse(
                workflow_file,
                numba_inputs,
                step_key,
                state,
                NUMBA_REPO,
                args.numba_branch
            )
            dispatched_steps.append(step_key)

        # After triggering all, monitor each to completion
        if not DRY_RUN:
            for step_key in dispatched_steps:
                wait_for_success(step_key, state, NUMBA_REPO)

    # Step: numba_wheels on NUMBA_REPO
    if "numba_wheels" in requested_steps:
        # For each platform, optionally use the matching llvmlite_wheels run
        numba_wheel_steps: List[str] = []
        for platform_name in PLATFORMS:
            workflow_platform = "linux-arm64" if platform_name == "linux-aarch64" else platform_name
            workflow_file = f"numba_{workflow_platform}_wheel_builder.yml"

            wheel_inputs: Dict[str, str] = {}
            ll_step_key = f"llvmlite_wheels_{platform_name}"
            ll_entry = state.get(ll_step_key, {})
            if ll_entry.get("run_id") and ll_entry.get("conclusion") == "success":
                wheel_inputs["llvmlite_wheel_runid"] = str(ll_entry["run_id"])
                logging.info("Using %s run for numba_wheels %s: %s", ll_step_key, platform_name, ll_entry["run_id"])

            step_key = f"numba_wheels_{platform_name}"
            dispatch_or_reuse(
                workflow_file,
                wheel_inputs,
                step_key,
                state,
                NUMBA_REPO,
                args.numba_branch
            )
            numba_wheel_steps.append(step_key)

        # Monitor each to completion
        if not DRY_RUN:
            for step_key in numba_wheel_steps:
                wait_for_success(step_key, state, NUMBA_REPO)

    # Step: download llvmlite conda artifacts
    if "download_llvmlite_conda" in requested_steps and not DRY_RUN:
        download_artifacts(
            "llvmlite_conda",
            state,
            LLVMLITE_REPO,
            Path("artifacts/llvmlite_conda")
        )

    # Step: download llvmlite wheel artifacts
    if "download_llvmlite_wheels" in requested_steps and not DRY_RUN:
        for step_key in state:
            if step_key.startswith("llvmlite_wheels_"):
                download_artifacts(
                    step_key,
                    state,
                    LLVMLITE_REPO,
                    Path("artifacts/llvmlite_wheels") / step_key
                )

    # Step: download numba conda artifacts
    if "download_numba_conda" in requested_steps and not DRY_RUN:
        for step_key in state:
            if step_key.startswith("numba_conda_"):
                download_artifacts(
                    step_key,
                    state,
                    NUMBA_REPO,
                    Path("artifacts/numba_conda") / step_key
                )

    # Step: download numba wheel artifacts
    if "download_numba_wheels" in requested_steps and not DRY_RUN:
        for step_key in state:
            if step_key.startswith("numba_wheels_"):
                download_artifacts(
                    step_key,
                    state,
                    NUMBA_REPO,
                    Path("artifacts/numba_wheels") / step_key
                )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
        sys.exit(0)
