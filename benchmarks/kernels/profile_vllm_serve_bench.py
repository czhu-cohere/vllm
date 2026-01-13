#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
import json
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class RunConfig:
    model: str
    tp: int
    expert_parallel: bool
    port: int
    profile: bool


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def http_get_json(url: str, timeout_s: float = 2.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "profile-vllm-script"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def wait_for_server_ready(port: int, timeout_s: int = 600, poll_s: float = 1.0) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while True:
        js = http_get_json(url, timeout_s=2.0)
        if js is not None:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Server not ready after {timeout_s}s (polling {url})")
        time.sleep(poll_s)


def start_server(cfg: RunConfig, run_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    server_log = run_dir / "server.log"
    cmd = [
        "vllm", "serve", cfg.model,
        "--no-enable-prefix-caching",
        "--load-format", "dummy",
        "--tensor-parallel-size", str(cfg.tp),
        "--port", str(cfg.port),
    ]
    if cfg.expert_parallel:
        cmd.append("--enable-expert-parallel")
        env["VLLM_USE_FLASHINFER_MOE_FP16"] = "1"
    
    if cfg.profile:
        env["VLLM_TORCH_PROFILER_DIR"] = run_dir / "torch_profiler"

    # set uniform routing, it might make a difference in some cases
    env["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"
    # try flashinfer for fp8 - this doesnt work for ptpc?
    # env["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"
    (run_dir / "server_cmd.txt").write_text(" ".join(cmd) + "\n")

    f = open(server_log, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env=env,
    )
    return proc


def kill_process_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    try:
        if hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        if hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def run_bench(
    *,
    model: str,
    port: int,
    bench_kind: str,              # "prefill" or "decode"
    bench_cmd_base: List[str],    # list of args after "vllm bench serve"
    result_dir: Path,
    result_filename: str,
    metadata: Dict[str, str],
    log_path: Path,
    profile: bool,
) -> int:
    """
    Runs: vllm bench serve <base args> --save-result --append-result --result-dir ... --result-filename ...
          --metadata k=v k=v ...
    """
    cmd = ["vllm", "bench", "serve"] + bench_cmd_base + [
        "--save-result",
        "--append-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]

    if profile:
        cmd.append("--profile")

    # Metadata: also include bench kind & port for convenience
    md_items = dict(metadata)
    md_items["bench"] = bench_kind
    md_items["port"] = str(port)

    # vllm bench takes: --metadata KEY=VALUE (nargs="*")
    md_args: List[str] = []
    for k, v in md_items.items():
        md_args.append(f"{k}={v}")

    cmd += ["--metadata"] + md_args

    # Save the exact command we ran
    (log_path.parent / f"bench_{bench_kind}_cmd.txt").write_text(" ".join(cmd) + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/engines/c5-8a133t-bf16-dummy-shallow/")
    ap.add_argument("--tps", default="2,4,8", help="Comma-separated TP sizes")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=f"prof_{now_tag()}", help="Output directory")
    ap.add_argument("--ready-timeout-s", type=int, default=600)
    ap.add_argument("--profile", action="store_true", help="Whether to enable profiling in the server")
    args = ap.parse_args()

    tps = [int(x.strip()) for x in args.tps.split(",") if x.strip()]
    out_root = Path(args.out)
    mkdir(out_root)

    # Base args for each bench type (these are the args after "vllm bench serve")
    # for torch profiling traces do less stuff
    prefill_args = [
        "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", "8192",
        "--random-output-len", "1",
        "--ignore-eos",
        "--max-concurrency", "1",
        "--num-prompts", "5" if not args.profile else "1",
    ]
    decode_args = [
        "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", "16",
        "--random-output-len", "100" if not args.profile else "10",
        "--ignore-eos",
        "--max-concurrency", "32",
        "--num-prompts", "32",
    ]

    # All bench results for ALL runs go into one JSON file (append mode).
    # Each bench invocation adds a new record, and we add metadata to identify it.
    result_dir = out_root
    result_filename = "bench_results.json"

    configs: List[RunConfig] = []
    for tp in tps:
        for ep in [False, True]:
            configs.append(RunConfig(model=args.model, tp=tp, expert_parallel=ep, port=args.port, profile=args.profile))

    # High-level manifest for convenience (optional, but nice)
    manifest_path = out_root / "manifest.jsonl"

    for idx, cfg in enumerate(configs, start=1):
        cfg_tag = f"tp{cfg.tp}_ep{int(cfg.expert_parallel)}"
        run_dir = out_root / f"{idx:02d}_{cfg_tag}"
        mkdir(run_dir)

        print(f"\n=== [{idx}/{len(configs)}] {cfg_tag} ===", flush=True)
        server = start_server(cfg, run_dir)

        try:
            print(f"Waiting for readiness on port {cfg.port} ...", flush=True)
            wait_for_server_ready(cfg.port, timeout_s=args.ready_timeout_s, poll_s=1.0)
            print("Server ready.", flush=True)

            common_md = {
                "tp": str(cfg.tp),
                "expert_parallel": str(int(cfg.expert_parallel)),
                "config_tag": cfg_tag,
            }

            # Prefill bench
            print("Running prefill bench ...", flush=True)
            rc1 = run_bench(
                model=cfg.model,
                port=cfg.port,
                bench_kind="prefill",
                bench_cmd_base=prefill_args,
                result_dir=result_dir,
                result_filename=result_filename,
                metadata=common_md,
                log_path=run_dir / "bench_prefill.log",
                profile=cfg.profile,
            )

            # Decode bench
            print("Running decode bench ...", flush=True)
            rc2 = run_bench(
                model=cfg.model,
                port=cfg.port,
                bench_kind="decode",
                bench_cmd_base=decode_args,
                result_dir=result_dir,
                result_filename=result_filename,
                metadata=common_md,
                log_path=run_dir / "bench_decode.log",
                profile=cfg.profile,
            )

            # Record a simple manifest line (so you can quickly see failures)
            manifest_rec: Dict[str, Any] = {
                "timestamp": dt.datetime.now().isoformat(),
                "config": asdict(cfg),
                "config_tag": cfg_tag,
                "prefill_rc": rc1,
                "decode_rc": rc2,
                "run_dir": str(run_dir),
                "result_file": str(result_dir / result_filename),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_rec) + "\n")

        except Exception as e:
            (run_dir / "ERROR.txt").write_text(f"{type(e).__name__}: {e}\n")
            print(f"ERROR in {cfg_tag}: {e}", file=sys.stderr, flush=True)

        finally:
            print("Stopping server ...", flush=True)
            if cfg.profile:
                # wait some time to let profiler flush data. i think 30s enough
                time.sleep(30.0)
            kill_process_group(server, grace_s=10.0)
            time.sleep(2.0)

    print("\nDone.")
    print(f"- Output dir:   {out_root}")
    print(f"- Bench JSON:   {out_root / result_filename}")
    print(f"- Manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
