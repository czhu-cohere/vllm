# %%
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Hardcoded input path
# original run
results_path = Path("/root/vllm/prof_20260108_115316/bench_results.json")
# flashinfer ep env + uniform routing
results_path = Path("/root/vllm/prof_20260108_124908/bench_results.json")
# rerun of above
results_path = Path("/root/vllm/prof_20260108_131257/bench_results.json")
# fp8
results_path = Path("/root/vllm/prof_20260109_115735/bench_results.json")
# fp8 but also include tp1
results_path = Path("/root/vllm/prof_20260109_121103/bench_results.json")

# Load JSONL (one dict per line) OR JSON array
text = results_path.read_text(encoding="utf-8").strip()
rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
df = pd.DataFrame(rows)

# Normalize needed columns
df["tp"] = pd.to_numeric(df.get("tp"), errors="coerce")
df["ep"] = pd.to_numeric(df.get("expert_parallel", 0), errors="coerce").fillna(0).astype(int)

# Pick which bench you want TTFT from:
# - prefill usually most meaningful for TTFT
# - decode can also have TTFT (queueing + first token)
bench = "prefill" if (df.get("bench") is not None and (df["bench"] == "prefill").any()) else "decode"

metric = "median_ttft_ms"
sub = df[(df["bench"] == bench) & df["tp"].notna() & df["ep"].notna() & df[metric].notna()].copy()

# If you have repeated runs per config, average them; otherwise this is a no-op.
agg = (
    sub.groupby(["tp", "ep"], as_index=False)[metric]
       .mean()
       .sort_values(["ep", "tp"])
)

# Pivot into two series: ep=0 and ep=1
piv = agg.pivot(index="tp", columns="ep", values=metric).sort_index()

plt.figure(figsize=(7.5, 4.5))
if 0 in piv.columns:
    plt.plot(piv.index, piv[0], marker="o", label="TP only (ep=0)")
if 1 in piv.columns:
    plt.plot(piv.index, piv[1], marker="o", label="TP + Expert Parallel (ep=1)")

plt.xticks(sorted(piv.index.dropna().astype(int).tolist()))
plt.xlabel("# GPUs (tensor parallel size)")
plt.ylabel("Median TTFT (ms)")
plt.title(f"Median TTFT vs GPUs ({bench})")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: show the underlying numbers
piv

# %%
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Load JSONL (one dict per line) OR JSON array
text = results_path.read_text(encoding="utf-8").strip()
rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
df = pd.DataFrame(rows)

# Normalize needed columns
df["tp"] = pd.to_numeric(df.get("tp"), errors="coerce")
df["ep"] = pd.to_numeric(df.get("expert_parallel", 0), errors="coerce").fillna(0).astype(int)

# TPOT is most meaningful for decode
bench = "decode" if (df.get("bench") is not None and (df["bench"] == "decode").any()) else "prefill"

metric = "median_tpot_ms"
sub = df[(df["bench"] == bench) & df["tp"].notna() & df["ep"].notna() & df[metric].notna()].copy()

# Average in case of repeated runs per config
agg = (
    sub.groupby(["tp", "ep"], as_index=False)[metric]
       .mean()
       .sort_values(["ep", "tp"])
)

# Pivot into two series: ep=0 and ep=1
piv = agg.pivot(index="tp", columns="ep", values=metric).sort_index()

plt.figure(figsize=(7.5, 4.5))
if 0 in piv.columns:
    plt.plot(piv.index, piv[0], marker="o", label="TP only (ep=0)")
if 1 in piv.columns:
    plt.plot(piv.index, piv[1], marker="o", label="TP + Expert Parallel (ep=1)")

plt.xticks(sorted(piv.index.dropna().astype(int).tolist()))
plt.xlabel("# GPUs (tensor parallel size)")
plt.ylabel("Median TPOT (ms)")
plt.title(f"Median TPOT vs GPUs ({bench})")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: show the underlying numbers
piv

# %%
