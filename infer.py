"""
Standalone POMO single-trajectory greedy inference script for LEHD-format CVRP test files.

Runs in "single trajectory, no augment" mode as reported in the POMO paper:
  - pomo_size = 1  (one greedy rollout from the depot; no POMO multi-start fan-out)
  - augmentation_enable = False  (no 8× coordinate augmentation)
  - eval_type = argmax  (deterministic greedy, not sampling)

Usage examples:
  python infer.py --test-file "/home/jkschin/orcd/pool/CVRP/testing dataset/vrp100_test_lkh.txt" \
      --model-path NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt \
      --problem-size 100

  python infer.py --test-file "/home/jkschin/orcd/pool/CVRP/testing dataset/vrp200_test_lkh.txt" \
      --model-path /path/to/cvrp200_checkpoint.pt \
      --problem-size 200 --batch-size 32

  # Use CPU explicitly:
  python infer.py --test-file ... --model-path ... --problem-size 100 --no-cuda
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make POMO source importable regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_DIR = os.path.join(_SCRIPT_DIR, "NEW_py_ver", "CVRP", "POMO")
_CVRP_DIR = os.path.join(_SCRIPT_DIR, "NEW_py_ver", "CVRP")
_UTILS_DIR = os.path.join(_SCRIPT_DIR, "NEW_py_ver")

for _p in [_POMO_DIR, _CVRP_DIR, _UTILS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from CVRPEnv import CVRPEnv          # noqa: E402
from CVRPModel import CVRPModel      # noqa: E402


# ---------------------------------------------------------------------------
# LEHD file parser  (mirrors lehd.py logic, self-contained here)
# ---------------------------------------------------------------------------

def parse_lehd_line(line: str) -> dict:
    """Parse one line from an LEHD-format CVRP file."""
    tokens = line.strip().split(",")

    idx_depot    = tokens.index("depot")
    idx_customer = tokens.index("customer")
    idx_capacity = tokens.index("capacity")
    idx_demand   = tokens.index("demand")
    idx_cost     = tokens.index("cost")

    depot_coords    = list(map(float, tokens[idx_depot + 1 : idx_customer]))
    customer_coords = list(map(float, tokens[idx_customer + 1 : idx_capacity]))
    coords          = np.array(depot_coords + customer_coords, dtype=np.float32).reshape(-1, 2)

    capacity = int(float(tokens[idx_capacity + 1]))

    # cost token is immediately after the "cost" keyword; demand fills the gap
    demand = np.array(tokens[idx_demand + 1 : idx_cost], dtype=np.float32).astype(int)
    if len(demand) == len(coords) - 1:          # depot demand omitted
        demand = np.insert(demand, 0, 0)

    cost = float(tokens[idx_cost + 1])

    return {
        "depot_xy":   coords[[0]],              # (1, 2)
        "node_xy":    coords[1:],               # (n_customers, 2)
        "demand":     demand[1:] / capacity,    # normalized, customers only  (n_customers,)
        "lkh_cost":   cost,
    }


def load_lehd_file(filepath: str, max_samples: int | None = None):
    """
    Load every instance from an LEHD text file.

    Returns
    -------
    depot_xy    : float32 tensor  (N, 1, 2)
    node_xy     : float32 tensor  (N, problem_size, 2)
    node_demand : float32 tensor  (N, problem_size)
    lkh_costs   : float32 tensor  (N,)
    """
    depot_list, node_list, demand_list, cost_list = [], [], [], []

    with open(filepath, "r") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parsed = parse_lehd_line(line)
            depot_list.append(parsed["depot_xy"])
            node_list.append(parsed["node_xy"])
            demand_list.append(parsed["demand"])
            cost_list.append(parsed["lkh_cost"])
            if max_samples is not None and len(cost_list) >= max_samples:
                break

    depot_xy    = torch.tensor(np.stack(depot_list,  axis=0), dtype=torch.float32)
    node_xy     = torch.tensor(np.stack(node_list,   axis=0), dtype=torch.float32)
    node_demand = torch.tensor(np.stack(demand_list, axis=0), dtype=torch.float32)
    lkh_costs   = torch.tensor(cost_list,                     dtype=torch.float32)

    return depot_xy, node_xy, node_demand, lkh_costs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_greedy_inference(
    depot_xy:    torch.Tensor,   # (N, 1, 2)
    node_xy:     torch.Tensor,   # (N, problem_size, 2)
    node_demand: torch.Tensor,   # (N, problem_size)
    model:       CVRPModel,
    env:         CVRPEnv,
    batch_size:  int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Run a single-trajectory greedy rollout and return per-instance tour length (N,).

    pomo_size must be 1 (single trajectory, no POMO multi-start fan-out).
    """
    assert env.pomo_size == 1, (
        f"pomo_size={env.pomo_size}; set pomo_size=1 for single-trajectory mode"
    )
    N = depot_xy.size(0)
    all_lengths = []

    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            b_depot  = depot_xy   [start : start + batch_size].to(device)
            b_node   = node_xy    [start : start + batch_size].to(device)
            b_demand = node_demand[start : start + batch_size].to(device)
            bsz = b_depot.size(0)

            # ---- feed problem directly into env ----
            env.batch_size = bsz
            env.depot_node_xy = torch.cat((b_depot, b_node), dim=1)
            depot_demand = torch.zeros(bsz, 1, device=device)
            env.depot_node_demand = torch.cat((depot_demand, b_demand), dim=1)

            env.BATCH_IDX = torch.arange(bsz, device=device)[:, None].expand(bsz, env.pomo_size)
            env.POMO_IDX  = torch.arange(env.pomo_size, device=device)[None, :].expand(bsz, env.pomo_size)

            env.reset_state.depot_xy   = b_depot
            env.reset_state.node_xy    = b_node
            env.reset_state.node_demand = b_demand

            env.step_state.BATCH_IDX = env.BATCH_IDX
            env.step_state.POMO_IDX  = env.POMO_IDX

            # ---- rollout ----
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

            # reward shape: (batch, pomo=1)  — negative tour length
            tour_lengths = (-reward).squeeze(dim=1)   # single trajectory: (batch,)
            all_lengths.append(tour_lengths.cpu())

    return torch.cat(all_lengths, dim=0)  # (N,)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="POMO greedy inference on LEHD-format CVRP test files"
    )
    parser.add_argument(
        "--test-file", required=True,
        help="Path to the LEHD text file, e.g. vrp100_test_lkh.txt"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to a POMO checkpoint (.pt file)"
    )
    parser.add_argument(
        "--problem-size", type=int, required=True,
        help="Number of customer nodes (e.g. 100, 200, 500, 1000)"
    )
    parser.add_argument(
        "--pomo-size", type=int, default=1,
        help="POMO rollout width (default: 1 = single trajectory, no multi-start fan-out)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Instances per inference batch (default: 100)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of test instances loaded (default: all)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Force CPU even if CUDA is available"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=0,
        help="CUDA device index (default: 0)"
    )
    # Model architecture (defaults match the published POMO CVRP model)
    parser.add_argument("--embedding-dim",    type=int,   default=128)
    parser.add_argument("--encoder-layers",   type=int,   default=6)
    parser.add_argument("--qkv-dim",          type=int,   default=16)
    parser.add_argument("--head-num",         type=int,   default=8)
    parser.add_argument("--ff-hidden-dim",    type=int,   default=512)
    parser.add_argument("--logit-clipping",   type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()

    pomo_size = args.pomo_size  # default 1 = single trajectory

    # single-trajectory, no augmentation (as per the POMO paper "no augment" row)
    augmentation_enable = False   # no 8× coordinate augmentation
    aug_factor = 1 if not augmentation_enable else 8
    assert aug_factor == 1, "augmentation must be disabled for pure greedy inference"

    # ---- device ----
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        device = torch.device("cuda", args.cuda_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    print(f"Device       : {device}")
    print(f"Test file    : {args.test_file}")
    print(f"Model        : {args.model_path}")
    print(f"Problem size : {args.problem_size}  |  POMO size: {pomo_size}  |  augmentation: {augmentation_enable}")

    # ---- load data ----
    print("\nLoading test instances …", flush=True)
    t0 = time.time()
    depot_xy, node_xy, node_demand, lkh_costs = load_lehd_file(
        args.test_file, max_samples=args.max_samples
    )
    N = depot_xy.size(0)
    actual_problem_size = node_xy.size(1)
    print(f"  Loaded {N} instances  (customers per instance: {actual_problem_size})  [{time.time()-t0:.1f}s]")

    if actual_problem_size != args.problem_size:
        print(
            f"WARNING: --problem-size {args.problem_size} does not match "
            f"actual data size {actual_problem_size}. Using {actual_problem_size}."
        )
        args.problem_size = actual_problem_size

    # ---- build model ----
    model_params = {
        "embedding_dim":     args.embedding_dim,
        "sqrt_embedding_dim": args.embedding_dim ** 0.5,
        "encoder_layer_num": args.encoder_layers,
        "qkv_dim":           args.qkv_dim,
        "head_num":          args.head_num,
        "logit_clipping":    args.logit_clipping,
        "ff_hidden_dim":     args.ff_hidden_dim,
        "eval_type":         "argmax",   # greedy (argmax, not sampling)
    }

    assert aug_factor == 1, "augmentation must be disabled for pure greedy inference"
    assert pomo_size == 1, (
        f"pomo_size={pomo_size}; use --pomo-size 1 for single-trajectory mode"
    )
    env_params = {
        "problem_size": args.problem_size,
        "pomo_size":    pomo_size,
    }

    env   = CVRPEnv(**env_params)
    model = CVRPModel(**model_params)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"  Model loaded from {args.model_path}")

    # ---- inference ----
    print(f"\nRunning single-trajectory greedy inference …", flush=True)
    t1 = time.time()
    tour_lengths = run_greedy_inference(
        depot_xy, node_xy, node_demand,
        model, env,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed = time.time() - t1

    # ---- results ----
    avg_pomo   = tour_lengths.mean().item()
    avg_lkh    = lkh_costs.mean().item()
    gap_pct    = (avg_pomo - avg_lkh) / avg_lkh * 100.0

    print(f"\n{'='*55}")
    print(f"  Instances      : {N}")
    print(f"  Inference time : {elapsed:.2f}s  ({elapsed/N*1000:.1f} ms/instance)")
    print(f"  Avg greedy cost : {avg_pomo:.4f}")
    print(f"  Avg LKH  cost  : {avg_lkh:.4f}")
    print(f"  Gap vs LKH     : {gap_pct:+.2f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
