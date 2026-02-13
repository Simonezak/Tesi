"""
Global face (2-cell) selection for pruning columns of B2, using the SAME
simulation pattern used in train_CWGGNN.py.

Key points (same as training)
-----------------------------
- Use WNTREnv(inp_path, num_steps=...)
- Build cols from build_adj_matrix(env.wn) ONCE
- For each episode:
    env.reset(num_leaks)
    (optionally override env.leak_start_step like training does)
    run for step in range(num_steps):
        if step == leak_start: sim.start_leak(...)
        sim.step_sim()
    results = sim.get_results()
    df_pressure = results.node["pressure"]
- Build a node signal and score faces via:
    dP = P_leak - P_base
    edges = dP @ B1          # [T, E]
    faces = edges @ B2       # [T, F]
    score_face = sum_t |faces_t|

This file is deterministic if you pass a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix

AggMode = Literal["mean_abs", "p90_abs"]


@dataclass(frozen=True)
class EpisodeInfo:
    num_leaks: int
    leak_nodes: Tuple[str, ...]
    leak_start_step: int
    leak_area: float


def _run_episode_get_pressure(
    env: WNTREnv,
    cols: Sequence[str],
    *,
    num_steps: int,
    leak_nodes: Optional[Sequence[str]],
    leak_start_step: Optional[int],
    leak_area: float,
) -> np.ndarray:
    """
    Run one episode with the same structure as train_CWGGNN and return
    pressure array [T, N] aligned to 'cols'.
    """
    sim = env.sim

    for step in range(num_steps):
        if leak_nodes is not None and leak_start_step is not None and step == leak_start_step:
            for ln in leak_nodes:
                sim.start_leak(ln, leak_area=leak_area)
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    return df_pressure.iloc[:num_steps][list(cols)].values.astype(np.float64)


def compute_global_face_scores_pretrain(
    inp_path: str,
    B1: np.ndarray,  # [N, E]
    B2: np.ndarray,  # [E, F]
    *,
    num_steps: int = 50,
    hydraulic_timestep: int = 3600,
    num_episodes: int = 50,
    leaks_per_episode: Tuple[int, int] = (2, 3),
    leak_area: float = 0.1,
    seed: int = 0,
    # train_CWGGNN forces leak_start_step to 20; replicate with leak_start_step_fixed=20
    leak_start_step_fixed: Optional[int] = 20,
    post_leak_only: bool = True,
    per_episode_normalize: bool = True,
    agg: AggMode = "mean_abs",
) -> Tuple[np.ndarray, List[EpisodeInfo], List[str]]:
    """
    Returns:
      scores: [F] global face scores (bigger => more relevant)
      episodes: list of EpisodeInfo (for reproducibility/debug)
      cols: junction ordering used for pressures and must match B1 rows

    NOTE: This uses a PRESSURE residual signal:
      dP(t) = P_leak(t) - P_base(t)
    """

    # --- determinism, matching train setup (python random for num_leaks, numpy for env.reset internals)
    random.seed(seed)
    np.random.seed(seed)

    # --- create env and cols ONCE (like train_CWGGNN)
    env = WNTREnv(inp_path, num_steps=num_steps, hydraulic_timestep=hydraulic_timestep)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    cols = list(node2idx.keys())

    N, E = B1.shape
    E2, F = B2.shape

    if len(cols) != N:
        raise ValueError(
            f"Mismatch: len(cols)={len(cols)} but B1 has N={N} rows. "
            "Ensure B1 was built using the same junction ordering (node2idx) used here."
        )
    if E2 != E:
        raise ValueError(f"Mismatch: B2 rows={E2} but B1 cols={E}.")

    # --- baseline episode (NO leak), same simulation pattern
    env.reset(num_leaks=0)
    P_base = _run_episode_get_pressure(
        env,
        cols,
        num_steps=num_steps,
        leak_nodes=None,
        leak_start_step=None,
        leak_area=leak_area,
    )  # [T, N]

    # --- run leak episodes and collect per-episode face scores
    episode_scores = np.zeros((num_episodes, F), dtype=np.float64)
    episodes: List[EpisodeInfo] = []

    a, b = leaks_per_episode

    for i in range(num_episodes):
        num_leaks = random.randint(a, b)

        env.reset(num_leaks=num_leaks)

        # replicate training behaviour: override leak_start_step deterministically
        if leak_start_step_fixed is not None:
            env.leak_start_step = int(leak_start_step_fixed)

        leak_nodes = tuple(env.leak_node_names)
        leak_start = int(env.leak_start_step) if env.leak_start_step is not None else 0

        P_leak = _run_episode_get_pressure(
            env,
            cols,
            num_steps=num_steps,
            leak_nodes=leak_nodes,
            leak_start_step=leak_start,
            leak_area=leak_area,
        )  # [T, N]

        t0 = leak_start if post_leak_only else 0
        dP = (P_leak[t0:] - P_base[t0:])  # [T', N]

        # node -> edge -> face
        edges = dP @ B1          # [T', E]
        faces = edges @ B2       # [T', F]

        s = np.sum(np.abs(faces), axis=0)  # [F]
        if per_episode_normalize and faces.shape[0] > 0:
            s = s / float(faces.shape[0])

        episode_scores[i, :] = s

        episodes.append(
            EpisodeInfo(
                num_leaks=num_leaks,
                leak_nodes=leak_nodes,
                leak_start_step=leak_start,
                leak_area=float(leak_area),
            )
        )

    # --- aggregate across episodes to get global face score
    if agg == "mean_abs":
        scores = episode_scores.mean(axis=0)
    elif agg == "p90_abs":
        scores = np.percentile(episode_scores, 90, axis=0)
    else:
        raise ValueError(f"Unsupported agg={agg}")

    return scores, episodes, cols


def prune_B2_by_scores(
    B2: np.ndarray,
    scores: np.ndarray,
    *,
    drop_frac: float = 0.2,
    min_keep: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the top (1-drop_frac) faces by score.
    Returns:
        B2_small: [E, F_small]
        keep_idx: [F_small] sorted indices of kept columns
    """
    if not (0.0 <= drop_frac < 1.0):
        raise ValueError("drop_frac must be in [0,1).")

    F = B2.shape[1]
    k_keep = max(min_keep, int(np.round((1.0 - drop_frac) * F)))
    k_keep = min(k_keep, F)

    keep_idx = np.argsort(scores)[-k_keep:]
    keep_idx = np.sort(keep_idx)
    return B2[:, keep_idx], keep_idx


def save_face_selection_npz(
    out_path: str,
    keep_idx: np.ndarray,
    scores: np.ndarray,
    episodes: Sequence[EpisodeInfo],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save selection info so train/test can reuse the same B2 subset.
    """
    meta = {} if meta is None else dict(meta)

    leak_nodes = np.array([";".join(ep.leak_nodes) for ep in episodes], dtype=object)
    leak_start = np.array([ep.leak_start_step for ep in episodes], dtype=np.int32)
    leak_area = np.array([ep.leak_area for ep in episodes], dtype=np.float64)
    num_leaks = np.array([ep.num_leaks for ep in episodes], dtype=np.int32)

    np.savez_compressed(
        out_path,
        keep_idx=keep_idx.astype(np.int64),
        scores=scores.astype(np.float64),
        leak_nodes=leak_nodes,
        leak_start_step=leak_start,
        leak_area=leak_area,
        num_leaks=num_leaks,
        meta=np.array([meta], dtype=object),
    )
