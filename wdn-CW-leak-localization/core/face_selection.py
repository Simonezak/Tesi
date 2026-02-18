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

"""

import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix


def compute_global_face_scores(inp_path, B1, B2):
    """
    Returns:
      scores: [F] global face scores (bigger => more relevant)
      episodes: list of EpisodeInfo (for reproducibility/debug)
      cols: junction ordering used for pressures and must match B1 rows

      This uses a PRESSURE residual signal:
      dP(t) = P_leak(t) - P_base(t)
    """
    num_episodes = 50
    num_steps = 50
    leaks_per_episode = 2
    leak_area = 0.1

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # --- create env and cols ONCE (like train_CWGGNN)
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    cols = list(node2idx.keys())

    N, E = B1.shape
    E2, F = B2.shape

    # -------------------------
    # 1) BASELINE (NO LEAK)
    # -------------------------

    env.reset(num_leaks=0)
    sim = env.sim

    for _ in range(num_steps):
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    P_base = df_pressure[cols].to_numpy(dtype=np.float64)  # [T, N]

    # -------------------------
    # 2) EPISODI CON LEAK
    # -------------------------
    episode_scores = np.zeros((num_episodes, F), dtype=np.float64)
    episodes: List[dict] = []

    for ep in range(num_episodes):
        env.reset(num_leaks=leaks_per_episode)

        leak_nodes = tuple(env.leak_node_names)
        leak_start = int(env.leak_start_step) if env.leak_start_step is not None else 0

        sim = env.sim

        for step in range(num_steps):
            if step == leak_start:
                for ln in leak_nodes:
                    sim.start_leak(ln, leak_area=leak_area)

            sim.step_sim()

        results = sim.get_results()
        df_pressure = results.node["pressure"]
        P_leak = df_pressure[cols].to_numpy(dtype=np.float64)  # [T, N]

        # post_leak_only = True (fisso)
        t0 = leak_start
        dP = P_leak[t0:] - P_base[t0:]  # [T', N]

        # node -> edge -> face
        edges = dP @ B1 
        faces = edges @ B2 

        s = np.sum(np.abs(faces), axis=0)  # [F]

        if faces.shape[0] > 0:
            s /= float(faces.shape[0])

        episode_scores[ep, :] = s

    scores = episode_scores.mean(axis=0).astype(np.float64)

    return scores


def reduce_B2_by_scores(B2, scores, drop_frac, min_keep=5):

    if not (0.0 <= drop_frac < 1.0):
        raise ValueError("drop_frac must be in [0,1).")

    F = B2.shape[1]
    k_keep = max(min_keep, int(np.round((1.0 - drop_frac) * F)))
    k_keep = min(k_keep, F)

    keep_idx = np.argsort(scores)[-k_keep:]
    keep_idx = np.sort(keep_idx)
    return B2[:, keep_idx], keep_idx

