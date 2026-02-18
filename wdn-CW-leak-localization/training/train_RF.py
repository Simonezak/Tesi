
import os
import pickle
import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from models.RF import RandomForestLeakOnsetDetector


def train_RF(inp_path, num_episodes = 200, num_steps = 50, leak_area = 0.1, n_trees=240, max_depth=20,min_samples_split=2,min_samples_leaf=7,class_weight="balanced",random_state=42):

    env = WNTREnv(inp_path, num_steps=num_steps)

    rf = RandomForestLeakOnsetDetector(n_trees, max_depth, min_samples_split, min_samples_leaf, class_weight, random_state)
    rf_training_data = []

    _, node2idx, _ = build_adj_matrix(env.wn)
    cols = list(node2idx.keys())

    print("\n=== TRAINING RANDOM FOREST ===")

    # ====== TRAIN LOOP ======
    for ep in range(num_episodes):

        print(f"\n--- Episodio {ep+1}/{num_episodes}")

        num_leaks = random.randint(0, 2)
        env.reset(num_leaks)
        sim = env.sim

        episode_pressures = []

        for step in range(num_steps):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=leak_area)

            sim.step_sim()
        
        results = sim.get_results()
        df_pressure = results.node["pressure"]

        episode_pressures = df_pressure[cols].to_numpy(dtype=np.float32)

        rf_training_data.append({
            "feature_vector": episode_pressures,
            "leak_start": env.leak_start_step
        })

    rf.fit(rf_training_data)

    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/rf_leak_onset.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("\n[OK] Random Forest salvata come rf_leak_onset.pkl")


if __name__ == "__main__":

    # PARAMETRI
    inp_path= r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"

    num_episodes = 300
    num_steps = 50
    leak_area = 0.1

    # IPERPARAMETRI
    n_trees = 240 
    max_depth = 20
    min_samples_split = 2
    min_samples_leaf = 7
    class_weight = "balanced"
    random_state = 42

    train_RF(inp_path, num_episodes, num_steps, leak_area)
