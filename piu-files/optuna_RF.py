import optuna
import numpy as np
import pickle

from wntr_exp_multi import WNTREnv
from GGNN_multi import RandomForestLeakOnsetDetector
from optuna_GGNN import leak_detection_error

def estimate_onset_from_scores(
    onset_scores,
    threshold=0.5
):
    """
    Stima onset come primo superamento di soglia.
    """
    for t, p in enumerate(onset_scores):
        if p >= threshold:
            return t
    return len(onset_scores) - 1


def objective(trial):

    n_trees = trial.suggest_int("n_trees", 50, 400)
    max_depth = trial.suggest_int("max_depth", 4, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    num_train_episodes = 30
    num_test = 20
    max_steps = 50

    env = WNTREnv(inp_path, max_steps=max_steps)

    # ================= TRAIN =================
    rf_training_data = []

    for _ in range(num_train_episodes):
        n_leaks = np.random.randint(1, 3)
        env.reset(num_leaks=n_leaks)
        sim = env.sim

        for step in range(max_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=0.1)
            sim.step_sim()

        df_pressure = sim.get_results().node["pressure"]

        rf_training_data.append({
            "feature_vector": df_pressure.to_numpy(dtype=np.float32),
            "leak_start": env.leak_start_step
        })

    rf = RandomForestLeakOnsetDetector(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    rf.fit(rf_training_data)

    # ================= TEST =================
    det_errors = []

    for _ in range(num_test):
        n_leaks = np.random.randint(1, 3)
        env.reset(num_leaks=n_leaks)
        sim = env.sim

        for step in range(max_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=0.1)
            sim.step_sim()

        df_pressure = sim.get_results().node["pressure"]

        onset_scores = [
            rf.predict(df_pressure.iloc[t].to_numpy(dtype=np.float32))
            for t in range(len(df_pressure))
        ]

        predicted_onset = int(np.argmax(onset_scores))

        det_errors.append(
            abs(predicted_onset - env.leak_start_step)
        )

    return float(np.mean(det_errors))



if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        study_name="RF_Leak_Onset_Optimization",
        storage="sqlite:///optuna_rf_leak.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=30)

    print("\n=== OPTUNA RF FINISHED ===")
    print("Best mean:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


