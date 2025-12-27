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

    # ===============================
    # IPERPARAMETRI RF
    # ===============================
    n_trees = trial.suggest_int("n_trees", 50, 400)
    max_depth = trial.suggest_int("max_depth", 4, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    threshold = trial.suggest_float("threshold", 0.1, 0.9)

    # ===============================
    # TRAINING DATA
    # ===============================
    inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"

    num_train_episodes = 80
    max_steps = 50

    env = WNTREnv(inp_path, max_steps=max_steps)

    rf_training_data = []

    for ep in range(num_train_episodes):
        env.reset(with_leak=True)
        sim = env.sim

        for step in range(max_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=0.1)
            sim.step_sim()

        results = sim.get_results()
        df_pressure = results.node["pressure"]

        rf_training_data.append({
            "feature_vector": df_pressure.to_numpy(dtype=np.float32),
            "leak_start": env.leak_start_step
        })

    # ===============================
    # TRAIN RF
    # ===============================
    rf = RandomForestLeakOnsetDetector(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    rf.fit(rf_training_data)

    # ===============================
    # TEST RF
    # ===============================
    num_test = 30
    det_errors = []

    for _ in range(num_test):
        env.reset(with_leak=True)
        sim = env.sim

        for step in range(max_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=0.1)
            sim.step_sim()

        results = sim.get_results()
        df_pressure = results.node["pressure"]

        onset_scores = []
        for t in range(len(df_pressure)):
            pressures = df_pressure.iloc[t].to_numpy(dtype=np.float32)
            onset_scores.append(rf.predict(pressures))

        predicted_onset = estimate_onset_from_scores(
            onset_scores,
            threshold=threshold
        )

        det_errors.append(
            abs(leak_detection_error(predicted_onset, env.leak_start_step))
        )

    # ===============================
    # METRICA DA MINIMIZZARE
    # ===============================
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
    print("Best mean |Δt|:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


