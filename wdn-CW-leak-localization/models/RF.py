
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ============================================================
#  RANDOM FOREST - LEAK DETECTION
# ============================================================

class RandomForestLeakOnsetDetector:
    """
    Classificatore binario per prevedere se in uno step
    è appena iniziato un LEAK.
    """

    def __init__(self, n_trees=240, max_depth=20,min_samples_split=2,min_samples_leaf=7,class_weight="balanced",random_state=42):
        
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state
        )

    def fit(self, snapshots):
        X, Y = [], []

        for ep in snapshots:
            ep_steps = ep["feature_vector"]
            leak_start = ep["leak_start"]   # step in cui parte il leak

            for step_idx, data in enumerate(ep_steps):

                # LABEL:
                # 1 SOLO nello step in cui parte il leak
                if leak_start is None:
                    label = 0   # episodio senza leak → SEMPRE 0
                else:
                    label = 1 if step_idx == leak_start else 0

                X.append(data)
                Y.append(label)

        X = np.array(X)
        Y = np.array(Y)

        print("\nTraining RandomForest per leak onset...")
        self.model.fit(X, Y)
        print("RandomForest addestrato.")

    def predict(self, snapshot):
        if hasattr(snapshot, "cpu"):
            snapshot = snapshot.cpu().numpy()

        snapshot = np.asarray(snapshot)

        if snapshot.ndim == 1:
            x = snapshot.reshape(1, -1)
        else:
            x = snapshot.reshape(1, -1)

        return self.model.predict_proba(x)[0, 1]

