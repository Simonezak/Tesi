
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    @staticmethod
    def extract_features(snapshot):
        """
        snapshot = PyG Data di build_pyg_from_wntr
        Feature ≡ pressioni nodi + flowrates archi
        """
        pressures = snapshot.x[:, 2].cpu().numpy()        # pressure
        #flows     = snapshot.edge_attr[:, 2].cpu().numpy()  # flowrate
        #vector = np.concatenate([pressures, flows])
        return pressures


    def fit(self, snapshots):
        X, Y = [], []

        for ep in snapshots:
            ep_steps = ep["feature_vector"]
            leak_start = ep["leak_start"]   # step in cui parte il leak

            for step_idx, data in enumerate(ep_steps):

                # LABEL:
                # 1 SOLO nello step in cui parte il leak
                label = 1 if step_idx == leak_start else 0

                X.append(data)
                Y.append(label)

        X = np.array(X)
        Y = np.array(Y)

        print("Training RandomForest per leak onset...")
        self.model.fit(X, Y)
        print("RandomForest addestrato.")

    def predict(self, snapshot):
        if hasattr(snapshot, "cpu"):  # torch -> numpy
            snapshot = snapshot.cpu().numpy()

        snapshot = np.asarray(snapshot)

        # Se è un vettore di pressioni [N], usalo direttamente come feature
        # (oppure se vuoi ancora fare feature engineering, lascialo a extract_features)
        if snapshot.ndim == 1:
            x = snapshot.reshape(1, -1)
        else:
            # se già arriva (1, N) o simile
            x = snapshot.reshape(1, -1)

        return self.model.predict_proba(x)[0, 1]

