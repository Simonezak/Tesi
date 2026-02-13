import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):    
        super(GRUCell, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Layers                                 
        self.linear_z = nn.Linear(input_size+hidden_size, hidden_size)
        self.linear_r = nn.Linear(input_size+hidden_size, hidden_size)
        self.linear = nn.Linear(input_size+hidden_size, hidden_size)
        
        self._initialization()
        
    def _initialization(self):   
        # Qui vengono iniziallizzati pesi e bias     
        a = -np.sqrt(1/self.hidden_size)
        b = np.sqrt(1/self.hidden_size)        
        torch.nn.init.uniform_(self.linear_z.weight, a, b)
        torch.nn.init.uniform_(self.linear_z.bias, a, b)        
        torch.nn.init.uniform_(self.linear_r.weight, a, b)
        torch.nn.init.uniform_(self.linear_r.bias, a, b)        
        torch.nn.init.uniform_(self.linear.weight, a, b)
        torch.nn.init.uniform_(self.linear.bias, a, b)                

    def forward(self, input_, hidden_state):  
        
            inputs_and_prev_state = torch.cat((input_, hidden_state), -1)
            
            # z = sigma(W_z * a + U_z * h(t-1)) (3)
            # Update Gate (formula GRU standard): Decide quanto del vecchio stato tenere e quanto sostituire.
            update_gate = self.linear_z(inputs_and_prev_state).sigmoid()

            # r = sigma(W_r * a + U_r * h(t-1)) (4)
            # Reset Gate: Decide quanto “dimenticare” della memoria precedente quando si combinano informazione nuova e vecchia.
            # Se r=0 il nodo ignora completamente il vecchio stato.
            reset_gate = self.linear_r(inputs_and_prev_state).sigmoid()  

            # h_hat(t) = tanh(W * a + U*(r o h(t-1))) (5) 
            # prende l’input combina solo la parte “permessa” del vecchio stato (grazie a r)
            # genera una nuova informazione potenziale
            new_hidden_state = self.linear(torch.cat((input_, reset_gate * hidden_state), -1)).tanh()           
            
            # h(t) = (1-z) o h(t-1) + z o h_hat(t) (6)
            # Se update_gate è alto, l'output si aggiorna molto
            output = (1 - update_gate) * hidden_state + update_gate * new_hidden_state               
            
            return output   
       
class GGNNEncoder(nn.Module):
    def __init__(self, attr_size, hidden_size, propag_steps):
        super().__init__()

        self.hidden_size = hidden_size
        self.linear_i = nn.Linear(attr_size, hidden_size)
        self.gru = GRUCell(2 * hidden_size, hidden_size)

        self.propag_steps = propag_steps

    def forward(self, attr_matrix, adj_matrix):
        """
        attr_matrix: [B, N, F]
        adj_matrix:  [B, N, N]
        return:      [B, N, hidden_size]
        """

        A_in  = adj_matrix.float()
        A_out = adj_matrix.float().transpose(-2, -1)

        h = self.linear_i(attr_matrix).relu()

        for _ in range(self.propag_steps):
            a_in  = torch.bmm(A_in, h)
            a_out = torch.bmm(A_out, h)
            h = self.gru(torch.cat((a_in, a_out), dim=-1), h)

        return h

     

# ============================================================
#  RANDOM FOREST PER DETECTION DELLO STEP DI INIZIO LEAK
# ============================================================

from sklearn.ensemble import RandomForestClassifier

class RandomForestLeakOnsetDetector:
    """
    Classificatore binario per prevedere se in uno step
    è appena iniziato un LEAK.
    """

    def __init__(self, n_trees=200, max_depth=12,min_samples_split=4,min_samples_leaf=2,class_weight="balanced",random_state=42):
        
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
        x = self.extract_features(snapshot).reshape(1, -1)
        return self.model.predict_proba(x)[0, 1]


class GGNN_LSTM_LeakDetector(nn.Module):
    def __init__(
        self,
        ggnn: GGNNEncoder,
        hidden_size,
        lstm_hidden=64,
        lstm_layers=1
    ):
        super().__init__()

        self.ggnn = ggnn

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # output per nodo
        self.readout = nn.Linear(lstm_hidden, 1)

    def forward(self, attr_seq, adj):
        """
        attr_seq: [T, B, N, F]
        adj:      [B, N, N]
        """

        gnn_embeddings = []

        for t in range(attr_seq.size(0)):
            h_t = self.ggnn(attr_seq[t], adj)     # [B, N, H]
            gnn_embeddings.append(h_t)

        # [B, T, N, H]
        H = torch.stack(gnn_embeddings, dim=1)

        # LSTM su ogni nodo indipendentemente
        B, T, N, Hdim = H.shape
        H = H.view(B * N, T, Hdim)

        lstm_out, _ = self.lstm(H)
        z_T = lstm_out[:, -1, :]                  # [B*N, lstm_hidden]

        u = self.readout(z_T)                     # [B*N, 1]
        u = u.view(B, N)

        return u
