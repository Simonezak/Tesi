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
       
class GGNNModel(nn.Module):
    
    def __init__(self, attr_size, hidden_size, propag_steps):    
        super(GGNNModel, self).__init__()        
        
        self.attr_size = attr_size
        self.hidden_size = hidden_size
        self.propag_steps = propag_steps            
        
        # Input: grandezza dell'attributo, output dell'input uno stato di size hidden size
        self.linear_i = nn.Linear(attr_size,hidden_size)

        # 2*hidden size perche riceve in input a_in || a_out = due messaggi concatenati, ma in 
        # output deve restituire in output un output di grandezza hidden_size
        self.gru = GRUCell(2*hidden_size, hidden_size)  

        # output: un solo numero per nodo
        self.linear_o = nn.Linear(hidden_size, 1)
        self._initialization()

    # Funzione che serve ad iniziallizzare i pesi        
    def _initialization(self): 
        # Inizializza i pesi di linear_i per reti ReLU in modo ottimale
        torch.nn.init.kaiming_normal_(self.linear_i.weight)
        torch.nn.init.constant_(self.linear_i.bias, 0)
        # Inizializza la matrice dei pesi finali con Xavier
        torch.nn.init.xavier_normal_(self.linear_o.weight)
        torch.nn.init.constant_(self.linear_o.bias, 0)          
    
    def forward(self, attr_matrix, adj_matrix):

        mask = (attr_matrix[:,:,0] != 0)*1

        A_in  = adj_matrix.float() 
        A_out = adj_matrix.float().transpose(-2, -1)

        if len(A_in.shape) < 3:
            A_in  = A_in.unsqueeze(0)
            A_out = A_out.unsqueeze(0)
        if len(attr_matrix.shape) < 3:
            attr_matrix = attr_matrix.unsqueeze(0)

        hidden_state = self.linear_i(attr_matrix).relu()

        for _ in range(self.propag_steps):
            a_in  = torch.bmm(A_in,  hidden_state)
            a_out = torch.bmm(A_out, hidden_state)
            hidden_state = self.gru(torch.cat((a_in, a_out), dim=-1),
                                    hidden_state)

        # ⭐ output continuo
        anomaly = self.linear_o(hidden_state).squeeze(-1)

        return anomaly

     

# ============================================================
#  RANDOM FOREST PER DETECTION DELLO STEP DI INIZIO LEAK
# ============================================================

from sklearn.ensemble import RandomForestClassifier

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


class GGNN_LSTM(nn.Module):
    def __init__(self, attr_size, hidden_size, propag_steps, lstm_hidden=64, lstm_layers=1):
        super().__init__()

        # GNN per estrazione spaziale
        self.ggnn = GGNNModel(attr_size, hidden_size, propag_steps)

        # LSTM per modellare la dinamica temporale
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # layer finale per predire u[k]
        self.linear_out = nn.Linear(lstm_hidden, 1)

    def forward(self, attr_seq, adj_seq):
        """
        attr_seq: lista di tensors [1, N, attr_dim]
        adj_seq:  lista di tensors [1, N, N]
        """

        gnn_outputs = []

        # 1) Applica GGNN per ogni timestep
        for attr, adj in zip(attr_seq, adj_seq):
            h_t = self.ggnn(attr, adj)             # shape [1, N]
            h_t = h_t.squeeze(0)                   # → [N]
            gnn_outputs.append(h_t)

        # 2) Stack temporale: [T, N]
        H = torch.stack(gnn_outputs, dim=0)        # [T, N]

        # 3) LSTM richiede batch dimension → [1, T, N]
        H = H.unsqueeze(0)

        lstm_out, _ = self.lstm(H)                 # output: [1, T, lstm_hidden]

        # 4) Prendi l’ultimo timestep
        z_T = lstm_out[:, -1, :]                   # [1, lstm_hidden]

        # 5) Predizione finale del vettore nodale u[k]
        # Lo estendiamo a tutti i nodi
        u_pred = self.linear_out(z_T)              # [1, 1]
        return u_pred
