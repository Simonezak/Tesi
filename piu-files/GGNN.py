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
        
        '''        
        attr_matrix of shape (batch, graph_size, attributes dimension)
        adj_matrix of shape (batch, graph_size, graph_size)   
        
            > Only 0 (nonexistent) or 1 (existent) edge types
        
        '''

        # maschera nodi non validi (es. con pressione = 0)
        mask = (attr_matrix[:,:,0] != 0)*1
        
        # matrice di adiacenza entrante e uscente
        A_in = adj_matrix.float() 
        A_out = torch.transpose(A_in,-2,-1) 
        
        if len(A_in.shape) < 3:
            A_in = torch.unsqueeze(A_in,0)  
            A_out = torch.unsqueeze(A_out,0)  
        if len(attr_matrix.shape) < 3:
            attr_matrix = torch.unsqueeze(attr_matrix,0)
               
        #print(np.shape(attr_matrix))

        # inizia l'hidden state attraverso le pressioni in input
        hidden_state = self.linear_i(attr_matrix.float()).relu()
                
        #print(np.shape(self.linear_i(attr_matrix.float()).relu()))
        for step in range(self.propag_steps):            
            # a_v = A_v[h_1 ...  h_|V|]
            
            #print(np.shape(A_in))
            #print(np.shape(A_out))
            #exit()

            # Formula i messaggi (message passing a vicini entranti e vicini uscenti)
            a_in = torch.bmm(A_in, hidden_state)
            a_out = torch.bmm(A_out, hidden_state)

            # Update dello stato GRU-like
            hidden_state = self.gru(torch.cat((a_in, a_out), -1), hidden_state)
                    
        # Crea l'output e fa la soft
        output = self.linear_o(hidden_state).squeeze(-1)  
        output = output + (mask + 1e-45).log() # Mask output
        output = output.log_softmax(1) 

        return output       
     

# ============================================================
#  RANDOM FOREST PER DETECTION DELLO STEP DI INIZIO LEAK
# ============================================================

from sklearn.ensemble import RandomForestClassifier

class RandomForestLeakOnsetDetector:
    """
    Classificatore binario per prevedere se in uno step
    è appena iniziato un LEAK.
    """

    def __init__(self, n_trees=300, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            class_weight="balanced"
        )

    @staticmethod
    def extract_features(snapshot):
        """
        snapshot = PyG Data di build_pyg_from_wntr
        Feature ≡ pressioni nodi + flowrates archi
        """
        pressures = snapshot.x[:, 2].cpu().numpy()        # pressure
        flows     = snapshot.edge_attr[:, 2].cpu().numpy()  # flowrate
        vector = np.concatenate([pressures, flows])
        return vector

    def fit(self, snapshots):
        X, Y = [], []

        for ep in snapshots:
            ep_steps = ep["steps"]
            leak_start = ep["leak_start"]   # step in cui parte il leak

            for step_idx, data in enumerate(ep_steps):
                x = self.extract_features(data)

                # LABEL:
                # 1 SOLO nello step in cui parte il leak
                label = 1 if step_idx == leak_start else 0

                X.append(x)
                Y.append(label)

        X = np.array(X)
        Y = np.array(Y)

        print("➡ Training RandomForest per leak onset...")
        self.model.fit(X, Y)
        print("✔ RandomForest addestrato.")

    def predict(self, snapshot):
        x = self.extract_features(snapshot).reshape(1, -1)
        return self.model.predict_proba(x)[0, 1]


