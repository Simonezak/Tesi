import torch
import torch.nn as nn
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
        a = -np.sqrt(1 / self.hidden_size)
        b = np.sqrt(1 / self.hidden_size)

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
        self.linear_i = nn.Linear(attr_size, hidden_size)

        # 2*hidden size perche riceve in input a_in || a_out = due messaggi concatenati, ma in 
        # output deve restituire in output un output di grandezza hidden_size
        self.gru = GRUCell(2 * hidden_size, hidden_size)  

        # output: un solo numero per nodo
        self.linear_o = nn.Linear(hidden_size, 1)

        self._initialization()

    def _initialization(self): 
        # Inizializza i pesi di linear_i per reti ReLU in modo ottimale
        torch.nn.init.kaiming_normal_(self.linear_i.weight)
        torch.nn.init.constant_(self.linear_i.bias, 0)
        # Inizializza la matrice dei pesi finali con Xavier
        torch.nn.init.xavier_normal_(self.linear_o.weight)
        torch.nn.init.constant_(self.linear_o.bias, 0)          
    
    def forward(self, attr_matrix, adj_matrix):
        """
        attr_matrix: [B, N, 1]  (pressione istantanea)
        adj_matrix : [B, N, N]  o [N, N]
        """
        if attr_matrix.dim() == 2:  # [N,F]
            attr_matrix = attr_matrix.unsqueeze(0)  # [1,N,F]

        if adj_matrix.dim() == 2:  # [N,N]
            adj_matrix = adj_matrix.unsqueeze(0)  # [1,N,N]

        A_in  = adj_matrix.float()
        A_out = adj_matrix.transpose(-2, -1).float()

        # Stato iniziale dai valori di pressione
        hidden_state = self.linear_i(attr_matrix).relu()   # [B, N, H]

        # Message passing
        for _ in range(self.propag_steps):
            a_in  = torch.bmm(A_in,  hidden_state)
            a_out = torch.bmm(A_out, hidden_state)

            hidden_state = self.gru(
                torch.cat((a_in, a_out), dim=-1),
                hidden_state
            )

        # Score finale per nodo
        anomaly = self.linear_o(hidden_state).squeeze(-1)

        if anomaly.size(0) == 1:
            return anomaly.squeeze(0)

        return anomaly
