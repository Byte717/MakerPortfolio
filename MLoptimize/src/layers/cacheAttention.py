import torch
import torch.nn as nn

class AttentionOptimized(nn.Module):
    def __init__(self, input_dim, key_query_dim, value_dim):
        super().__init__()
        self.d_out_kq = key_query_dim

        self.W_query = nn.Parameter(torch.rand(input_dim, key_query_dim))
        self.W_key   = nn.Parameter(torch.rand(input_dim, key_query_dim))
        self.W_value = nn.Parameter(torch.rand(input_dim, value_dim))

        self.reset()

    def reset(self):
        self._Q = None
        self._K = None
        self._V = None
        self._scores = None
        self._len = 0

    def _full_recompute(self, x):
        Q = x.matmul(self.W_query)
        K = x.matmul(self.W_key)
        V = x.matmul(self.W_value)

        scores = Q.matmul(K.T)
        scale = self.d_out_kq ** 0.5
        attn = torch.softmax(scores / scale, dim=-1)
        out = attn.matmul(V)

        self._Q = Q.detach()
        self._K = K.detach()
        self._V = V.detach()
        self._scores = scores.detach()
        self._len = Q.size(0)

        return out

    def forward(self, x):
        T = x.size(0)

        if self._Q is None or T != self._len + 1:
            return self._full_recompute(x)

        Q_old, K_old, V_old = self._Q, self._K, self._V
        scores_old = self._scores
        N = self._len

        x_new = x[-1:].detach()
        q_new = x_new.matmul(self.W_query)
        k_new = x_new.matmul(self.W_key)
        v_new = x_new.matmul(self.W_value)
        Q = torch.cat([Q_old, q_new], dim=0)
        K = torch.cat([K_old, k_new], dim=0) #type: ignore
        V = torch.cat([V_old, v_new], dim=0) #type: ignore

        row_prefix = q_new.matmul(K_old.T) #type: ignore
        row_last = q_new.matmul(k_new.T)
        new_row = torch.cat([row_prefix, row_last], dim=1)

        col_prefix = Q_old.matmul(k_new.T)
        top = torch.cat([scores_old, col_prefix], dim=1)#type: ignore
        scores = torch.cat([top, new_row], dim=0)

        scale = self.d_out_kq ** 0.5
        attn = torch.softmax(scores / scale, dim=-1)
        out = attn.matmul(V)
        self._Q = Q.detach()
        self._K = K.detach()
        self._V = V.detach()
        self._scores = scores.detach()
        self._len = N + 1
        return out