from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ManualLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.w_i = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.u_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.w_f = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.u_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.w_o = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.u_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.w_g = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.u_g = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i_t = torch.sigmoid(x_t @ self.w_i.T + h_prev @ self.u_i.T + self.b_i)
        f_t = torch.sigmoid(x_t @ self.w_f.T + h_prev @ self.u_f.T + self.b_f)
        o_t = torch.sigmoid(x_t @ self.w_o.T + h_prev @ self.u_o.T + self.b_o)
        g_t = torch.tanh(x_t @ self.w_g.T + h_prev @ self.u_g.T + self.b_g)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ManualBLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.layer_dropout = nn.Dropout(dropout_prob)

        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()

        for layer_idx in range(num_layers):
            input_size = embedding_size if layer_idx == 0 else 2 * hidden_size
            self.forward_cells.append(ManualLSTMCell(input_size=input_size, hidden_size=hidden_size))
            self.backward_cells.append(ManualLSTMCell(input_size=input_size, hidden_size=hidden_size))

        self.w_out = nn.Parameter(torch.randn(vocab_size, 2 * hidden_size) * 0.05)
        self.b_out = nn.Parameter(torch.zeros(vocab_size))

    def _run_direction(
        self,
        cell: ManualLSTMCell,
        sequence: torch.Tensor,
        reverse: bool,
    ) -> List[torch.Tensor]:


        batch_size, seq_len, _ = sequence.shape
        device = sequence.device

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        states = [None] * seq_len

        for t in time_indices:
            x_t = sequence[:, t, :]
            h_t, c_t = cell(x_t, h_t, c_t)
            states[t] = h_t

        return states

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        current = self.embedding_dropout(self.embedding(input_ids))

        for layer_idx in range(self.num_layers):
            forward_states = self._run_direction(self.forward_cells[layer_idx], current, reverse=False)
            backward_states = self._run_direction(self.backward_cells[layer_idx], current, reverse=True)

            merged = []
            for t in range(current.shape[1]):
                merged_t = torch.cat([forward_states[t], backward_states[t]], dim=-1)
                merged.append(merged_t.unsqueeze(1))
            current = torch.cat(merged, dim=1)
            if layer_idx < self.num_layers - 1:
                current = self.layer_dropout(current)

        logits = current @ self.w_out.T + self.b_out
        return logits, current

    @torch.no_grad()
    def sample(
        self,
        start_idx: int,
        end_idx: int,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
    ) -> List[int]:

        if device is None:
            device = next(self.parameters()).device

        generated = [start_idx]

        for _ in range(max_length):
            prefix = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = self.forward(prefix)
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                top_values, top_indices = torch.topk(step_logits, k=min(top_k, step_logits.shape[-1]), dim=-1)
                filtered_logits = torch.full_like(step_logits, -1e9)
                filtered_logits.scatter_(1, top_indices, top_values)
                step_logits = filtered_logits
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == end_idx:
                break

            generated.append(next_token)

        return generated[1:]
