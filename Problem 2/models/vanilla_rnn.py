from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class VanillaCharRNN(nn.Module):
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
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.layer_dropout = nn.Dropout(dropout_prob)

        self.w_xh = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for layer_idx in range(num_layers):
            input_dim = embedding_size if layer_idx == 0 else hidden_size
            self.w_xh.append(nn.Parameter(torch.randn(hidden_size, input_dim) * 0.05))
            self.w_hh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05))
            self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

        self.w_hy = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.05)
        self.b_y = nn.Parameter(torch.zeros(vocab_size))

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[List[torch.Tensor]] = None,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:


        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size, device=device)

        embeddings = self.embedding_dropout(self.embedding(input_ids))
        outputs_over_time = []

        for t in range(seq_len):
            layer_input = embeddings[:, t, :]

            for layer_idx in range(self.num_layers):
                prev_hidden = hidden[layer_idx]
                hidden[layer_idx] = torch.tanh(
                    layer_input @ self.w_xh[layer_idx].T
                    + prev_hidden @ self.w_hh[layer_idx].T
                    + self.b_h[layer_idx]
                )
                if layer_idx < self.num_layers - 1:
                    layer_input = self.layer_dropout(hidden[layer_idx])
                else:
                    layer_input = hidden[layer_idx]

            logits_t = layer_input @ self.w_hy.T + self.b_y
            outputs_over_time.append(logits_t.unsqueeze(1))

        logits = torch.cat(outputs_over_time, dim=1)
        return logits, hidden

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

        current_token = torch.tensor([[start_idx]], dtype=torch.long, device=device)
        hidden = self.init_hidden(batch_size=1, device=device)
        generated = []

        for _ in range(max_length):
            logits, hidden = self.forward(current_token, hidden=hidden)
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k and top_k > 0:
                top_values, top_indices = torch.topk(step_logits, k=min(top_k, step_logits.shape[-1]), dim=-1)
                filtered_logits = torch.full_like(step_logits, -1e9)
                filtered_logits.scatter_(1, top_indices, top_values)
                step_logits = filtered_logits

            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id == end_idx:
                break

            generated.append(token_id)
            current_token = next_token

        return generated
