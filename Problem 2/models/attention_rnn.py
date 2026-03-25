from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ManualAttentionRNN(nn.Module):
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

        self.wxh_enc = nn.ParameterList()
        self.whh_enc = nn.ParameterList()
        self.bh_enc = nn.ParameterList()

        self.wxh_dec = nn.ParameterList()
        self.whh_dec = nn.ParameterList()
        self.bh_dec = nn.ParameterList()

        for layer_idx in range(num_layers):
            input_dim = embedding_size if layer_idx == 0 else hidden_size

            self.wxh_enc.append(nn.Parameter(torch.randn(hidden_size, input_dim) * 0.05))
            self.whh_enc.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05))
            self.bh_enc.append(nn.Parameter(torch.zeros(hidden_size)))

            self.wxh_dec.append(nn.Parameter(torch.randn(hidden_size, input_dim) * 0.05))
            self.whh_dec.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05))
            self.bh_dec.append(nn.Parameter(torch.zeros(hidden_size)))

        self.w_att = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)

        self.w_combine_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.w_combine_c = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b_combine = nn.Parameter(torch.zeros(hidden_size))

        self.w_out = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.05)
        self.b_out = nn.Parameter(torch.zeros(vocab_size))

    def _encode(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Runs the encoder RNN over full input sequence.

        input_embeddings shape: [B, T, E]
        returns encoder states: [B, T, H]
        """

        current = self.embedding_dropout(input_embeddings)

        for layer_idx in range(self.num_layers):
            batch_size, seq_len, _ = current.shape
            h_t = torch.zeros(batch_size, self.hidden_size, device=current.device)
            encoder_states = []

            for t in range(seq_len):
                x_t = current[:, t, :]
                h_t = torch.tanh(
                    x_t @ self.wxh_enc[layer_idx].T
                    + h_t @ self.whh_enc[layer_idx].T
                    + self.bh_enc[layer_idx]
                )
                encoder_states.append(h_t.unsqueeze(1))

            current = torch.cat(encoder_states, dim=1)
            if layer_idx < self.num_layers - 1:
                current = self.layer_dropout(current)

        return current

    def _attention(
        self,
        decoder_state: torch.Tensor,
        encoder_states: torch.Tensor,
        lengths: Optional[torch.Tensor],
        step_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        transformed = decoder_state @ self.w_att
        scores = torch.bmm(encoder_states, transformed.unsqueeze(-1)).squeeze(-1)

        if lengths is not None:
            batch_size, max_len = scores.shape
            positions = torch.arange(max_len, device=scores.device).unsqueeze(0).expand(batch_size, -1)
            mask = positions >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        # Causal masking: at decoder step t, only allow attention over encoder
        # positions <= t. This keeps training behavior closer to autoregressive
        # generation and avoids relying on future positions.
        if step_index is not None:
            _, max_len = scores.shape
            future_mask = torch.arange(max_len, device=scores.device) > step_index
            scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embeddings = self.embedding_dropout(self.embedding(input_ids))
        encoder_states = self._encode(embeddings)

        batch_size, seq_len, _ = embeddings.shape
        decoder_states = [
            torch.zeros(batch_size, self.hidden_size, device=input_ids.device) for _ in range(self.num_layers)
        ]

        logits_steps = []
        attention_steps = []

        for t in range(seq_len):
            if t == 0:
                decoder_token_ids = input_ids[:, 0]
            else:
                use_teacher = (target_ids is not None) and (torch.rand(1, device=input_ids.device).item() < teacher_forcing_ratio)
                if use_teacher:
                    decoder_token_ids = input_ids[:, t]
                else:
                    decoder_token_ids = prev_pred_ids

            layer_input = self.embedding(decoder_token_ids)
            if self.dropout_prob > 0:
                layer_input = self.embedding_dropout(layer_input)

            for layer_idx in range(self.num_layers):
                prev_state = decoder_states[layer_idx]
                decoder_states[layer_idx] = torch.tanh(
                    layer_input @ self.wxh_dec[layer_idx].T
                    + prev_state @ self.whh_dec[layer_idx].T
                    + self.bh_dec[layer_idx]
                )
                if layer_idx < self.num_layers - 1:
                    layer_input = self.layer_dropout(decoder_states[layer_idx])
                else:
                    layer_input = decoder_states[layer_idx]

            decoder_state = decoder_states[-1]

            context, attn_weights = self._attention(decoder_state, encoder_states, lengths, step_index=t)

            combined = torch.tanh(
                decoder_state @ self.w_combine_h.T + context @ self.w_combine_c.T + self.b_combine
            )

            logits_t = combined @ self.w_out.T + self.b_out
            logits_steps.append(logits_t.unsqueeze(1))
            attention_steps.append(attn_weights.unsqueeze(1))
            prev_pred_ids = torch.argmax(logits_t, dim=-1)

        logits = torch.cat(logits_steps, dim=1)
        attn_map = torch.cat(attention_steps, dim=1)
        return logits, attn_map

    @torch.no_grad()
    def sample(
        self,
        start_idx: int,
        end_idx: int,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        min_length: int = 3,
        repetition_penalty: float = 1.2,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        if device is None:
            device = next(self.parameters()).device

        generated = [start_idx]

        for _ in range(max_length):
            current = torch.tensor([generated], dtype=torch.long, device=device)
            lengths = torch.tensor([current.shape[1]], dtype=torch.long, device=device)
            logits, _ = self.forward(current, lengths=lengths)
            step_logits = logits[:, -1, :].clone() / max(temperature, 1e-6)

            # Reduce immediate loops by penalizing already generated tokens.
            for token_id in set(generated[1:]):
                step_logits[0, token_id] = step_logits[0, token_id] / repetition_penalty

            # Prevent early stop so generated names are not trivially short.
            if len(generated) - 1 < min_length:
                step_logits[0, end_idx] = -1e9

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
