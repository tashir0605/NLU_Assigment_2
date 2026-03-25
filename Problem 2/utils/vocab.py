from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NameVocab:


    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_token: str = "<pad>"
    start_token: str = "<s>"
    end_token: str = "</s>"

    @classmethod
    def from_names(cls, names: List[str]) -> "NameVocab":
        chars = set()
        for name in names:
            chars.update(name.strip().lower())

        ordered_tokens = ["<pad>", "<s>", "</s>"] + sorted(chars)
        stoi = {token: idx for idx, token in enumerate(ordered_tokens)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def start_idx(self) -> int:
        return self.stoi[self.start_token]

    @property
    def end_idx(self) -> int:
        return self.stoi[self.end_token]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode_name(self, name: str) -> List[int]:
        """
        Converts a raw name into token ids with boundaries.

        Example for "rahul":
            [<s>, r, a, h, u, l, </s>]
        """

        name = name.strip().lower()
        ids = [self.start_idx]
        ids.extend(self.stoi[ch] for ch in name)
        ids.append(self.end_idx)
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        """Converts token ids back to a plain name string."""

        chars = []
        for token_id in ids:
            token = self.itos[token_id]
            if token in {self.start_token, self.pad_token}:
                continue
            if token == self.end_token:
                break
            chars.append(token)
        return "".join(chars)
