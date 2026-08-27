from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        alphabets = list(set(list(text)))
        alphabets.sort()
        stoi = {c:idx for idx, c in enumerate(alphabets)}
        itos = {stoi[key]:key for key in stoi}
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encode = [stoi[w] for w in list(text)]
        return encode

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decode = [itos[w] for w in ids]
        return "".join(decode)
