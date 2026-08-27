from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        result = []
        for num in numbers:
            s = str(num)
            tokens = []
            i = 0
            while i < len(s):
                matched = False
                for j in range(len(s), i, -1):
                    candidate = s[i:j]
                    if candidate in vocab:
                        tokens.append(candidate)
                        i = j
                        matched = True
                        break
                if not matched:
                    tokens.append(s[i])
                    i += 1
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        count = 0
        i = 0
        while i < len(text):
            matched = False
            for j in range(len(text), i, -1):
                candidate = text[i:j]
                if candidate in vocab:
                    count += 1
                    i = j
                    matched = True
                    break
            if not matched:
                count += 1
                i += 1
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        num_tokens = self.count_tokens(text, vocab)
        num_words = len(text.split())
        return round(num_tokens / num_words, 4)
