from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            count = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                count[pair] = count.get(pair, 0) + 1

            if not count:
                break  # nothing left to merge

            max_count_val = max(count.values())
            # among pairs with max count, pick lexicographically smallest
            best_pair = min(p for p, c in count.items() if c == max_count_val)

            merges.append([best_pair[0], best_pair[1]])
            tokens = self.merge_pair(tokens, best_pair)

        return merges

    def merge_pair(self, tokens, pair):
        a, b = pair
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged   
