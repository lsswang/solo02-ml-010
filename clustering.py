"""Mutual-best-link clustering subject to duplicate-free source constraints."""

import math


def mutual_best_partition(nodes, weights, duplicate_free_sources):
    sources = {node["id"]: node["source"] for node in nodes}
    if len(sources) != len(nodes):
        raise ValueError("duplicate node identity")
    edges = {}
    for (left, right), weight in weights.items():
        if left == right or left not in sources or right not in sources:
            raise ValueError("unknown or repeated edge endpoint")
        if not math.isfinite(weight):
            raise ValueError("edge weight must be finite")
        key = tuple(sorted((left, right)))
        if key in edges:
            raise ValueError("repeated unordered edge")
        edges[key] = weight
    free = set(duplicate_free_sources)
    groups = [frozenset((node,)) for node in sorted(sources)]
    while True:
        best = {}
        for i, left in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right = groups[j]
                left_sources = {sources[x] for x in left} & free
                right_sources = {sources[x] for x in right} & free
                if left_sources & right_sources:
                    continue
                edge_weight = max(edges.get(tuple(sorted((a, b))), 0.0)
                                  for a in left for b in right)
                if edge_weight <= 0:
                    continue
                for current, other in ((i, j), (j, i)):
                    option = (-edge_weight, tuple(sorted(groups[other])), other)
                    if current not in best or option < best[current]:
                        best[current] = option
        pairs = [(i, choice[2]) for i, choice in best.items()
                 if i < choice[2] and best.get(choice[2], (None, None, None))[2] == i]
        if not pairs:
            break
        consumed = {index for pair in pairs for index in pair}
        groups = ([groups[i] | groups[j] for i, j in pairs]
                  + [group for i, group in enumerate(groups) if i not in consumed])
        groups.sort(key=lambda group: tuple(sorted(group)))
    return sorted(tuple(sorted(group)) for group in groups)
