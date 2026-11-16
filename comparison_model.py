"""Supervised independent-Bernoulli comparison model with optional missing values."""

import itertools
import math


class IndependentComparisonModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.dimensions = len(probabilities[0])

    @classmethod
    def fit(cls, rows, dimensions, alpha=2.0):
        if dimensions < 1 or alpha < 0 or not math.isfinite(alpha):
            raise ValueError("invalid dimensions or smoothing")
        ones = [[alpha] * dimensions for _ in range(2)]
        totals = [[2 * alpha] * dimensions for _ in range(2)]
        for row in rows:
            label, values, weight = row["label"], row["values"], row["weight"]
            if label not in (0, 1) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("invalid label or weight")
            cls._validate_values(values, dimensions)
            for axis, value in enumerate(values):
                if value is not None:
                    ones[label][axis] += weight * value
                    totals[label][axis] += weight
        if any(total <= 0 for label in totals for total in label):
            raise ValueError("a feature has no observations and no smoothing")
        return cls([[ones[c][j] / totals[c][j] for j in range(dimensions)]
                    for c in range(2)])

    @staticmethod
    def _validate_values(values, dimensions):
        if len(values) != dimensions or any(v not in (0, 1, None) for v in values):
            raise ValueError("comparison vector must contain binary values or None")

    def likelihood(self, values, label):
        self._validate_values(values, self.dimensions)
        if label not in (0, 1):
            raise ValueError("unknown class")
        value = 1.0
        for observed, probability in zip(values, self.probabilities[label]):
            if observed is not None:
                value *= probability if observed else 1.0 - probability
        return value

    def score(self, values, prior):
        if not 0 < prior < 1:
            raise ValueError("prior must be strictly between zero and one")
        m, u = self.likelihood(values, 1), self.likelihood(values, 0)
        denominator = prior * m + (1 - prior) * u
        if denominator == 0:
            raise ValueError("observed comparison has zero model probability")
        return {"m": m, "u": u, "probability": prior * m / denominator}

    def joint_table(self, label):
        return [self.likelihood(bits, label)
                for bits in itertools.product((0, 1), repeat=self.dimensions)]
