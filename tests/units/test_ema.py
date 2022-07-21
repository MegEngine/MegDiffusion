import pytest

import megengine.module as M

from megengine import Parameter
from megdiffusion.model.ema import ema

class SourceModel(M.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([1, 2, 3, 4])
        self.bias = Parameter([0, 0, 1, 1])

    def forward(self, x):
        return x * self.weight + self.bias

class TargetModel(M.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([2, 3, 4, 5])
        self.bias = Parameter([1, 1, 0, 0])

    def forward(self, x):
        return x * self.weight + self.bias

def test_ema():
    source = SourceModel()
    target = TargetModel()
    ema(source, target, 0.8)
    assert target.weight.tolist() == pytest.approx([1.8, 2.8, 3.8, 4.8])
    assert target.bias.tolist() == pytest.approx([0.8, 0.8, 0.2, 0.2])
    