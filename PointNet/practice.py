import copy

import torch

from chamferdist import ChamferDistance



# Initialize Chamfer distance module
chamferDist = ChamferDistance()

ex1 = torch.randn(32, 2048, 3)
ex2 = copy.deepcopy(ex1)
ex2[0][0] = -0.05

ex_dist = chamferDist(ex1, ex2)

print(f'Chamfer Distance : {ex_dist}')

print(torch.__version__)