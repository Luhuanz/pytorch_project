import torch
import torch.nn.functional as F

weight = torch.FloatTensor([
    [1.0, 2.0, 3.1],
    [0.1, 0.1, 0.1], 
    [0.2, 0.2, 0.2],
]).view(1, 1, 3, 3)

bias = torch.FloatTensor([0.0]).view(1)

input = torch.FloatTensor([
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [-1, 1, 1],
        [1, 0, 1],
        [1, 1, -1],
    ]
]).view(2, 1, 3, 3)

print(F.conv2d(input, weight, bias, padding=1))