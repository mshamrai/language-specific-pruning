import torch
from torch import nn
import math
import torch.nn.functional as F
import copy

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True


class SparseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, sparse_dense_tensor: torch.Tensor, bias: nn.Parameter=None) -> None:
        super().__init__()
        
        sparse_data = to_sparse_semi_structured(sparse_dense_tensor.clone())
        self.weight = nn.Parameter(sparse_data)

        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.in_features = self.weight.data.shape[1]
        self.out_features = self.weight.data.shape[0]
        if bias:
            self.bias = copy.deepcopy(bias)
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.in_features
        if len(input.shape) == 3:
            batch_size = input.shape[0]
            input = input.reshape(-1, self.in_features)
            output = F.linear(input, self.weight, self.bias)
            output = output.reshape(batch_size, -1, self.out_features)
        else:
            output = F.linear(input, self.weight, self.bias)
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
