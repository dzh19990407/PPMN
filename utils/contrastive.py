import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

eps = 1e-6


def ckd_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        index: torch.Tensor,
        gamma: float = 0.1,
) -> torch.Tensor:
    # pred: [M, C]
    # target: [TBN, C], T is the gpu nums
    # index: [M, BN]
    M, C = pred.shape
    index_matrix = index.new_zeros((M, target.shape[0]))
    # [M, TBN]
    index_matrix[:, :index.shape[1]] = index
    # index_matrix: [M, TBN]

    # Extract nonzero sample from target
    target_nonzero = target.sum(dim=-1)
    target_nonzero = target_nonzero != 0
    target = target[target_nonzero]
    index_matrix = index_matrix[:, target_nonzero]

    similarity = torch.einsum('mc,nc->mn', pred, target) / gamma
    # similarity: [M, TBN]
    similarity = similarity.exp()
    neg_similarity = similarity * (index_matrix==0)
    # neg_similarity: [M, TBN]
    neg_similarity = torch.sum(neg_similarity, dim=-1, keepdim=True)
    similarity = similarity * (index_matrix!=0)
    # neg_similarity: [M, 1]
    similarity = similarity / (similarity + neg_similarity + eps)
    similarity = similarity * index_matrix + 1 - index_matrix + eps * index_matrix
    similarity = -similarity.log()
    similarity = torch.sum(similarity, dim=-1) / torch.sum(index_matrix, dim=-1)
    return similarity.mean()


class MemoryPool:
    def __init__(self, size: int = 1000):
        self._memory = None
        self._size = size
        self._rank, self._world_size = dist.get_rank(), dist.get_world_size()

    def register(self, tensor: torch.Tensor) -> int:
        tensor = tensor.contiguous()
        if self._world_size > 1:
            tensor_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
            dist.all_gather(tensor_list, tensor)
            tensor_list[0], tensor_list[self._rank] = tensor_list[self._rank], tensor_list[0]
            tensor = torch.cat(tensor_list)
        else:
            tensor = tensor.detach()
        # self._memory.insert(0, tensor)
        tensor = tensor[tensor.sum(-1) != 0]
        if self._memory is None:
            self._memory = tensor
        else:
            self._memory = torch.cat([tensor, self._memory])
        if len(self._memory) > self._size:
            # self._memory.pop(-1)
            self._memory = self._memory[:self._size]

    @property
    def memory(self) -> torch.Tensor:
        # return torch.cat(self._memory)
        return self._memory


class CKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._memory_pool = MemoryPool()

    def forward(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            index_ori: torch.Tensor,
    ):
        # preds: [B, N, C]
        # targets: [B, N, C]
        # index: [B, N]
        assert preds.shape == targets.shape, (preds.shape, targets.shape)
        index = index_ori.clone()
        index_bool = (index.clone() > 0)
        if len(index) > 1:
            for i in range(1, len(index)):
                index[i] = index[i] + index[i - 1].max()
        index = index * index_bool

        B, N, C = preds.shape
        targets = targets.reshape(B * N, C)
        index = index.reshape(-1)
        # preds: [BN, C]
        # targets: [BN, C]
        # index: [BN]

        ignore = (index == 0)
        targets = targets[~ignore]
        targets = F.normalize(targets, dim=-1)
        targets_new = targets.new_zeros((B*N, C))
        targets_new[:len(targets), :] = targets
        self._memory_pool.register(targets_new)

        preds = preds.reshape(B*N, C)
        preds = preds[~ignore]
        preds = F.normalize(preds, dim=-1)
        # preds: [M, C]

        # index: [BN]
        # index: [1, 1, 2, 2, 2, ..., N, N]
        #  positive sample matrix
        # [1, 1, 0, 0, 0, ..., 0, 0]
        # [1, 1, 0, 0, 0, ..., 0, 0]
        # [0, 0, 1, 1, 1, ..., 0, 0]
        # [0, 0, 1, 1, 1, ..., 0, 0]
        # [0, 0, 1, 1, 1, ..., 0, 0]
        # [0, 0, 0, 0, 0, .  , 0, 0]
        # [0, 0, 0, 0, 0,  . , 0, 0]
        # [0, 0, 0, 0, 0,   ., 0, 0]
        # [0, 0, 0, 0, 0, ..., 1, 1]
        # [0, 0, 0, 0, 0, ..., 1, 1]
        index = index[~ignore]
        index = index.unsqueeze(0)
        # [1, BN]
        # [1, M]
        index_matrix = index * index.T
        # [M, BN]
        ref_matrix = index * index
        ref_matrix = ref_matrix.T
        # [1, M]
        index_matrix = (index_matrix == ref_matrix)
        index_matrix = index_matrix.long()
        # index_matrix: [M, BN]
        # index_matrix_padding = index_matrix.new_zeros((index_matrix.shape[0], B*N))
        # index_matrix_padding[:, :index_matrix.shape[0]] = index_matrix

        loss = ckd_loss(preds, self._memory_pool.memory, index_matrix)

        return loss
