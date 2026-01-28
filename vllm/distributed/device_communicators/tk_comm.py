import sys

# hack to get the TK path
sys.path.insert(0, "/root/ThunderKittens/kernels/parallel")
sys.path.insert(0, "/root/ThunderKittens/kernels/parallel/all_reduce")

import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.logger import init_logger
logger = init_logger(__name__)


class TKCommunicator:
    """Single-node ThunderKittens all-reduce adapter (copy-in/copy-out)."""

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size_override: int,
        sequence_lengths,
        hidden_size: int,
    ):
        # try 64mb
        # max_size_override = 4096 * 8192 * 2
        self.disabled = True

        try:
            from _C import TKParallelTensor, tk_all_reduce
        except Exception as e:
            logger.warning("TKCommunicator: failed to import _C: %s", e)
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        torch.cuda.set_device(device)

        self.group = group
        self.device = device

        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

        # single-node assumptions
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size)))
        assert self.rank == local_rank, (self.rank, local_rank)
        assert self.world_size == local_world_size, (self.world_size, local_world_size)

        self.dtype = torch.bfloat16

        self.max_size = int(max_size_override)
        self.max_elems = self.max_size // self.dtype.itemsize

        self.TKParallelTensor = TKParallelTensor
        self.tk_all_reduce = tk_all_reduce
        self.cache = {}
        try:
            # add to the cache
            for seq_len in sequence_lengths:
                n_elements = seq_len * hidden_size
                buffer = self.TKParallelTensor(
                    (n_elements, 1),
                    dtype=self.dtype,
                    local_rank=self.rank,
                    local_world_size=self.world_size,
                    multicast=True,
                )
                self.cache[n_elements] = buffer


            # self.buffer = self.TKParallelTensor(
            #     (self.max_elems, 1),
            #     dtype=self.dtype,
            #     local_rank=self.rank,
            #     local_world_size=self.world_size,
            #     multicast=True,
            # )
            self.barrier = self.TKParallelTensor(
                (1, 1),
                dtype=torch.int,
                local_rank=self.rank,
                local_world_size=self.world_size,
                multicast=True,
            )
            self.barrier.data_.zero_()
            dist.barrier(self.group)
        except Exception as e:
            logger.warning("TKCommunicator: initialization failed: %s", e)
            return

        self.disabled = False
        logger.info("Rank %s: TKCommunicator initialized", self.rank)

    def should_use_tk(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        if not inp.is_cuda:
            return False
        if not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes % 4 != 0:
            return False
        return nbytes < self.max_size

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor | None = None, copy: bool = True):
        if not self.should_use_tk(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)

        flat_in = inp.view(-1)
        n = flat_in.numel()

        buffer = self.cache[n]
        flat_buf = buffer.data_.view(-1)
        if copy: flat_buf[:n].copy_(flat_in)

        self.tk_all_reduce(buffer, self.barrier)

        if copy: out.view(-1).copy_(flat_buf[:n])
        return out
