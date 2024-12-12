#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
import logging


class MoE_Conv(AbsPostEncoder):
    """Linear Projection Preencoder."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        self.output_dim = output_size
        self.depthwise_conv_fusion = torch.nn.Conv2d(
            input_size,
            input_size,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            groups=input_size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(input_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, man_encoder_out: torch.Tensor, eng_encoder_out: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""

        input_tensor = torch.cat((torch.unsqueeze(man_encoder_out,0), torch.unsqueeze(eng_encoder_out,0)), dim=0)
        # logging.info(f"input_tensor-{input_tensor.shape}")
        tmp = input_tensor.permute(0,3,1,2).contiguous()
        # logging.info(f"tmp1-{tmp.shape}")
        tmp = self.depthwise_conv_fusion(tmp)
        # logging.info(f"tmp2-{tmp.shape}")
        tmp = tmp.permute(0,2,3,1).contiguous()
        # logging.info(f"tmp3-{tmp.shape}")
        output = self.dropout(self.merge_proj(input_tensor + tmp))
        output = torch.nn.functional.softmax(output,dim=0)
        # logging.info(f"output-{output.shape}")
        # logging.info(f"output[0]-{output[0]}")
        # logging.info(f"output[1]-{output[1]}")
        return output[0], output[1]  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
