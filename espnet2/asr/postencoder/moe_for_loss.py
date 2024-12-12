#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
import logging


class MoE_for_Loss(AbsPostEncoder):
    """Linear Projection Preencoder."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        self.output_dim = output_size
        self.linear_man = torch.nn.Linear(input_size, input_size)
        self.linear_eng = torch.nn.Linear(input_size, input_size)
        self.linear_mix = torch.nn.Linear(input_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, man_encoder_out: torch.Tensor, eng_encoder_out: torch.Tensor, input_lengths: torch.Tensor):
        """Forward."""
        output_man = self.linear_man(self.dropout(man_encoder_out))
        # logging.info(f"output_man:{output_man.shape}")
        output_eng = self.linear_eng(self.dropout(eng_encoder_out))
        # logging.info(f"output_eng:{output_eng.shape}")
        input_mix = torch.cat((torch.unsqueeze(output_man,0), torch.unsqueeze(output_eng,0)), dim=0)
        # logging.info(f"input_mix:{input_mix.shape}")
        output_mix = self.linear_mix(self.dropout(input_mix))
        output = torch.nn.functional.softmax(output_mix,dim=0)
        return output[0], output[1]  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim