"""
This module defines the AdaEncoder abstract base class, which serves as a template
for implementing adaptive image encoders that generate variable-length vector sequences.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn

class AdaEncoder(nn.Module, ABC):
    """
    An abstract base class for adaptive image encoders.

    AdaEncoder defines the interface for models that can encode images into
    variable-length sequences of vectors. This class combines PyTorch's nn.Module
    functionality with abstract methods to ensure consistent implementation across
    different encoder architectures.

    Attributes:
        encoding_dim (int): The dimensionality of the output encoding vectors.
        max_sequence_length (int): The maximum allowed length for generated sequences.
    """

    def __init__(self, encoding_dim: int):
        """
        Initialize the AdaEncoder.

        Args:
            encoding_dim (int): The dimensionality of the output encoding vectors.
            max_sequence_length (int): The maximum allowed length for generated sequences.
        """
        super().__init__()
        self.encoding_dim = encoding_dim

    @abstractmethod
    def generate(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate a sequence of vectors from an input image.
        
        Args:
            image (torch.Tensor): The input image tensor.
        
        Returns:
            torch.Tensor: A tensor representing the encoded image.
        """