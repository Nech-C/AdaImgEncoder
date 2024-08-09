import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoderLayer
from typing import Optional

class CustomTransformerDecoderLayer(TransformerDecoderLayer):
    """Custom TransformerDecoderLayer that uses MultiheadAttention with different dim"""
    def __init__(self, d_model: int, encoded_img_dim: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 norm_first: int = False, batch_first: bool = True):
        """
        Constructor for CustomTransformerDecoderLayer
        
        Args:
            d_model (int): The number of expected features in the input
            encoded_img_dim (int): The number of expected features in the encoded image
            nhead (int): The number of heads in the multiheadattention models
            dim_feedforward (int): The dimension of the feedforward network model
            dropout (float): The dropout value
            norm_first (bool): If True, apply layer normalization before sublayer
            batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature)
        """
        super(CustomTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward,
                                                            dropout, batch_first=batch_first,
                                                            norm_first=norm_first)
        # Override the multihead_attn layer to attend to encoded image with different dim
        self.multihead_attn  = nn.MultiheadAttention(d_model, nhead, dropout, kdim=encoded_img_dim,
                                                     vdim=encoded_img_dim, batch_first=batch_first)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None, tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:
        """ Forward pass for CustomTransformerDecoderLayer

        Args:
            tgt (torch.Tensor): The sequence to the decoder (required)
            memory (torch.Tensor): The sequence from the last layer of the encoder (required)
            tgt_mask (torch.Tensor): The mask for the tgt sequence (optional)
            memory_mask (torch.Tensor): The mask for the memory sequence (optional)
            tgt_key_padding_mask (torch.Tensor): The mask for the tgt keys per batch (optional)
            memory_key_padding_mask (torch.Tensor): The mask for the memory keys per batch (optional)
            tgt_is_causal (bool): If True, the decoder should be causal (optional)
            memory_is_causal (bool): If True, the memory should be causal (optional)
            
        Returns:
            torch.Tensor: The output tensor
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask,
                                    memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask,
                                               memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
                   attn_mask: torch.Tensor, key_padding_mask: torch.Tensor,
                   is_causal: bool = False) -> torch.Tensor:
        """ Multi-head attention block
        
        Args:
            x (torch.Tensor): The input tensor
            mem (torch.Tensor): The memory tensor
            attn_mask (torch.Tensor): The mask for the attention
            key_padding_mask (torch.Tensor): The mask for the keys
            is_causal (bool): If True, the attention should be causal
        """
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

class CustomTransformerDecoder(nn.TransformerDecoder):
    """Custom TransformerDecoder that uses CustomTransformerDecoderLayer

    """
    def __init__(self, d_model: int, d_encoding: int, nhead: int, dim_feedforward: int,
                 dropout: float, num_layers: int, norm: nn.Module = None):
        """ Constructor for CustomTransformerDecoder
        
        Args:
            d_model (int): The number of expected features in the input
            d_encoding (int): The number of expected features in the encoded image
            nhead (int): The number of heads in the multiheadattention models
            dim_feedforward (int): The dimension of the feedforward network model
            dropout (float): The dropout value
            num_layers (int): The number of sub-decoder-layers in the decoder
            norm (nn.Module): The layer normalization component
        """
        decoder_layer = CustomTransformerDecoderLayer(d_model, d_encoding, nhead,
                                                      dim_feedforward, dropout)
        super(CustomTransformerDecoder, self).__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None, tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass for CustomTransformerDecoder
        
        Args:
            tgt (torch.Tensor): The sequence to the decoder (required)
            memory (torch.Tensor): The sequence from the last layer of the encoder (required)
            tgt_mask (torch.Tensor): The mask for the tgt sequence (optional)
            memory_mask (torch.Tensor): The mask for the memory sequence (optional)
            tgt_key_padding_mask (torch.Tensor): The mask for the tgt keys per batch (optional)
            memory_key_padding_mask (torch.Tensor): The mask for the memory keys per batch (optional)
            tgt_is_causal (bool): If True, the decoder should be causal (optional)
            memory_is_causal (bool): If True, the memory should be causal (optional)
            
        Returns:
            torch.Tensor: The output tensor
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

class CustomImageEncoder(nn.Module):
    """
    Custom Image Encoder that uses a transformer decoder architecture to generate
    variable-length image encodings.

    This encoder takes an encoded image (memory) as input and generates a sequence
    of vectors that represent the image content.
    """

    def __init__(self, d_model: int, d_encoding: int, nhead: int, dim_feedforward: int,
                 dropout: float, num_layers: int, max_length: int,
                 norm: Optional[nn.Module] = None):
        """
        Initialize the CustomImageEncoder.

        Args:
            d_model (int): The number of expected features in the decoder inputs.
            d_encoding (int): The number of expected features in the encoded image.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
            num_layers (int): The number of sub-decoder-layers in the decoder.
            max_length (int): The maximum length of the generated sequence.
            norm (Optional[nn.Module]): The layer normalization component (default: None).
        """
        super().__init__()
        self.decoder = CustomTransformerDecoder(d_model, d_encoding, nhead, dim_feedforward,
                                                dropout, num_layers, norm)
        self.max_length = max_length
        self.d_model = d_model

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform a forward pass through the encoder.

        Args:
            tgt (torch.Tensor): The sequence to the decoder (required).
            memory (torch.Tensor): The sequence from the last layer of the encoder (required).
            tgt_mask (Optional[torch.Tensor]): The mask for the tgt sequence (optional).
            memory_mask (Optional[torch.Tensor]): The mask for the memory sequence (optional).
            tgt_key_padding_mask (Optional[torch.Tensor]): The mask for the tgt keys per batch (optional).
            memory_key_padding_mask (Optional[torch.Tensor]): The mask for the memory keys per batch (optional).

        Returns:
            torch.Tensor: The output tensor representing the generated sequence.
        """
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate(self, memory: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Generate a sequence of vectors representing the input image.

        Args:
            memory (torch.Tensor): The encoded image tensor.
            max_length (Optional[int]): The maximum length of the generated sequence.
                                        If None, uses self.max_length (default: None).

        Returns:
            torch.Tensor: The generated sequence of vectors.
        """
        device = next(self.parameters()).device
        batch_size = memory.size(0)
        max_length = max_length or self.max_length

        # Initialize the input sequence with start tokens
        input_seq = torch.zeros(batch_size, 1, self.d_model, device=device)

        for _ in range(max_length - 1):  # -1 because we already have one token
            # Create causal mask
            tgt_mask = self.generate_square_subsequent_mask(input_seq.size(1)).to(device)

            # Generate the next token
            output = self.forward(input_seq, memory, tgt_mask=tgt_mask)
            next_token = output[:, -1, :].unsqueeze(1)

            # Append the next token to the input sequence
            input_seq = torch.cat([input_seq, next_token], dim=1)

        return input_seq

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence. The mask ensures that the
        predictions for position i depend only on the known outputs at
        positions less than i.

        Args:
            sz (int): The size of the square matrix.

        Returns:
            torch.Tensor: A square mask tensor.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask