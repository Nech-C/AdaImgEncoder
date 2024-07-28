import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoderLayer


class CustomTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self,
                 d_model,
                 encoded_img_dim,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 norm_first=False,
                 batch_first=True):
        super(CustomTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first, norm_first=norm_first)
        # Override the multihead_attn layer to attend to encoded image with different dim
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    kdim=encoded_img_dim, vdim=encoded_img_dim, batch_first=batch_first)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:
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
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

class CustomTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, d_model, d_encoding, nhead, dim_feedforward, dropout, num_layers, norm=None):
        decoder_layer = CustomTransformerDecoderLayer(d_model, d_encoding, nhead, dim_feedforward, dropout)
        super(CustomTransformerDecoder, self).__init__(decoder_layer, num_layers, norm)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output
