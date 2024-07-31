import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoderLayer


class CustomTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, encoded_img_dim, nhead, dim_feedforward=2048,
                 dropout=0.1, norm_first=False, batch_first=True):
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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None, tgt_is_causal: bool = False,
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


class CustomImageEncoder(nn.Module):
    def __init__(self, d_model, d_encoding, nhead, dim_feedforward, dropout,
                 num_layers, output_dim, max_length, norm=None):
        super().__init__()
        self.decoder = CustomTransformerDecoder(d_model, d_encoding, nhead, dim_feedforward,
                                                dropout, num_layers, norm)
        self.projection = nn.Linear(d_model, output_dim)
        self.max_length = max_length

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return self.projection(output)

    def generate(self, memory, start_token, max_length):
        device = next(self.parameters()).device
        batch_size = memory.size(0)

        # Initialize the input sequence with start tokens
        input_seq = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Create causal mask
            tgt_mask = self.generate_square_subsequent_mask(input_seq.size(1)).to(device)

            # Generate the next token probabilities
            output = self.forward(input_seq, memory, tgt_mask=tgt_mask)
            next_token= output[:, -1, :]

            # Append the next token to the input sequence
            input_seq = torch.cat([input_seq, next_token], dim=1)

        return input_seq

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
