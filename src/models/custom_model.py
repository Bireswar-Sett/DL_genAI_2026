import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2)*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,x):
        return x+self.pe[:,:x.size(1)].to(x.device)

class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,d_model=256,n_heads=8,num_layers=4,dropout=0.1):
        super().__init__()

        self.embedding=nn.Embedding(vocab_size,d_model,padding_idx=0)

        self.position=PositionalEncoding(d_model)
        self.dropout=nn.Dropout(dropout)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self,input_ids,mask):
        x=self.embedding(input_ids)
        x=self.position(x)
        x=self.dropout(x)
        padding_mask=~mask.bool()
        x=self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        mask=mask.unsqueeze(-1)
        x=x*mask
        x=x.sum(dim=1)/mask.sum(dim=1).clamp(min=1)
        return self.dropout(x)

class MCQBiEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.encoder=TransformerEncoder(
            vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )


    def forward(self,q_ids,q_mask,opt_ids,opt_mask):
        q=self.encoder(
            q_ids,
            q_mask
        )
        options=[]
        for i in range(5):
            o=self.encoder(
                opt_ids[:,i],
                opt_mask[:,i]
            )
            options.append(o)

        options=torch.stack(
            options,
            dim=1
        )
        q=q.unsqueeze(1)
        scores=F.cosine_similarity(
            q,
            options,
            dim=-1
        )
        return scores
