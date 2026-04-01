import torch
import torch.nn as nn
import torchvision.models as models
import math

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the classification head and the pooling layer to keep spatial dimensions (7x7)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # 2048 is the output channel size for ResNet50
        # We project it to match the Transformer's embed_size
        self.project = nn.Linear(2048, embed_size)
        self.train_CNN = train_CNN
        
        # Freeze or unfreeze the CNN
        for name, param in self.resnet.named_parameters():
            if "layer4" in name and train_CNN:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def forward(self, images):
        # images: (batch, 3, 224, 224)
        features = self.resnet(images) # (batch, 2048, 7, 7)
        features = features.view(features.size(0), features.size(1), -1) # (batch, 2048, 49)
        features = features.permute(0, 2, 1) # (batch, 49, 2048)
        features = self.project(features) # (batch, 49, embed_size)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        x = x + self.pe[:, :x.size(1)]
        return x

class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom Decoder Layer to expose cross-attention weights.
    Standard nn.TransformerDecoderLayer doesn't easily return cross-attention weights.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention (Memory is encoder output)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, cross_attn_weights

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=3, nhead=8, max_seq_length=50, dropout=0.1):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model=embed_size, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, features, captions, return_attention=False):
        # features: (batch, 49, embed_size) - this is the memory
        # captions: (batch, seq_len)
        
        seq_len = captions.size(1)
        embeddings = self.embed(captions) * math.sqrt(self.embed_size)
        tgt = self.pos_encoder(embeddings)
        
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        
        # Passing through decoder layers
        cross_attentions = []
        for layer in self.layers:
            tgt, attn_weights = layer(tgt, features, tgt_mask=tgt_mask)
            if return_attention:
                cross_attentions.append(attn_weights)
                
        out = self.fc_out(tgt)
        
        if return_attention:
            # Return the attention maps from the last layer
            return out, cross_attentions[-1]
            
        return out

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=3, nhead=8, train_CNN=False):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN)
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_layers, nhead)
        
    def forward(self, images, captions, return_attention=False):
        features = self.encoder(images)
        # Exclude the last token from captions for teacher forcing (the target doesn't include the input)
        # Standard autoregressive training: input is <start> word1 word2, target is word1 word2 <end>
        # So we pass captions[:, :-1] to decoder
        outputs = self.decoder(features, captions[:, :-1], return_attention)
        return outputs
