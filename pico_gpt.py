# pico_gpt.py (FINAL & CORRECTED)
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """
    One head of self-attention.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_s'], config['block_s'])))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config['n_head'])])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    A simple feed-forward neural network layer.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: communication followed by computation.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    The main GPT language model.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_s'], config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['block_s'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_l'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_s'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens based on the input context.

        Args:
            idx (torch.Tensor): The input context.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0.
            top_k (int, optional): The number of top-k candidates to consider. Defaults to None.

        Returns:
            torch.Tensor: The generated sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_s']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx