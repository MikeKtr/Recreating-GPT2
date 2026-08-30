from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F

# Parameters 
@dataclass 
class GPTConfig:
    block_size : int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # making sure that the nr of naurons in passed layer can be split evenly to all heads
        assert config.n_embd % config.n_head == 0
        # creating space for Key,Querry and Value for every nauron
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # creating space for agregating the results
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch Time Channels
        # input is projected ontu c_attn
        qkv = self.c_attn(x)
        # qkv gets splitted onto seperate objects for q,k and v
        q, k, v = qkv.split(self.n_embd, dim=2)
        # changing the way matlab is seeing k q and v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # softmax 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        # reducing the matrix into the lower triangle matrix 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # projecting the matrix onto the final result
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        # Here you create 4 times more neurons so the model can 'think' for a while about the connections it made 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU algorithim is a activation function similar to RELU, it is diferent because it doesn't alway turns off the negative values
        self.gelu = nn.GELU(approximate= 'tanh')
        # Here you go back to original number of naurons
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        

    def forward(self,x):
        # Here the model goes thrue all defined layers
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# One block of learning as per "Attention is all you need" paper 
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        # this layer is supposed to normalize the values of token along side the trait axis so the atteintion block works properly
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # attention block
        self.attn = CausalSelfAttention(config)
        # another normalization layer
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # GELU function
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        # config defined at the begining
        self.config = config

        # Here are defined all the final traits that each token will have
        self.transformer = nn.ModuleDict(dict(
            # Relations with other tokens
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Relations with other positions
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # hidden layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # last layer, normalization 
            ln_f = nn.LayerNorm(config.n_embd),

        ))

        # layer that project attention block from every token from n_embd size to vocab_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # enforcking shared buffers
        self.transformer.wte.weight = self.lm_head.weight

        # custom weight init
        self.apply(self._init_weights)

    # custom weight inicialization, ranom weights with standard diviation = std.
    def _init_weights(self,module):
        std = 0.02
        if hasattr(module,'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean = 0.0, std = 0.02)


    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# Using proper device based on available specs
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print("using : ",device)

import tiktoken

# Self explanatory
class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        with open('ims15_6_clean.txt','r' ) as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)}")

        self.current_position = 0

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B * T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x,y 


train_loader = DataLoaderLite(4,32)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)

# 
for i in range(50):
    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits,loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step: {i}, loss: {loss.item()}")

