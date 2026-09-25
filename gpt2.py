from dataclasses import dataclass
import os
import math
import time
import inspect
import numpy as np
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

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
        # making sure that the nr of neurons in passed layer can be split evenly to all heads
        assert config.n_embd % config.n_head == 0
        # creating space for Key, Query and Value for every neuron
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # creating space for aggregating the results
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch Time Channels
        # input is projected onto c_attn
        qkv = self.c_attn(x)
        # qkv gets split into separate objects for q, k and v
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshapes the tensors for k q and v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        # reassembles all attention heads back into a single representation per each token
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # projecting the matrix onto the final result
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        # Here you create 4 times more neurons so the model can 'think' for a while about the connections it made 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU algorithm is an activation function similar to RELU, it is different because it doesn't always turn off the negative values
        self.gelu = nn.GELU(approximate= 'tanh')
        # Here you go back to original number of neurons
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        

    def forward(self,x):
        # Here the model goes through all feed-forward layers
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# One block of learning as per "Attention is all you need" paper 
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        # this layer is supposed to normalize the values of token along side the trait axis so the attention block works properly
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # attention block
        self.attn = CausalSelfAttention(config)
        # another normalization layer
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Feed-forward network (MLP)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        # config defined at the beginning
        self.config = config

        # Here are defined all the final traits that each token will have
        self.transformer = nn.ModuleDict(dict(
            # maps tokens into vector representations
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # map tokens into positional vectors
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # hidden layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # last layer, normalization 
            ln_f = nn.LayerNorm(config.n_embd),

        ))

        # layer that projects final hidden states from n_embd size to vocab_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # enforcing weight tying (sharing weights between wte and lm_head)
        self.transformer.wte.weight = self.lm_head.weight

        # custom weight init
        self.apply(self._init_weights)

    # custom weight initialization, random weights with standard deviation = 0.02.
    def _init_weights(self,module):
        std = 0.02
        # as for every step x = x + layer(x) the values would grow too fast, it is slowed down here
        if hasattr(module,'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        # for embedded weights the std is always 0.02 and for linear it can be different 
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean = 0.0, std = 0.02)


    # main forward function 
    def forward(self, idx, targets=None):
        # batch, time
        B, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # array [0,1,...,T-1] for every word's position      
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # passing pos through the embedding table [pos x traits]
        pos_emb = self.transformer.wpe(pos)
        # passing idx through the embedding table [batch x time x trait]
        tok_emb = self.transformer.wte(idx)
        # adding both of them together so now each token in batch can correlate with different particular tokens on different positions
        x = tok_emb + pos_emb
        # passing the relations through the hidden layers
        for block in self.transformer.h:
            x = block(x)
        # passing the relations through normalization layer
        x = self.transformer.ln_f(x)
        # here the model compares all the traits to all the tokens 
        logits = self.lm_head(x)
        # calculating loss 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # downloading pretrained model from Hugging face
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


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    if master_process:
        print(f"using device: {device}")

# ensuring consistency among different attempts
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


# defining batch size and time
total_batch_size = 524288
B = 16
T = 1024
# in case of parallel calculations, calculate the correct number of steps
assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size is not divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# loading tokens from source
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt.astype(np.int64), dtype=torch.long)
    return ptt

# Self explanatory
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# parameters for changing learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

# returng learning rate based on the # of steps
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
torch.set_float32_matmul_precision('high')

# creating the GPT model and compiling it if possible using cuda
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if device == 'cuda':
    model = torch.compile(model)
# setting the distributet calculations if possible
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device_type=device)
# Main loop of learning 
for i in range(50):
    t0 = time.time()
    # zeroing the gradient
    optimizer.zero_grad()
    loss_accum = torch.zeros(1, device=device)
    # perfonrimg gradient accumulation
    for micro_step in range(grad_accum_steps):
        # creating a new batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # forward step
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        lass_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # backward step
        loss.backward()
    # averaging the loss if using ddp
    if ddp:
        dist.all_reduce(loss_accum,op =dist.ReduceOp.AVG)

    # gradient clipping to prevent explosion of results
    norm = torch.nn.utils.clip_grad_norm(model.parameters(),1.0)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    dt = t1 - t0 # time difference in seconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {i:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
