# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops",
#     "einx",
#     "fire",
#     "numpy",
#     "rotary-embedding-torch",
#     "torch",
#     "tqdm",
# ]
# ///

import math
import gzip
import random
import tqdm
import numpy as np

import fire
import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from deep_cross_attention import (
    GPT,
    DCAGPT
)

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1).to(t.device)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]
        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# data classes

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# main training loop

def main(
    cpu = False,
    num_batches = int(1e5),
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    prime_length = 128,
    generate_every = 500,
    generate_length = 512,
    seq_len = 512,
    use_dca = True
):
    accelerator = Accelerator(cpu=cpu)
    device = accelerator.device

    # model

    if use_dca:
        model = DCAGPT(
            num_tokens = 256,
            dim = 512,
            depth = 6,
            past_layers_k = 2  # the `k` value in their paper, which refers to how many layers in the past it looks, a la Denseformer
        )
    else:
        model = GPT(
            num_tokens = 256,
            dim = 512,
            depth = 6
        )

    # prepare enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(1, num_batches + 1), mininterval = 10.0, desc = "training", disable = not accelerator.is_local_main_process):
        model.train()

        for _ in range(grad_accum_every):
            data = next(train_loader_iter)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)

        accelerator.print(f"training loss: {loss.item():.3f}")

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader_iter)

                loss = model(valid_data, return_loss = True)
                accelerator.print(f"validation loss: {loss.item():.3f}")

        if i % generate_every == 0 and accelerator.is_main_process:
            model.eval()

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(device)

            prime = decode_tokens(inp)
            print(f"\n{prime}\n")

            prompt = inp[None, ...]

            sampled = base_decoding(model, prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            print(f"\n{base_decode_output}\n")

if __name__ == '__main__':
    fire.Fire(main)
