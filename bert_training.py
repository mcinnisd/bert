#!/usr/bin/env python
# coding: utf-8

# # EECS 595 HW3: Parts 1-3 Building and Pre-Training BERT from Scratch

import os
import math
import numpy as np
import random
import logging

# Bring in PyTorch
import torch
import torch.nn as nn
# optimizer transfo

# Most of the examples have typing on the signatures for readability
from typing import Optional, Callable, List, Tuple
from copy import deepcopy
# For data loading
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
import json
import glob
import gzip
import bz2
import wandb

import matplotlib.pyplot as plt

# For progress and timing
from tqdm.auto import tqdm, trange
import time


# # Part 1: Tokenization

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase=True)


# # Part 2: Building a Transformer Encoder and BERT

class BertPositionalEmbedding(nn.Module):
    def __init__(self, vocab_dim: int, 
                 hidden_dim: int = 768, 
                 padding_idx: int = 0, 
                 max_seq_length: int = 512):
        
        super().__init__()

        '''
        Initialize the Embedding Layers
        '''

        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx=padding_idx)
        self.positional_embeddings = nn.Embedding(max_seq_length, hidden_dim)
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor, 
                ) -> torch.Tensor:
        
        '''
        Define the forward pass of the Embedding Layers
        '''
        
        B = token_ids.size(0)
        T = token_ids.size(1)

        positions = torch.arange(T, device=token_ids.device, dtype=torch.long)

        position_embeddings = self.positional_embeddings(positions).unsqueeze(0)
        token_embeddings = self.word_embeddings(token_ids)

        embeddings = token_embeddings + position_embeddings
        return embeddings

class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        '''
        Arguments:
        hidden_size: The total size of the hidden layer (across all heads)
        num_heads: The number of attention heads to use
        '''

        super().__init__()

        '''
        Initialize the Multi-Headed Attention Layer
        '''

        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

        self.scale_factor = 1.0 / (self.head_size ** 0.5)


    def forward(self, embeds: torch.Tensor, 
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        '''
        Arguments:
        embeds: The input embeddings to compute the attention over
        mask: A boolean mask of which tokens are valid to use for computing attention (see collate below)
        '''
   
        B, T, D = embeds.size()
        H = self.num_heads
        
        query = self.query_proj(embeds).view(B, T, H, self.head_size).transpose(1, 2)
        key = self.key_proj(embeds).view(B, T, H, self.head_size).transpose(1, 2)
        value = self.value_proj(embeds).view(B, T, H, self.head_size).transpose(1, 2)

        attention_scores = (query @ key.transpose(-1, -2)) * self.scale_factor

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))
            
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        
        weighted_sum = attention_weights @ value
        weighted_sum = weighted_sum.transpose(1,2).contiguous()
        
        output = self.output_layer(weighted_sum.view(B, T, -1))

        return output, attention_weights
        
def feed_forward_layer(
    hidden_size: int, 
    feed_forward_size: Optional[int] = None, 
    activation: nn.Module = nn.GELU()
):
    '''
    Arguments:
      - hidden_size: The size of the input and output of the feed forward layer. 
      - feed_forward_size: The size of the hidden layer in the feed forward network. If None, defaults to 4 * hidden_size. This size
        specifies the size of the middle layer in the feed forward network.
      - activation: The activation function to use in the feed forward network

    Returns: 
    '''

    feed_forward_size = feed_forward_size or 4 * hidden_size

    feed_forward = nn.Sequential(
        nn.Linear(hidden_size, feed_forward_size),
        activation,
        nn.Linear(feed_forward_size, hidden_size)
    )

    return feed_forward

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 256, # NOTE: normally 768, but keep it small for homework
        num_heads: int = 12,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
    ):
        super().__init__()
        
        self.multihead_attention = MultiHeadedAttention(hidden_size, num_heads)
        
        self.feed_forward = feed_forward_layer(hidden_size, feed_forward_size, activation)
        
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_size = feed_forward_size

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:

        if self.dropout.p > 0:
            return self.dropout(x)
        else:
            return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        '''
        Returns the output of the transformer encoder layer and the attention weights from the self-attention layer
        '''

        attention_output, attention_weights = self.multihead_attention(x, mask)
        
        attention_output = x + self.maybe_dropout(attention_output)
        
        ff_output = self.feed_forward(attention_output)
        
        output = attention_output + self.maybe_dropout(ff_output)
        
        return output, attention_weights

class MLMHead(nn.Module):
    def __init__(self, word_embeddings: nn.Embedding):
        '''
        Arguments:
            word_embeddings: The word embeddings to use for the prediction
        '''

        super().__init__()
        self.word_embeddings = word_embeddings

    def forward(self, x):
        '''
        x: The input tensor to the MLM head containing a batch of sequences of
           contextualized word embeddings (activations from the transformer encoder 
           layers)
        '''
        
        return x @ self.word_embeddings.weight.transpose(0, 1)

class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()

        self.pooler_layer = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        cls = x[:, 0]
        
        pooled_output = self.pooler_layer(cls)

        pooled_output = self.activation(pooled_output)

        return pooled_output

class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        mode: str = "mlm",
        num_classes: Optional[int] = None,
    ):
        '''
        Defines BERT model architecture. Note that the arguments are the same as the default
        BERT model in HuggingFace but we'll be training a *much* smaller model for this homework.

        Arguments:
        vocab_size: The size of the vocabulary (determined by the tokenizer)
        padding_idx: The index of the padding token in the vocabulary (defined by the tokenizer)
        hidden_size: The size of the hidden layer and embeddings in the transformer encoder
        num_heads: The number of attention heads to use in the transformer encoder
        num_layers: The number of layers to use in the transformer encoder (each layer is a TransformerEncoderLayer)
        dropout: The dropout rate to use in the transformer encoder (what % of times to randomly zero out activations)
        activation: The activation function to use in the transformer encoder
        feed_forward_size: The size of the hidden layer in the feed forward network in the transformer encoder. If None, defaults to 4 * hidden_size
        mode: The mode of the BERT model. Either "mlm" for masked language modeling or "classification" for sequence classification
        num_classes: The number of classes to use in the classification layer.
        '''


        super().__init__()

        self.mode = mode

        self.embedding = BertPositionalEmbedding(vocab_size, hidden_size, padding_idx)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, num_heads, dropout, activation, feed_forward_size)
             for _ in range(num_layers)]
        )

        self.mlm_head = MLMHead(self.embedding.word_embeddings)

        self.pooler = Pooler(hidden_size)

        if mode == "classification" and num_classes is not None:

            self.classification = nn.Linear(hidden_size, num_classes)

        self.apply(self.init_layer_weights)


    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        '''
        arguments:
        x: The input token ids
        mask: The attention mask to apply to the input (see the collate function below)
        '''
        
        embeddings = self.embedding(x)

        attention_weights = []
        for encoder_layer in self.encoder_layers:
            
            embeddings, attn_weights = encoder_layer(embeddings, mask)
            attention_weights.append(attn_weights)
        
        if self.mode == "mlm":

            output = self.mlm_head(embeddings)
            
        elif self.mode == "classification":
            pooled_output = self.pooler(embeddings[:,0])
            output = self.classification(pooled_output)

        return output, attention_weights
    
    def init_layer_weights(self, module):
        
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        pass



# # Part 3: Training

class MLMDataset(Dataset):
    def __init__(self, tokenizer, data: list[str], max_seq_length=128, mlm_probability=0.15):

        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.data)
    
    def tokenize(self):
        '''
        Tokenizes the text in self.data, performing any preprocessing and storing the tokenized data in a new list.
        '''

        tokenized_data = []
        for text in self.data:
            
            tokens = self.tokenizer.encode(text).ids

            tokens = tokens[:self.max_seq_length] # might need to do -1 and add a sep?
            tokenized_data.append(tokens)
        self.data = tokenized_data

    def __getitem__(self, idx):
        '''
        Returns the list of the token ids of an instance in the dataset and a list of the labels for MLM (one label per token).
        '''

        token_ids = self.data[idx] 
        input_ids = deepcopy(token_ids)
        labels = deepcopy(token_ids)

        for i, token_id in enumerate(token_ids):

            if token_id in [101, 102, 0]:
                labels[i] = -100
            else:
                if random.random() < self.mlm_probability:
                    input_ids[i] = self.tokenizer.token_to_id("[MASK]")
                else:
                    labels[i] = -100
    
        return input_ids, labels



# Let's generate with our real data!
# review_data_path = 'test.txt'
review_data_path = './reviews-word2vec.med.txt' # <- for sanity checking / debugging
# review_data_path = './reviews-word2vec.large.txt' # <- for CPU pre-training and validating
#review_data_path = './reviews-word2vec.larger.txt.gz' <- for GPU pre-training and validating (Part 3.5)

ofunc = gzip.open if review_data_path.endswith('gz') else open
with ofunc(review_data_path, 'rt') as f:
    reviews = f.readlines()
    reviews = [review.strip() for review in reviews]

dataset = MLMDataset(tokenizer, reviews)
dataset.tokenize()

def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    '''
    A function that takes a list of instances in the dataset and collates them into a batch.
    '''

    input_ids, labels = zip(*batch)

    padded_input_ids = nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=tokenizer.token_to_id("[PAD]"))
    padded_labels = nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in labels], batch_first=True, padding_value=-100) 

    attention_mask = (padded_input_ids != tokenizer.token_to_id("[PAD]")).long()

    return padded_input_ids, attention_mask, padded_labels

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# If you are feeling _really_ adventurous, you can try to speed up your model by trying a few of the much fancier things in pytorch. In practice, you should not need these for the homework. However, they can be fun to explore even with a CPU, though they make the biggest impact if you have access to a GPU too
# 
# - Try using [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to optimize the `nn.Module` code (this tries to pre-compile the computation graph)
# - Rather than train with 32-bit floating point, use [amp](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) to do mixed-precision training (i.e., fewer bits and faster). If you want to test this on Great Lakes, the GPUs there support "fp16" which will greatly speed up training. At the moment, `amp` isn't supported on "mps" devices, though it might show up [soon](https://github.com/pytorch/pytorch/issues/88415).
# - Implement an alternative training loop using [`accelerate`](https://github.com/huggingface/accelerate) and try using mixed precision (fp8, fp16, bf16) training

# check if gpu is available
device = 'cpu' 
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using '{device}' device")

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens in the loss calculation
learning_rate = 4e-5
epochs = 5

bert = BERT(vocab_size=tokenizer.get_vocab_size(), 
            hidden_size=256, 
            num_heads=4, 
            num_layers=2,
        )

optimizer = torch.optim.AdamW(bert.parameters(), lr=learning_rate)

losses = []

wandb.init(project="Homework3", name="Great_lakes_test_med")

bert.to(device)
loss_fn.to(device)

bert.train()

for epoch in trange(epochs, desc="Epoch"):
    epoch_loss = 0.0
    num_batches = 0
    
    for i, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader, position=1, leave=True, desc="Step")):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, attn = bert(input_ids, attention_mask)  

        B, T, V = outputs.shape
        outputs = outputs.view(B * T, V)
        labels = labels.view(B * T)

        # Calculate the loss
        loss = loss_fn(outputs, labels)
        
        # Backpropagation
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        # Log the loss
        epoch_loss += loss.item()
        num_batches += 1

        if (i + 1) % 100 == 0:
            wandb.log({"Loss": loss.item()}, step=epoch * len(dataloader) + i)

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs}, Avg. Loss: {avg_epoch_loss}")

    # Save the model (optional)
    torch.save(bert.state_dict(), f"bert_epoch_{epoch + 1}.pt")


plt.plot(losses)


torch.save(bert.state_dict(), f"bert_fully_trained_{review_data_path}.pt")

