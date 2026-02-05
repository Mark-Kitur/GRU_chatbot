**GRU Seq2Seq Chatbot with Luong Attention**

A fully custom, lightweight sequence-to-sequence (seq2seq) neural conversational model built with PyTorch. This project demonstrates a complete NLP pipeline from raw text preprocessing to training, evaluation, and inference. It closely follows research-grade implementations while remaining easy to read and extend.

**Dataset**

Cornell Movie-Dialogs Corpus
Download ZIP:
https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip

**Overview**

This repository implements a GRU-based encoder–decoder chatbot enhanced with Luong attention. It includes the full training pipeline, batching utilities, vocabulary management, checkpointing, and a terminal-based inference interface.

**Features**

Encoder–decoder architecture using multi-layer GRUs
Luong attention (dot, general, concat)
Configurable teacher forcing
Masked cross-entropy for padded batches
Gradient clipping for stable training
Greedy decoding for inference
Full checkpoint system (encoder, decoder, embeddings, optimizers, vocabulary)
Clean modular code structured as a real-world ML pipeline

**Architecture**
**Encoder**
A multi-layer GRU processes token embeddings and outputs:
encoder_outputs: hidden states used for attention
encoder_hidden: final hidden state for initializing the decoder

**Decoder**
Autoregressive GRU-based decoder containing:
Embedding layer

GRU cell
Luong attention module
Context concatenation
Linear output projection into vocabulary space
Attention (Luong)
Implements the score functions:
Dot
General
Concat

Attention creates a context vector that weights encoder outputs based on decoder state similarity.
Data Processing Pipeline
The following utilities implement standardized NLP preprocessing:
Sentence normalization (lowercasing, punctuation handling, trimming)
Vocabulary construction with token frequency counts
Index conversion (indexesFromSentence)
Batch padding (zeroPadding)
Binary masks (binaryMatrix)
Batch assembly (batch2trainData)
These ensure efficient training with variable-length sequences.
Training
Training executes over randomly sampled sentence pairs with:
Teacher forcing
Masked negative log-likelihood loss
Optimizer step updates
Gradient clipping to prevent exploding gradients
Periodic loss reporting

**Checkpoint saving during training**
Inference
Inference is performed via greedy decoding directly in the terminal.
Given an input sentence, the model encodes, attends, and generates a response token-by-token.

**Checkpoints**
Saved components include:
Encoder state dict
Decoder state dict
Optimizer states
Embedding weights
Vocabulary object
This allows full training recovery and reproducible experiments.