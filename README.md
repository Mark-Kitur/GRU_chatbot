**Dataset Download Link**
Cornell Movie-Dialogs Corpus 
Download the ZIP file here:
https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip

**GRU Seq2Seq Chatbot with Luong Attention**
A lightweight, fully custom implementation of a neural conversational model using PyTorch. This project demonstrates how to build and train a sequence-to-sequence (seq2seq) chatbot using GRUs and Luong attention, including data preprocessing, batching, masking, and inference.

**Features**

Encoder–decoder architecture using GRU layers
Luong attention (dot, general, concat)
Teacher forcing with configurable ratio
Masked cross-entropy loss for padded batches
Greedy decoding for inference
Full training pipeline with gradient clipping
Checkpoint saving and loading (encoder, decoder, embeddings, optimizers, vocabulary)

**Architecture Overview**

Encoder
Processes the input sentence using a multi-layer GRU. Produces:
encoder_outputs: hidden states for attention
encoder_hidden: final hidden state passed to the decoder
Decoder
Runs autoregressively to generate output tokens:
Token embeddings
GRU cell
Luong attention
Context concatenation layer
Linear projection to vocabulary logits
Attention
Luong attention computes alignment scores between decoder states and encoder outputs to form a weighted context vector.

**Data Processing**

Utilities included:
Sentence normalization
Vocabulary construction
Index conversion (indexesFromSentence)
Batch padding (zeroPadding)
Binary masks (binaryMatrix)
Batch preparation (batch2trainData)
These ensure that variable-length sequences can be trained in batches efficiently.

**Training**

Training is performed in iterations over randomly sampled sentence pairs.
Key components:
Teacher forcing
Masked negative log likelihood
Gradient clipping
Batch-wise propagation through encoder and decoder
Periodic loss reporting
