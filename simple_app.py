import torch
import wordninja    
import os
import torch.nn as nn
device = torch.device('cpu')


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers ==1 else dropout), bidirectional=True)
        
    def forward(self, input_seq , input_length, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded,input_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:,:, self.hidden_size:]

        return outputs, hidden

class attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not a valid attention method")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size *2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output),2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        return nn.functional.softmax(attn_energies, dim=1)
    

# greedy search
class GreedySearch(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder =encoder
        self.decoder= decoder

    def forward(self, input_seq, input_length, max_lenth):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1,1,device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_lenth):
            decoder_output, decoder_hidden= self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores = decoder_input = torch.max(decoder_output,dim=1)
            all_tokens = torch.cat((all_tokens,decoder_input), dim=1)
            all_scores = torch.cat((all_scores,decoder_scores),dim=0)

            decoder_input =torch.unsqueeze(decoder_input,0)

        return all_tokens, all_scores



loadfilename = '/media/mark/New Volume/data_science/convo_bot/real_model_1.pth'

def load_model(filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename,map_location=torch.device('cpu'))

        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        voc.__dict__= checkpoint['voc_dict']
    
    print("Starting to build the model...")

    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = Decoder(attn_model, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder.eval()
    decoder.eval()

    print('Model loaded successfully')

