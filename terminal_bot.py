import torch
import wordninja    
import os
import torch.nn as nn
import unicodedata
import re 
import pickle
device = torch.device('cpu')

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


PAD_token = 0
SOS_token = 1
EOS_token = 2

#index2oword
class Voc:
    def __init__(self,vocab):
        self.trimmed =False
        self.word2index={}
        self.word2count={}
        self.index2word = vocab
        self.num_words =3



voc = Voc(vocab)


def unicodetoascii(s):
    return "".join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )
def normalize_string(s):
    s = unicodetoascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1",s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    r = re.sub(r"\s+", r" ", s).strip()
    return s

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
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'general','concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size *2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)
    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output),2)).tanh()
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_output):
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_output)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_output)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_output)
        
        attn_energies = attn_energies.t()
        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, attn_model, embedding,hidden_size, output_size, n_layers, dropout=0.1):
        super().__init__()
        self.att_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout =dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size,hidden_size, n_layers, dropout=(0 if n_layers ==1 else dropout))
        self.concat = nn.Linear(hidden_size *2, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

        self.attn = attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context),1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = nn.functional.softmax(output, dim=1)

        return output, hidden


class GreedySearch(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs , encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1,1,device=device, dtype=torch.long) *SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        # Iteratively decode one word token at a time
        for _ in range(max_length):
            decoder_output ,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores , decoder_input = torch.max(decoder_output,dim=1)
            all_tokens = torch.cat((all_tokens,decoder_input), dim=0)
            all_scores = torch.cat((all_scores,decoder_scores),dim=0)
            decoder_input = torch.unsqueeze(decoder_input,0)

        return all_tokens, all_scores 

def indexes_from_sentence(voc,sentense):
    return [voc.word2index[word] for word in sentense.split(' ')] +[EOS_token]

def  evaluate(encoder, decoder, searcher, voc , sentence, max_length =10):
    indexes_batch = [indexes_from_sentence(voc,sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
    input_batch =input_batch.to(device)
    lengths = lengths.to(device)
    tokens , scores = searcher(input_batch, lengths,max_length)
    decoded_words = [vocab[token.item()] for  token in tokens]

    return decoded_words


# Important params
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

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
    decoder = Decoder(attn_model,embedding,hidden_size,voc.num_words, decoder_n_layers,dropout )

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    
    print('Model loaded successfully')
    return encoder, decoder

encoder , decoder =load_model(loadfilename)
searcher = GreedySearch(encoder, decoder)

def take_input():
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('>>> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = normalize_string(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_words[:] = [x for x in output_words if not (x =='EOS' or x =='PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")
   
take_input()
