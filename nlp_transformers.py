import torch
import torch.optim as optim
import torch.nn as nn
import spacy
from utils import translate_sentence, bleu, load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        # Embed size of input -> From words embedding
        self.embed_size = embed_size
        # Number of heads
        self.heads = heads
        # Dimention of each head
        self.head_dim = embed_size // heads

        # Check if head dimension is properly divided
        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be divisible by heads"

        # Values weight matrix
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        # Keys weight matrix
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        # Queries weight matrix
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        # Multi-head attention weight matrix
        self.fullyConnectedOut = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):

        # Batch size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Run through weights
        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # Split embedding into self.heads peices
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Computes Q(K^t) -> Query x Keys[transpose]
        # queris shape: (N, query_len, head, head_dim)
        # keys shape : (N, key_len, head, head_dim)
        # energy shape : (N, head, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask with -ve infinity. Here with e^-20 to reach very low value
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Calculate self attention with the formula
        # softmax(QK(t)/sqroot(embed_size))
        attention = torch.softmax(energy / (self.embed_size)**(1/2), dim=3)

        # Multiply attention matrix with value matrix
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, value_len, heads, heads_dim)
        # ouput shape : (N, query_len, heads, heads_dim) -> and flatten last two dimentions
        out = torch.einsum(
            "nhqk,nkhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        # Train final weight matrix for concatenated attention heads
        out = self.fullyConnectedOut(out)

        return out


class TranformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        # Define the multi-head attention
        self.attention = SelfAttention(embed_size, heads)
        # Define normalization layer 1
        self.norm1 = nn.LayerNorm(embed_size)
        # Define normalization layer 2
        self.norm2 = nn.LayerNorm(embed_size)

        # Define feed forward network.
        # Input: embed_size -> hidden layer : forward_expansion -> output : embed_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Self attention output
        attention = self.attention(value, key, query, mask)

        # First add and normalize.
        # Since all query, key and value matrix are same, add any one.
        norm1 = self.dropout(self.norm1(attention + query))

        # Forward it to feed forward neural network.
        forward = self.feed_forward(norm1)

        # Second add and normalize
        # Add the first normalized matrix
        out = self.dropout(self.norm2(forward + norm1))

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        droput,
        max_length,
    ):
        super().__init__()
        # Define embed size
        self.embed_size = embed_size
        # Define device. Either 'cpu' or 'cuda'
        self.device = device
        # Define word embedding in total input vocabulary size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # Define positional enbedding for all words
        # Max Length is to define the max input size it trains on. Anything more filters out.
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Define layers of transformers.
        self.layers = nn.ModuleList(
            [TranformerBlock(embed_size, heads, droput, forward_expansion)
             for _ in range(num_layers)]
        )

        # Define dropout
        self.dropout = nn.Dropout(droput)

    def forward(self, input, mask):
        # N : Batch size
        # seq_length : input length for each vector
        N, seq_length = input.shape

        # Generate position vector for positional embedding
        # Generate vector from 0 to seq_length : [ 0,1,2,3,...,(seq_length-1) ]
        # Expand the same across the batch : N
        # Transfer position vector to device
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        # Get final encoding with word embedding and positional embedding
        out = self.dropout(self.word_embedding(input) +
                           self.position_embedding(positions))

        # Encode embedded inputs over layers of encoders
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        # Define attention for multi-head attention
        self.attention = SelfAttention(embed_size, heads)
        # Define normalization
        self.norm = nn.LayerNorm(embed_size)
        # Define transformer block as majority of decoder block is a transformer
        self.transformer_block = TranformerBlock(
            embed_size, heads, dropout, forward_expansion)

        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, value, key, src_mask, target_mask):
        # Get masked multi-head attention
        attention = self.attention(input, input, input, target_mask)

        # Get query out for upper transformer.
        # Add attention to input and normalize
        query = self.dropout(self.norm(attention + input))

        # Sed Key, Value from encoder and Query fromm decoder to transformer
        out = self.transformer_block(value, key, query, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super().__init__()
        # Define word embedding in total output vocabulary size
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        # Define positional enbedding for all words
        # Max Length is to define the max input size it trains on. Anything more filters out
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # Define device
        self.device = device

        # Define layers of decoder blocks.
        self.layers = nn.ModuleList([DecoderBlock(
            embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])

        # Final linear layer maps to target_vocab to predict next word.
        self.fullyConnected = nn.Linear(embed_size, target_vocab_size)

        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_out, src_mask, target_mask):
        # N : Batch size
        # seq_length : input length for each vector
        N, seq_length = input.shape

        # Generate position vector for positional embedding
        # Generate vector from 0 to seq_length : [ 0,1,2,3,...,(seq_length-1) ]
        # Expand the same across the batch : N
        # Transfer position vector to device
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        # Get final encoding with word embedding and positional embedding
        out = self.dropout(self.word_embedding(
            input) + self.position_embedding(positions))

        # Decode embedded inputs with masks over layers of decoders
        for layer in self.layers:
            out = layer(out, encoder_out, encoder_out, src_mask, target_mask)

        # Pass through a linear layer. Maps to target_vocab
        out = self.fullyConnected(out)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_index,
        target_pad_index,
        embed_size=256,
        num_layers=4,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device='cuda',
        max_length=100,
    ):
        super().__init__()
        # Define encoder
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers,
                               heads, device, forward_expansion, dropout, max_length)
        # Define decoder
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers,
                               heads, forward_expansion, dropout, device, max_length)

        # Define Source pad index
        self.src_pad_index = src_pad_index
        # Define Target pad index
        self.target_pad_index = target_pad_index
        # Define device
        self.device = device

    def make_src_mask(self, src):
        """If src is same as src_pad_index, it will be set to 0, or else 1."""
        # src_mask shape : (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)

    def make_target_mask(self, target):
        # N : Batch size
        # seq_length : input length for each vector
        N, target_length = target.shape

        # Generate target mask for lower triangle and expand to entire batch size.
        target_mask = torch.tril(torch.ones((target_length, target_length))).expand(
            N, 1, target_length, target_length)

        return target_mask.to(self.device)

    def forward(self, src, target):
        # Generate source mask
        src_mask = self.make_src_mask(src)
        # Generate target mask
        target_mask = self.make_target_mask(target)
        # Encode source
        encode_src = self.encoder(src, src_mask)
        # Decode for target
        out = self.decoder(target, encode_src, src_mask, target_mask)
        return out

def preprocessing():
    """
    To install spacy laguages do:
    python3 -m spacy download en
    python3 -m spacy download de
    """
    # Loads the vocabulary space
    spacy_ger = spacy.load("de_core_news_sm")
    spacy_eng = spacy.load("en_core_web_sm")

    # Tokenize input for german
    def tokenize_ger(text):
        return [token.text for token in spacy_ger.tokenizer(text)]

    # Tokenize input for english
    def tokenize_eng(text):
        return [token.text for token in spacy_eng.tokenizer(text)]

    # Specify fields for german and english. Begin and end tokens are set
    german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
    english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

    # Split data to training, validation and testing data
    train_data, valid_data, test_data = Multi30k(root="data/",language_pair=("de", "en"))

    # Build the vocabulary of German and English
    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)

    return german, english, train_data, valid_data, test_data
    


if __name__=="__main__":
    print("Starting...\n")

    german, english, train_data, valid_data, test_data = preprocessing()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model save - load status
    load_model = False
    save_model = True

    # Training hyperparameters
    num_epochs = 5
    learning_rate = 3e-4
    batch_size = 32

    # Model hyperparameters
    src_vocab_size = len(german.vocab)
    target_vocab_size = len(english.vocab)
    embedding_size = 512
    num_heads = 8
    num_layers = 3
    dropout = 0.10
    max_length = 100
    forward_expansion = 4
    src_pad_index = english.vocab.stoi["<pad>"] 
    target_pad_index = german.vocab.stoi["<pad>"]   # Not being used in Transformer.

    # TensorBoard for nice plots
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    # Build Iterator for training, validation and test
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        (batch_size, batch_size, batch_size), 
        sort_within_batch = True,
        sort_key = lambda x: len(x.src),
        device = device,
    )

    # Create Model
    model = Transformer(
            src_vocab_size, 
            target_vocab_size, 
            src_pad_index, 
            target_pad_index,
            embedding_size, 
            num_layers, 
            forward_expansion, 
            num_heads, 
            dropout,
            device,
            max_length,
        ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Criterion to ignore target pad index
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_index)

    # Load model
    if load_model:
        load_checkpoint(torch.load("nlp_transformer.pth.tar"), model, optimizer)

    # Demo sentence
    sentence = "ein pferd geht unter einer brücke neben einem boot."

    # Run Epochs
    for epoch in range(num_epochs):
        print(f"[Epoch: {epoch}/{num_epochs}]")

        # Save model 
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, "nlp_transformer.pth.tar")
        
        # Generates a prediction on demo sentence
        model.eval()
        translated_sentence = translate_sentence(model, sentence, german, english, device, max_length = 100)

        print(f"Translated example sentence \n{translated_sentence}")
        
        # Trains the model
        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            print(batch_idx, batch)
            input = batch.src.to(device)
            target = batch.target.to(device)

            # Forward propagation. Target size is 1 less than Input.
            output = model(input, target[:-1])
            # Reshape to Batch size, target vocab size
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output,target)
            loss.backward()

            # Used to avoid exploding gradient problems
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 

            optimizer.step()

            # Used or plotting losses
            writer.add_scalar("Training Loss: ", loss, global_step=step)
            step += 1
    
    # Calculate Bleu score
    score = bleu(test_data, model, german, english, device)
    print(f"Bleu score: {score*100:.2f}")