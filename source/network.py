# Neural network implementation

from source.routines import *

# LSTM architecture
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # Creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    # Forward method
    def forward(self, x, hidden):

        # Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        # Pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view (use contiguous to reshape the output)
        out = out.contiguous().view(-1, self.n_hidden)

        # Put x through the fully-connected layer
        out = self.fc(out)

        # Return the final output and the hidden state
        return out, hidden

    # Initializes hidden state
    def init_hidden(self, batch_size):
        
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
