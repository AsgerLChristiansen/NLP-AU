import torch
from torch import nn
import torch.nn.functional as F #

class RNN(nn.Module):
    def __init__(self, 
                output_dim: int,  # Takes an output dimension
                embedding_layer: nn.Embedding,  # Takes a pre-trained nn.Embedding model.
                hidden_dim_size: int): # Takes a hidden dimension size.
        super().__init__() # Initializes more stuff.

        # maps each token to an embedding_dim vector using our word embeddings
        self.embedding = embedding_layer
        self.embedding_size = embedding_layer.weight.shape[1]

        # the LSTM takes an embedded sentence
        self.lstm = nn.LSTM(self.embedding_size, # nn.LSTM needs an input size
                            hidden_dim_size, # And a hidden dim size
                            batch_first=True) # And a "batch first" argument

        # fc (fully connected) layer transforms the LSTM-output to give the final output layer
        self.fc = nn.Linear(hidden_dim_size,  # Standard hidden layer. Needs input size
                            output_dim) # and output size.

    def forward(self, X):
        # apply the embedding layer that maps each token to its embedding
        x = self.embedding(X)  # dim: batch_size x batch_max_len x embedding_dim
        # I.e.: Each word per sequence (batch_max_len), in each batch (batch_size), has a 50-dimensional vector (Embedding dim)
        # It's easier, perhaps, to see it as a sequence*embedding matrix, in 32 layers.

        # run the LSTM along the sentences of length batch_max_len
        x, _ = self.lstm(x)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # reshape the Variable so that each row contains one token
        # such that it is a valid input to the fully connected layer
        x = x.reshape(-1, x.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output for each token
        x = self.fc(x)  # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(x, dim=1)  # dim: batch_size*batch_max_len x num_tags

    @staticmethod
    def loss_fn(outputs, labels):
        """
        Custom loss function.
        In the section on preparing batches, we ensured that the labels for the PAD tokens were set to -1.
        We can leverage this to filter out the PAD tokens when we compute the loss.
        """
        #reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)
        print("Print what the hell labels is:", labels)
        #mask out 'PAD' tokens
        mask = (labels >= 0).float()
        print("What the hell is mask?", mask)
        #the number of tokens is the sum of elements in mask
        bla = torch.sum(mask)
        print("Wtf is torch.sum(mask)?", bla)
        num_tokens = int(bla)
        print("Wtf is int(torch.sum)?", num_tokens)
        #num_tokens = int(torch.sum(mask))

        #pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask

        #cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens