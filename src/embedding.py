import torch
from torch import nn

import numpy as np

def gensim_to_torch_embedding(gensim_wv): # Input is a pre-trained gensim model.
    """
    - Add type hints on input and output
    - add function description
    - understand the pad and unk embeddings, add an argument which makes these optional. 
        E.g. add_padding = True and add_unknown = True
    """

    embedding_size = gensim_wv.vectors.shape[1] # gensim_wv.vectors is a matrix, with each row being a word2vec. 
    #We want the length of such a row, equivalent to the number of columns (vector space dimensions) - that's our embedding size.

    # create unknown and padding embedding
    bla = np.mean(gensim_wv.vectors, axis=0)
    #print("np.mean(the_gensim_model.vectors, axis=0) gives the following output:")
    #print(bla)
    #print(type(bla))
    #blashape = bla.shape
    unk_emb = bla.reshape((1,embedding_size))
    #print("reshaped from ", blashape, " to (1, embedding_size), it looks like this:")
    #print(unk_emb)
    #print("In short, it represents the mean of all embeddings - which intuitively seems to be a reasonable estimate of the maximum distance you can get from all other vectors in the space.")
    
    #print("Now for pad_emb. It's encoded as np.zeros((1, the_gensim_model.vectors.shape[1])). Now why would you do that?")
    pad_emb = np.zeros((1, gensim_wv.vectors.shape[1]))
    #print(pad_emb)
    #print(pad_emb.shape)

    # add the new embedding
    #print("Let's add these wonderful embeddings to our gensim model. From ", gensim_wv.vectors.shape, " to:")
    embeddings = np.vstack([gensim_wv.vectors, unk_emb, pad_emb]) # Extracting array from gensim model, adding dimensionality
    #print(embeddings.shape)
    #print(embeddings[1])
    weights = torch.FloatTensor(embeddings) # Turning array into tensor containing floats.
    #print(type(weights))
    #print(weights[1])

    emb_layer = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=-1) # And voila! From torch, we've imported nn.
    # Here, we're creating an nn.Embedding in the same way we'd create an nn.Linear. Embedding has a .from_pretrained method,
    # which takes a torch tensor as input... and evidently has a variable for what a padding idx should be. FASCINATING!

    # So let's review. So far we have done the following:

    # Loaded a gensim model object (KeyedVectors object to be specific) into a function. We specify the vector size,
    # we add "Unknown" (best guess is the mean vector) and "Padding" (the origin) as dimensionalities in the vectors matrix,
    # then we turn it into a tensor (because nn is torch based!), and create an "embedding" layer using nn,
    # which can take a token and localize it in vector space.
    # This embedding layer also has a "padding index" variable, which is probably useful somehow.

    # creating vocabulary
    vocab = gensim_wv.key_to_index # Create a dictionary that maps words to semantic-space-vectors (i.e., each vector is indexed as a row of 50 entries).
    vocab["UNK"] = weights.shape[0] - 2 # Couldn't you just have used gensim_wv.vectors.shape[0] here?
    #print(weights.shape[0] -2 == gensim_wv.vectors.shape[0]) # This outputs true!
    vocab["PAD"] = emb_layer.padding_idx # So PAD gets an index in the vocabulary of -1, *because its the last vector!*
    # This step seems a bit superfluous!

    return emb_layer, vocab # Output is tensor + vocab * vectorspacecoordinates matrix.