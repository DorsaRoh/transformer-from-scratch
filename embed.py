import numpy as np
import gensim.downloader as api


# Goal: with a string of input (ex. a sentence), next word is predicted.
#       Should be able to form/complete a coherent sentence



# Part 1: Tokenize, embed, positional encode

# Steps:
    # Self Attention
        # 1. Load the embedding matrix
        # 2. Split input sentence into words (tokens)
        # 3. Create an array of embeddings of the tokens of the input sentence
        # 4. Add positional encoding to the token embeddings
    # Multi-Head Attention
        # 1. 


# Embedding matrix
embedding_model = api.load('glove-wiki-gigaword-300')


def get_embedding(word: str, embedding_model) -> np.ndarray:
    if word in embedding_model:
        return embedding_model[word]
    else: # if word is not in vocab
        return np.zeros(embedding_model.vector_size)

def tokenize_and_embed(word:str, embedding_model) -> list:
    tokens = word.split()  # split input sentence into words (tokens)
    embeddings = np.array([get_embedding(word, embedding_model) for word in tokens])
    return embeddings

def add_positional_encoding(embeddings: np.ndarray) -> np.ndarray:   
    sequence_len = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    # Initialize  positional encoding matrix
    pos_enc_matrix = np.zeros((sequence_len, embedding_dim))

    # Calculate the positional encodings
    for pos in range(sequence_len):
        for i in range(embedding_dim): 
            # even index
            if i % 2 == 0: 
                pos_enc_matrix[pos, i] = np.sin(pos / (10000 ** (i/embedding_dim)))
            else: # odd index
                pos_enc_matrix[pos, i] = np.cos(pos/(10000**(i/ embedding_dim)))

    # Add positional encodings
    embeddings_with_pos = embeddings + pos_enc_matrix
    return embeddings_with_pos


# print(add_positional_encoding(tokenize_and_embed("this is a test", embedding_model)))










