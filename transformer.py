import numpy as np
from attention import MultiHeadAttention, softmax
from embed import tokenize_and_embed, add_positional_encoding, embedding_model
import random


# Part 3: Transformer architecture
    # Input: sequence of tokens (ex. a string of sentences)
    # Output: a probability distribution of the possible next generated tokens. If desired, the completion of the sentence

# Steps:
    # 1. Initialize instance of Multi Head Attention class with the dimensions of the embedding and number of heads
    # 2. do rest here


class Transformer:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.output_projection = np.random.randn(embedding_dim, embedding_dim) 
        self.output_projection = self.output_projection * np.sqrt(1. / embedding_dim) # scale values down

    def forward(self, embeddings):
        # Add positional encoding 
        embeddings_with_pos = add_positional_encoding(embeddings) 

        # Output of MultiHeadAttention class
        attention_output = self.multi_head_attention.forward(embeddings_with_pos)

        # Apply final linear transformation
        output = self.linear_transformation(attention_output, self.output_projection)
        return output

    # Calculate linear transformation
    def linear_transformation(self, x, weight_matrix):
        return np.dot(x, weight_matrix)


    # Calculate next token
    def predict_next_word(self, sentence, temperature, top_k=5):

        # Tokenize and embed input sentence
        embeddings = tokenize_and_embed(sentence, embedding_model)
        output = self.forward(embeddings)
        
        # Apply softmax to get probabilities
        probs = self.softmax(output[-1] / temperature)
        
        # Sample from the top-k words instead of greedy argmax
        top_k_indices = np.argsort(probs)[-top_k:]
        chosen_index = random.choice(top_k_indices)
        next_word = embedding_model.index_to_key[chosen_index]
        
        return next_word
    
    # Complete the sentence from given input 
    def complete_sentence(self, sentence, max_length=20):
        for _ in range(max_length):
            next_word = self.predict_next_word(sentence)
            sentence += " " + next_word
            if next_word == "<EOS>":  # Assuming <EOS> is the end of sequence token
                break
        return sentence