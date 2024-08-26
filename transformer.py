import numpy as np
from attention import MultiHeadAttention
from embed import tokenize_and_embed, add_positional_encoding, embedding_model

import numpy as np
import random

class Transformer:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.output_projection = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)

    def forward(self, embeddings):
        embeddings_with_pos = add_positional_encoding(embeddings)
        attention_output = self.multi_head_attention.forward(embeddings_with_pos)
        output = self.linear_transformation(attention_output, self.output_projection)
        return output

    def linear_transformation(self, x, weight_matrix):
        return np.dot(x, weight_matrix)

    def predict_next_word(self, sentence, temperature=1.0, top_k=5):
        embeddings = tokenize_and_embed(sentence, embedding_model)
        output = self.forward(embeddings)
        
        # Apply softmax to get probabilities
        probs = self.softmax(output[-1] / temperature)
        
        # Sample from the top-k words instead of greedy argmax
        top_k_indices = np.argsort(probs)[-top_k:]
        chosen_index = random.choice(top_k_indices)
        next_word = embedding_model.index_to_key[chosen_index]
        
        return next_word
    
    def complete_sentence(self, sentence, max_length=20):
        for _ in range(max_length):
            next_word = self.predict_next_word(sentence)
            sentence += " " + next_word
            if next_word == "<EOS>":  # Assuming <EOS> is the end of sequence token
                break
        return sentence

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test the Transformer model with sentence completion
embedding_dim = 300  # GloVe embedding dimension
num_heads = 6  # Number of heads

transformer = Transformer(embedding_dim, num_heads)
sentence = "this is a test"
completed_sentence = transformer.complete_sentence(sentence)
print("Completed Sentence:")
print(completed_sentence)
