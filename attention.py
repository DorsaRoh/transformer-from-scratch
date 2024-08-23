import numpy as np
from embed import get_embedding


# Part 2: Self Attention mechanism
    # Input: Query, Key and Value matrices
    # Output: matrix where each vector is the weighted sum of the Value vectors, where 
        # the weights come from the attention scores (which are based on the dot product of the Query and Key matrices)

# Steps:
    # 1. Create weight matrices (intialized randomly initally. same dimensions as embeddings)
    # 2. Get Query, Key values from embed.py (i.e. linear transformation applied to the vectors of the (word embeddings & positional encoding) with weight matrices, for each token)
    # 3. Calculate the attention score (dot product of the Query and Key matrices)
    # 4. Apply masking to the attention scores
    # 5. Apply softmax to the (masked) attention scores (this is called normalization)
    # 6. Use attention scores to weight the Value vectors
    # 7. Return step 6.



class SelfAttention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

        # initialize weight matrices
        self.W_q = np.random.rand(embedding_dim, embedding_dim)
        self.W_k = np.random.rand(embedding_dim, embedding_dim)
        self.W_v = np.random.rand(embedding_dim, embedding_dim)

    # compute Query, Key, and Value matrices
    def forward(self, embeddings):
        query = np.dot(embeddings, self.W_q)
        key = np.dot(embeddings, self.W_k)
        values = np.dot(embeddings, self.W_v)

        # calculate attention scores
        attention_scores = self.calculate_attention_score(query, key)

        # apply softmax to attention scores
        attention_weights = self.softmax(attention_scores)

        # compute weighted sum of value vectors
        output = self.values_weighted_sum(attention_weights, values)

        return output

    # attention scores
    def calculate_attention_score(self, query, key):
        return np.dot(query, key.T) # transpose of the key matrix 
                                    # i.e. flipping the matrix over its diagonal, so that the rows become columns and the colums become rows
    
    # normalization
    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # numerical stability, and normalizes across each row (i.e. across all key vectors for each query)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # weighted sum
    def values_weighted_sum(self, weights, values):
        return np.dot(weights, values)



# Part 3: Multi-Head Attention mechanism
    # Input: Query, Key and Value matrices
    # Output: matrix where each vector is the weighted sum of the Value vectors, where 
        # the weights come from the attention scores (which are based on the dot product of the Query and Key matrices)

    # Essentially multiple instances of Self Attention class running in parallel, each instance with different weight matrices


# Steps:
    # 1. Declare multiple heads/instances of Self Attention running in parallel
        # Each head/instance of Self Attention class focuses on different parts of the input by having its own set of weight matrices (W_q, W_k, W_v)
    # 2. Each heads/instances of Self Attention/s output is concatenated along the embedding dimension (input of each Self Attention class)
    # 3. Concatenated output is passed through a final linear transformation (a weight matrix)
        # To combine the information from all heads into a single output


class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # check that embedding_dim is divisible by num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # dimension of each head
        self.head_dim = embedding_dim // num_heads

        # create multiple SelfAttention heads
        self.attention_heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        
        # weight matrix for the final linear transformation
        self.W_o = np.random.rand(num_heads * self.head_dim, embedding_dim)


    def forward(self, embeddings):
        # apply each self-attention head
        head_outputs = [head.forward(embeddings) for head in self.attention_heads]
        
        # concatenate the outputs of all heads along the last axis
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        
        # apply the final linear transformation
        output = self.linear_transformation(concatenated_output, self.W_o)
        
        return output


    def linear_transformation(self, concatenated_output, weight_matrix):
        return np.dot(concatenated_output, weight_matrix)





# TEST CASE!

embedding_dim = 8  # example embedding dimension
num_heads = 2  # example number of heads
sequence_length = 4  # example sequence length

# a dummy embedding matrix (shape: [sequence_length, embedding_dim])
dummy_embeddings = np.random.rand(sequence_length, embedding_dim)

# Test SelfAttention
self_attention = SelfAttention(embedding_dim)
self_attention_output = self_attention.forward(dummy_embeddings)
print("SelfAttention Output:")
print(self_attention_output)
print("Shape:", self_attention_output.shape)

# # Test MultiHeadAttention
# multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
# multi_head_output = multi_head_attention.forward(dummy_embeddings)
# print("MultiHeadAttention Output:")
# print(multi_head_output)
# print("Shape:", multi_head_output.shape)
