import numpy as np
from embed import get_embedding


# Part 2: Self Attention
    # Input: Query, Key and Value matrices
    # Output: matrix where each vector is the weighted sum of the Value vectors, where 
        # the weights come from the attention scores (which are based on the dot product of the Query and Key matrices)

# Steps:
    # 1. Create weight matrices (intialized randomly initally. same dimensions as embeddings)
    # 2. Get Query, Key values from embed.py (i.e. linear transformation applied to the vectors of the (word embeddings & positional encoding) with weight matrices, for each token)
    # 3. Calculate the attention score (dot product of the Query and Key matrices)
    # 4. Masking (optional here)
    # 5. Apply softmax to the (masked) attention scores (this is called normalization)
    # 6. Use attention scores to weigh the Value vectors
    # 7. Return step 5.


class SelfAttention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

        # Initialize weight matrices (with small random values)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)

        
    # Compute Query, Key, and Value matrices
    def forward(self, embeddings, mask=None):
        query = np.dot(embeddings, self.W_q)
        key = np.dot(embeddings, self.W_k)
        values = np.dot(embeddings, self.W_v)

        # Calculate attention scores
        attention_scores = self.calculate_attention_score(query, key)

        # Masking
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)      # where mask is 0, turns to -infinity. where mask is 1, keeps original values

        # Apply softmax to attention scores
        attention_weights = self.softmax(attention_scores)

        # Compute weighted sum of value vectors
        output = self.values_weighted_sum(attention_weights, values)

        return output
    

    # Calculate attention scores
    def calculate_attention_score(self, query, key):
        d_k = key.shape[-1]     # scaling factor to ensure no too large values are fed to softmax (would push softmax into regions where it has extremely small gradients)
        dot = np.dot(query, key.T) # key.T : transpose of the key matrix 
                                    # i.e. flipping the matrix over its diagonal, so that the rows become columns and the colums become rows
        return dot / np.sqrt(d_k)  # scale by the square root of the key dimension
    
    # Calculate softmax (normalization)
    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # numerical stability, and normalizes across each row (i.e. across all key vectors for each query)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Calcualte weighted sum
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

        # `embedding_dim` must be divisible by `num_heads`
            # otherwise, the context window will not be consistent (i.e. the input of each head will be different sizes)
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        
        # Calculate dimension of each head
        self.head_dim = embedding_dim // num_heads
        
        # Initialize heads (instances of self attention class)
        self.attention_heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        
        # Final transformation matrix (transform the concatenated outputs back to the original embedding dimension)
        self.W_o = np.random.rand(embedding_dim, embedding_dim)


    def forward(self, embeddings):
        # Split the embeddings into multiple heads
        sequence_length, embedding_dim = embeddings.shape
        split_embeddings = np.reshape(embeddings, (sequence_length, len(self.attention_heads), self.head_dim))

        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_output = head.forward(split_embeddings[:, i, :])
            head_outputs.append(head_output)
        
        # Concatenate outputs of all heads along the last axis
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        
        # Apply final linear transformation
        output = self.linear_transformation(concatenated_output, self.W_o)
        
        return output

    def linear_transformation(self, concatenated_output, weight_matrix):
        return np.dot(concatenated_output, weight_matrix)