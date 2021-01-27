import numpy as np

# Initial data: one entry per sample (in this example, a sample is a sentence, but it could be an entire document)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
print('samples: ', samples)
# Builds an index of all tokens in the data
token_index = {}

# Tokenizes the samples via the split method. 
# In real life, you'd also strip punctuation and special characters from the the samples.
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            # Assigns a unique index to each unique word. 
            # Note that you don't attribute index 0 to anything.
            token_index[word] = len(token_index) + 1

print('Token index:', token_index)

# Vectorizes the samples. You'll only consider the first max_lenght words in each sample.
max_length = 10

# This is where you store the results.
results =  np.zeros(shape=(
                len(samples),
                max_length, 
                max(token_index.values())+1
            ))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print('Result:', results)