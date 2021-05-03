#--- PARAMETERS ---#

# Choose an author to imitate
author = "Lope"
#author = "Verdaguer"
#author = "Donne"

# Name of file containing the data
filename = "data/"+author+"/processed.txt"

# First verse to start the sample poems
if author == "Lope":
    prime_verse = "versos de amor, conceptos esparcidos,"
elif author == "Verdaguer":
    prime_verse = "entre'ls arbres de l'illa delitosa"
elif author == "Donne":
    prime_verse = "no man is an island"

# Number of nodes of each layer
n_hidden = 512

# Number of hidden layers
n_layers = 3

# Batch size
batch_size = 128

# Length of the sequences
seq_length = 100

# Learning rate
lr = 1.e-3

# Weight decay
wd = 0.

# Number of epochs
n_epochs = 50

# Identifier
sufix = "_%s_n_layers_%d_n_hidden_%d_seq_length_%d_batch_size_%d_n_epochs_%d"%(author,n_layers,n_hidden,seq_length,batch_size,n_epochs)

# Name of the model
model_name = 'model'+sufix+'.net'
