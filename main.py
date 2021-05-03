# Main script to train the network
# Based on a tutorial from the Udacity course "Intro to Deep Learning with PyTorch" (https://classroom.udacity.com/courses/ud188/)

import matplotlib.pyplot as plt
import time, datetime
from source.routines import *
from source.network import *

time_ini = time.time()

# Set 1 for training the net, 0 for loading a model and predicting a sample text
train_net = 1

#--- MAIN ---#

# Open text file and read in data as `text`
with open(filename, 'r') as f:
    text = f.read()

# Encode the text and map each character to an integer and vice versa

# Create two dictionaries:
# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
print("Characters",sorted(chars))

# Encode the text
encoded = np.array([char2int[ch] for ch in text])

net = CharRNN(chars, n_hidden, n_layers)
print(net)

# If training
if train_net:

    # Train the model
    loss_train, loss_val = train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=lr, print_every=1)

    # Save a checkpoint
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}

    with open("models/"+model_name, 'wb') as f:
        torch.save(checkpoint, f)

    # Generate a sample poem
    sample_text = sample(net, 1000, prime=prime_verse, top_k=5)

    print("\n\n")
    print(sample_text)

    # Save the sample poem to a file
    text_out = open("sample_poem"+sufix+".txt", "w")
    text_out.write(sample_text)
    text_out.close()

    # Plot loss trend
    plt.plot(range(1,n_epochs+1),loss_train,"r-",label="Training")
    plt.plot(range(1,n_epochs+1),loss_val,"b:",label="Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Loss"+sufix+".pdf")

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))

# If not training, only load a model and sample poems
else:

    device = torch.device('cuda:0' if train_on_gpu else 'cpu')

    # Load a pretrained model
    with open("models/"+model_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)

    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    # Sample using the loaded model
    for i in range(5):
        sample_text = sample_lines(net, 8, prime=prime_verse, top_k=5)
        print("\n\n")
        print(sample_text)
