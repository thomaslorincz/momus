import numpy as np
import string
from util import softmax
from util import rnn_forward
from util import rnn_backward
from util import update_parameters
from util import get_initial_loss
from util import initialize_parameters
from util import print_sample
from util import smooth
import json


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya",
        "db", "dby"
    maxValue -- everything above this number is set to this number, and
        everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    dWaa = gradients['dWaa']
    dWax = gradients['dWax']
    dWya = gradients['dWya']
    db = gradients['db']
    dby = gradients['dby']

    # Clip to mitigate exploding gradients
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db,
                 "dby": dby}

    return gradients


def sample(parameters, char_to_ix):
    """
    Sample a sequence of characters according to a sequence of probability
    distributions output of the RNN

    Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya,
            by, and b.
            char_to_ix -- python dictionary mapping each character to an index.

    Returns:
        indices -- a list of length n containing the indices of the sampled
            characters.
    """
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    by = parameters['by']
    b = parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a
    # probability distribution and append its index to "indices". We'll stop if
    # we reach 50 characters (which should be very unlikely with a well trained
    # model), which helps debugging and prevents entering an infinite loop.
    newline_character = char_to_ix['\n']

    while (idx != newline_character):
        # Forward propogate
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # Sample the index of a character within the vocabulary from the
        # probability distribution y
        idx = np.random.choice(
            list(range(0, vocab_size)),
            p=np.ndarray.flatten(y))

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the
        # sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a

    return indices


def optimize(X, Y, a_prev, parameters, vocab_size, learning_rate=0.001):
    """
    Execute one step of the optimization to train the model.

    Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
            b --  Bias, numpy array of shape (n_a, 1)
            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
        db -- Gradients of bias vector, of shape (n_a, 1)
        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, vocab_size, num_iterations=35000,
          n_a=50):
    """
    Trains the model and generates jokes.

    Arguments:
        data -- text corpus
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the model for
        n_a -- number of units of the RNN cell
        vocab_size -- number of unique characters in vocabulary

    Returns:
        parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, 10)

    # Shuffle list of all jokes
    np.random.seed(0)
    np.random.shuffle(data)

    # Initialize the hidden state
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):
        index = j % len(data)
        X = [None] + [char_to_ix[ch] for ch in data[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters,
                                                vocab_size, learning_rate=0.01)

        # Use a latency trick to keep the loss smooth.
        loss = smooth(loss, curr_loss)

        # Every 1000 iterations, generate n characters to check if the model is
        # learning properly
        if j % 1000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            for name in range(10):
                sampled_indices = sample(parameters, char_to_ix)
                print_sample(sampled_indices, ix_to_char)
            print('\n')

    return parameters


if __name__ == '__main__':
    with open('./data/jokes.json') as file:
        jokes = json.load(file)

    data = []
    for joke in jokes:
        title = ''.join([ch for ch in joke['title'] if (ch in string.printable)])
        body = ''.join([ch for ch in joke['body'] if (ch in string.printable)])
        data.append(title + body)

    chars = list(string.printable)
    data_size = len(data)
    print('Data size: %d' % data_size)

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

    parameters = model(data, ix_to_char, char_to_ix, vocab_size=len(chars))
