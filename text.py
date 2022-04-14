import torch
from string import ascii_uppercase
from itertools import chain

from args import args

text_order = ascii_uppercase + " "
num_letters = len(text_order)



# How to turn a list of strings into a tensor for recurrence

def letter_to_one_hot(letter):
    index = text_order.index(letter)
    one_hot = [0]*num_letters
    one_hot[index] = 1
    return(one_hot)

def text_to_one_hots(text):
    while(len(text) < args.text_length):
        text += " "
    one_hots = list(chain(*[letter_to_one_hot(c) for c in text]))
    return(one_hots)

def texts_to_one_hots(texts):
    one_hots = [text_to_one_hots(text) for text in texts]
    one_hots = torch.tensor(one_hots)
    return(one_hots)



# How to turn one tensor into a string.

def one_hot_to_letter(one_hot):
    max_index = one_hot.index(max(one_hot))
    one_hot = [1 if i==max_index else 0 for i in range(len(one_hot))]
    letter = text_order[one_hot.index(1)]
    return(letter)

def one_hots_to_text(one_hots):
    one_hots = [one_hots[i:i+num_letters] for i in range(0, len(one_hots), num_letters)]
    text = [one_hot_to_letter(one_hot) for one_hot in one_hots]
    text = "".join(text)
    return(text)