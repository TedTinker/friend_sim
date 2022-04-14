from game import Game
from friend import *
import torch

friend = torch.load("friend.pt")

game = Game(friend)
run = True
while run: 
    run = game.one_step()
friend.update_networks()
torch.save(friend, 'friend.pt')