import pygame
import pygame_textinput
from string import ascii_uppercase
import numpy as np
import torch
from time import sleep
import keyboard

from args import args
from friend import Friend



red_face    = pygame.image.load("images/red_face.png")
red_face    = pygame.transform.scale(red_face,      (args.face_size, args.face_size))
yellow_face = pygame.image.load("images/yellow_face.png")
yellow_face = pygame.transform.scale(yellow_face,   (args.face_size, args.face_size))
green_face  = pygame.image.load("images/green_face.png")
green_face  = pygame.transform.scale(green_face,    (args.face_size, args.face_size))



def box_size(space):
    x = space[1][0] - space[0][0]
    y = space[1][1] - space[0][1]
    return(x,y)

def box_here(screen, space, color = (255,255,255), alpha = 255):
    box = np.full((1, 1, 3), color)
    box = pygame.surfarray.make_surface(box)
    box = pygame.transform.scale(box, box_size(space))
    box.set_alpha(alpha)
    screen.blit(box, space[0])
    
def mouse_in_box(pos, box):
    x = pos[0] >= box[0][0] and pos[0] <= box[1][0]
    y = pos[1] >= box[0][1] and pos[1] <= box[1][1]
    return(x and y)



class Game:
    def __init__(self, friend = None):
        if(friend == None):
            self.friend = Friend()
        else:
            self.friend = friend
        self.steps = 0
        self.robo_image = np.full((args.image_size, args.image_size), .5)
        self.robo_text = pygame_textinput.TextInputVisualizer()
        self.user_image = np.full((args.image_size, args.image_size), .5)
        self.user_text = pygame_textinput.TextInputVisualizer()
        self.rating = 0
        self.hidden = None
        
        self.screen = pygame.display.set_mode(box_size(args.game_space))
        pygame.init()
        self.pause = True
        self.mouse_down = False

    def background(self):
        self.pause_screen_up = False
        box_here(self.screen, args.game_space, (0,0,0))
        box_here(self.screen, args.robo_image_space)
        box_here(self.screen, args.robo_text_space)
        box_here(self.screen, args.user_image_space)
        box_here(self.screen, args.user_text_space)
        box_here(self.screen, args.rating_space)
        box_here(self.screen, args.rating_line, (100,100,100))
        box_here(self.screen, args.rating_center, (0,0,0))
        self.screen.blit(red_face, args.rating_space[0])
        self.screen.blit(green_face, (args.rating_space[1][0] - args.face_size, args.rating_space[0][1]))
        
    def foreground(self):
        robo_image = np.repeat(self.robo_image[:, :, np.newaxis], 3, axis=2)
        robo_image = robo_image * 255
        robo_image = pygame.surfarray.make_surface(robo_image.astype(int))
        robo_image = pygame.transform.scale(robo_image, box_size(args.robo_image_space))
        self.screen.blit(robo_image, args.robo_image_space[0])
        user_image = np.repeat(self.user_image[:, :, np.newaxis], 3, axis=2)
        user_image = user_image * 255
        user_image = pygame.surfarray.make_surface(user_image.astype(int))
        user_image = pygame.transform.scale(user_image, box_size(args.user_image_space))
        self.screen.blit(user_image, args.user_image_space[0])
        self.screen.blit(yellow_face, (self.rating_to_pos(), args.rating_space[0][1]))
        self.screen.blit(self.robo_text.surface, args.robo_text_space)
        self.screen.blit(self.user_text.surface, args.user_text_space)
        
    def rating_to_pos(self):
        pos = (self.rating + 1)/2
        pos *= args.game_space[1][0] - args.face_size
        return(pos)
    
    def pos_to_rating(self, x):
        rating = (x+1)/args.game_space[1][0]
        rating = (rating * 2) - 1
        return(rating)
    
    def pause_screen(self):
        box_here(self.screen, args.game_space, (150,150,150), 128)
    
    def one_step(self):
        cont = self.steps < args.episode_length
        for event in pygame.event.get():
            if event.type == pygame.QUIT:            
                cont = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
        self.background()
        self.foreground()     
        if(keyboard.is_pressed("enter")): self.pause = False
        else:                             self.pause = True

        if(self.pause): 
            self.pause_screen()
            pos = pygame.mouse.get_pos()
            if(self.mouse_down and mouse_in_box(pos, args.rating_space)):
                self.rating = self.pos_to_rating(pos[0])
            if(mouse_in_box(pos, args.user_text_space)):
                self.user_text.update(pygame.event.get())
                self.user_text.value = self.user_text.value[-args.text_length-1:].upper()
                self.user_text.value = \
                    "".join([c for c in self.user_text.value
                             if c in ascii_uppercase + " "])
        else:           
            self.steps += 1
            self.seed = torch.rand(args.seed_size)
            self.robo_image, self.robo_text.value, self.hidden = \
                self.friend.act(self.user_image, self.user_text.value, self.seed, self.hidden)
            self.friend.buffer.push(
                self.user_image, 
                self.user_text.value, 
                self.seed, 
                self.robo_image, 
                self.robo_text.value, 
                self.rating, 
                self.user_image,      # Technically,
                self.user_text.value, # these should be
                self.seed,            # the "next" ones.
                cont, cont)

        

        if(not cont): 
            box_here(self.screen, args.game_space, (0,0,0))
            print("Episode over!")
        pygame.display.flip()
        sleep(0 if self.pause else args.sleep_time)
        return(cont)

