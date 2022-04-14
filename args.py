# Parameters for an arena.
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_size',         type=float,     default = 16)
    parser.add_argument('--seed_size',          type=float,     default = 128)
    parser.add_argument('--face_size',          type=float,     default = 56)
    parser.add_argument('--text_length',        type=float,     default = 16)    
    parser.add_argument('--episode_length',     type=int,       default = 128)
    parser.add_argument('--sleep_time',         type=float,     default = .1)

    parser.add_argument('--game_space',         type=tuple, 
                                                default = ((0, 0), (959, 673)))
    parser.add_argument('--robo_image_space',   type=tuple, 
                                                default = ((10, 10), (474, 462)))
    parser.add_argument('--robo_text_space',    type=tuple, 
                                                default = ((0, 473), (959, 534)))
    parser.add_argument('--user_image_space',   type=tuple, 
                                                default = ((485, 10), (949, 462)))
    parser.add_argument('--user_text_space',    type=tuple, 
                                                default = ((0, 612), (959, 673)))
    parser.add_argument('--rating_space',       type=tuple, 
                                                default = ((0, 545), (959, 601)))
    parser.add_argument('--rating_line',        type=tuple, 
                                                default = ((0, 565), (959, 581)))
    parser.add_argument('--rating_center',      type=tuple, 
                                                default = ((477, 565), (482, 581)))
    return parser.parse_args()

args = get_args()