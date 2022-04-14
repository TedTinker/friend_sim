import gin
from collections import namedtuple
import numpy as np
import torch

from basics.utils import get_device

from args import args
from text import texts_to_one_hots, text_to_one_hots, one_hots_to_text

# U_I User image
# U_T User test
# S Seed
# R_I Robo image
# R_T Robo text
# R Reward
# D Done
# M Done
RecurrentBatch = namedtuple('RecurrentBatch', 'u_i u_t s r_i r_t r d m')


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(get_device())


@gin.configurable(module=__name__)
class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""

    def __init__(
        self,
        max_episode_len = args.episode_length + 1,  
        segment_len=None, 
        capacity=1000):

        # placeholders
        
        u_i_dim = (args.image_size, args.image_size, 1)
        u_t_dim = args.text_length * 27
        s_dim   = args.seed_size
        r_i_dim = (args.image_size, args.image_size, 1)
        r_t_dim = args.text_length * 27

        self.u_i = np.zeros((capacity, max_episode_len + 1) + u_i_dim)
        self.u_t = np.zeros((capacity, max_episode_len + 1, u_t_dim))
        self.s = np.zeros((capacity, max_episode_len + 1, s_dim))
        self.r_i = np.zeros((capacity, max_episode_len) + r_i_dim)
        self.r_t = np.zeros((capacity, max_episode_len, r_t_dim))
        self.r = np.zeros((capacity, max_episode_len, 1))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))
        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers

        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers

        self.starting_new_episode = True
        self.num_episodes = 0

        # hyper-parameters

        self.capacity = capacity
        self.u_i_dim = u_i_dim
        self.u_t_dim = u_t_dim
        self.s_dim = s_dim
        self.r_i_dim = r_i_dim
        self.r_t_dim = r_t_dim

        self.max_episode_len = max_episode_len

        if segment_len is not None:
            assert max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok

        self.segment_len = segment_len

    def push(self, u_i, u_t, s, r_i, r_t, r, n_u_i, n_u_t, n_s, d, cutoff):
        
        u_i = np.expand_dims(u_i, -1)
        n_u_i = np.expand_dims(n_u_i, -1)
        r_i = np.expand_dims(r_i, -1)
        u_t = text_to_one_hots(u_t)
        r_t = text_to_one_hots(r_t)

        # zero-out current slot at the beginning of an episode

        if self.starting_new_episode:

            self.u_i[self.episode_ptr] = 0
            self.u_t[self.episode_ptr] = 0
            self.s[self.episode_ptr] = 0
            self.r_i[self.episode_ptr] = 0
            self.r_t[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # fill placeholders

        self.u_i[self.episode_ptr, self.time_ptr] = u_i
        self.u_t[self.episode_ptr, self.time_ptr] = u_t
        self.s[self.episode_ptr, self.time_ptr] = s
        self.r_i[self.episode_ptr, self.time_ptr] = r_i
        self.r_t[self.episode_ptr, self.time_ptr] = r_t
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # fill placeholders

            n_u_t = text_to_one_hots(n_u_t)
            self.u_i[self.episode_ptr, self.time_ptr+1] = n_u_i
            self.u_t[self.episode_ptr, self.time_ptr+1] = n_u_t
            self.s[self.episode_ptr, self.time_ptr+1] = n_s
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers

            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:

            # update pointers

            self.time_ptr += 1

    def sample(self, batch_size = 64):

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=batch_size)

        ep_lens_of_choices = self.ep_len[choices]

        if self.segment_len is None:

            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

            u_i = self.u_i[choices][:, :max_ep_len_in_batch+1, :]
            u_t = self.u_t[choices][:, :max_ep_len_in_batch+1, :]
            s = self.s[choices][:, :max_ep_len_in_batch+1, :]
            r_i = self.r_i[choices][:, :max_ep_len_in_batch, :]
            r_t = self.r_t[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]

            # convert to tensors on the right device

            u_i = as_tensor_on_device(u_i).view((batch_size, max_ep_len_in_batch) + self.u_i_dim)
            u_t = as_tensor_on_device(u_t).view(batch_size, max_ep_len_in_batch+1, self.u_t_dim)
            s = as_tensor_on_device(s).view(batch_size, max_ep_len_in_batch+1, self.s_dim)
            r_i = as_tensor_on_device(r_i).view((batch_size, max_ep_len_in_batch) + self.r_i_dim)
            r_t = as_tensor_on_device(r_t).view(batch_size, max_ep_len_in_batch, self.r_t_dim)
            r = as_tensor_on_device(r).view(batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(batch_size, max_ep_len_in_batch, 1)

            return RecurrentBatch(u_i, u_t, s, r_i, r_t, r, d, m)

        else:

            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)

            u_i = self.u_i[choices]
            u_t = self.u_t[choices]
            s = self.s[choices]
            r_i = self.r_i[choices]
            r_t = self.r_t[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]

            u_i_seg = np.zeros((batch_size, self.segment_len + 1) + self.u_i_dim)
            u_t_seg = np.zeros((batch_size, self.segment_len, self.u_t_dim))
            s_seg = np.zeros((batch_size, self.segment_len + 1, self.s_dim))
            r_i_seg = np.zeros((batch_size, self.segment_len) + self.r_i_dim)
            r_t_seg = np.zeros((batch_size, self.segment_len, self.r_t_dim))
            r_seg = np.zeros((batch_size, self.segment_len, 1))
            d_seg = np.zeros((batch_size, self.segment_len, 1))
            m_seg = np.zeros((batch_size, self.segment_len, 1))

            for i_ in range(self.batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i_]) * self.segment_len
                u_i_seg[i_] = u_i[i_][start_idx:start_idx + self.segment_len + 1]
                u_t_seg[i_] = u_t[i_][start_idx:start_idx + self.segment_len + 1]
                s_seg[i_] = s[i_][start_idx:start_idx + self.segment_len + 1]
                r_i_seg[i_] = r_i[i_][start_idx:start_idx + self.segment_len]
                r_t_seg[i_] = r_t[i_][start_idx:start_idx + self.segment_len]
                r_seg[i_] = r[i_][start_idx:start_idx + self.segment_len]
                d_seg[i_] = d[i_][start_idx:start_idx + self.segment_len]
                m_seg[i_] = m[i_][start_idx:start_idx + self.segment_len]

            u_i_seg = as_tensor_on_device(u_i_seg)
            u_t_seg = as_tensor_on_device(u_t_seg)
            s_seg = as_tensor_on_device(s_seg)
            r_i_seg = as_tensor_on_device(r_i_seg)
            r_t_seg = as_tensor_on_device(r_t_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)

            return RecurrentBatch(u_i_seg, u_t_seg, s_seg, r_i_seg, r_t_seg, r_seg, d_seg, m_seg)
        
        
        
        
        
import gin

import numpy as np
import torch.optim as optim
from torch import nn
from torchinfo import summary as torch_summary

from basics.abstract_algorithms import RecurrentOffPolicyRLAlgorithm
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.utils import create_target, mean_of_unmasked_elements, polyak_update, save_net, load_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")


def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

class Summarizer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        example_image = torch.zeros((1, 1, args.image_size, args.image_size))

        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 1, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            ConstrainedConv2d(
                in_channels = 16, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1))
            )
        
        example_image = self.image_in(example_image).flatten(1)
        after_cnn_shape = example_image.shape[1]
    
        self.text_in = nn.Sequential(
            nn.Linear(
                in_features = args.text_length * 27,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = 256),
            nn.LeakyReLU()
            )
        
        self.lstm = nn.LSTM(
            input_size = 256 + after_cnn_shape + args.seed_size,
            hidden_size = 256,
            batch_first = True,
            num_layers = 1)
        
        self.image_in.apply(init_weights)
        self.text_in.apply(init_weights)
        self.lstm.apply(init_weights)
        self.to(device)
        
    def forward(self, image, text, seed, hidden = None):
        
        if(type(text) == str):
            text = [text]
        if(type(text) == list):
            text = texts_to_one_hots(text)
                    
        if(type(image) == np.ndarray):
            image = torch.from_numpy(image).float()
        if(len(image.shape) == 2):
            image = image.unsqueeze(0)
        if(len(text.shape) == 1):
            text = text.unsqueeze(0)
        if(len(seed.shape) == 1):
            seed = seed.unsqueeze(0)
            
        
        image = image.reshape((image.shape[0], 1, args.image_size, args.image_size)) / 255
        image = self.image_in(image).flatten(1)
        text = text.to(device).float()
        text = self.text_in(text)
        seed = seed.to(device)
        x = torch.cat([image, text, seed], dim = 1)
        shape = x.shape; x = x.view(shape[0], 1, shape[1])
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(x)
        else:               x, hidden = self.lstm(x, (hidden[0], hidden[1]))
        shape = x.shape; x = x.view(shape[0], shape[-1])
        return(x, hidden)
    

    
    
class Actor(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.image_out = nn.Sequential(
            nn.Linear(
                in_features = 256,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = 256),
            nn.Sigmoid()
            )
        
        self.text_out = nn.Sequential(
            nn.Linear(
                in_features = 256,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = args.text_length * 27)
            )
        
        self.image_out.apply(init_weights)
        self.text_out.apply(init_weights)
        self.to(device)
        
    def forward(self, summary):
        image = self.image_out(summary)
        image = image.reshape((summary.shape[0],16,16))
        text = self.text_out(summary)
        return(image, text)
    
    
    
class Critic(nn.Module):
    
    def __init__(self):
        super().__init__()

        example_image = torch.zeros((1, 1, args.image_size, args.image_size))

        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 1, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            ConstrainedConv2d(
                in_channels = 16, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1))
            )
        
        example_image = self.image_in(example_image).flatten(1)
        after_cnn_shape = example_image.shape[1]
        
        self.value = nn.Sequential(
            nn.Linear(
                in_features = 256 + after_cnn_shape + args.text_length*27,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = 1)
            )
        
        self.image_in.apply(init_weights)
        self.value.apply(init_weights)
        self.to(device)
        
    def forward(self, summary, image, text):
        image = image.reshape((image.shape[0], 1, args.image_size, args.image_size))
        image = self.image_in(image).flatten(1)
        x = torch.cat([summary, image, text], dim=1)
        value = self.value(x)
        return(value)
    
    
if __name__ == "__main__":
    summarizer = Summarizer()
    print("\n\n")
    print(summarizer)
    print()
    print(torch_summary(summarizer, 
                        ((1, args.image_size, args.image_size, 1),
                         (1, args.text_length * 27),
                         (1, args.seed_size))))
    
    actor = Actor()
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, (1, 256)))
    
    critic = Critic()
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, 
                        ((1, 256),
                         (1, args.image_size, args.image_size, 1),
                         (1, args.text_length * 27))))



@gin.configurable(module=__name__)
class Friend(RecurrentOffPolicyRLAlgorithm):

    def __init__(
        self,
        gamma=0.99,
        lr=.001,
        polyak=0.95,       # = (1 - tau)
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2
    ):

        # hyper-parameters

        self.hidden_size = 256
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # trackers

        self.hidden = None
        self.num_Q_updates = 0
        self.mean_Q1_value = 0

        # networks

        self.actor_summarizer = Summarizer().to(get_device())
        self.actor_summarizer_targ = create_target(self.actor_summarizer)

        self.Q1_summarizer = Summarizer().to(get_device())
        self.Q1_summarizer_targ = create_target(self.Q1_summarizer)

        self.Q2_summarizer = Summarizer().to(get_device())
        self.Q2_summarizer_targ = create_target(self.Q2_summarizer)

        self.actor = Actor().to(get_device())
        self.actor_targ = create_target(self.actor)

        self.Q1 = Critic().to(get_device())
        self.Q1_targ = create_target(self.Q1)

        self.Q2 = Critic().to(get_device())
        self.Q2_targ = create_target(self.Q2)

        # optimizers

        self.actor_summarizer_optimizer = optim.Adam(self.actor_summarizer.parameters(), lr=lr)
        self.Q1_summarizer_optimizer = optim.Adam(self.Q1_summarizer.parameters(), lr=lr)
        self.Q2_summarizer_optimizer = optim.Adam(self.Q2_summarizer.parameters(), lr=lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        self.buffer = RecurrentReplayBuffer()        

    def reinitialize_hidden(self) -> None:
        self.hidden = None
        
    def act(self, image, text, seed, hidden = None):
        return(self.image_and_text(image, text, seed, hidden))

    def image_and_text(self, image, text, seed, hidden = None):
        summary, hidden = self.actor_summarizer(image, text, seed, hidden)
        image, text = self.actor(summary)
        image = image.squeeze(0).cpu().detach().numpy()
        text = one_hots_to_text(text.squeeze(0).tolist())
        return(image, text, hidden)

    def update_networks(self, batch_size = 64):
        
        try:
            b = self.buffer.sample(batch_size)
        except:
            return

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o, b.s)
        Q1_summary = self.Q1_summarizer(b.o, b.s)
        Q2_summary = self.Q2_summarizer(b.o, b.s)

        actor_summary_targ = self.actor_summarizer_targ(b.o, b.s)
        Q1_summary_targ = self.Q1_summarizer_targ(b.o, b.s)
        Q2_summary_targ = self.Q2_summarizer_targ(b.o, b.s)

        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary_targ[:, 1:, :]
        Q1_summary_1_T, Q1_summary_2_Tplus1 = Q1_summary[:, :-1, :], Q1_summary_targ[:, 1:, :]
        Q2_summary_1_T, Q2_summary_2_Tplus1 = Q2_summary[:, :-1, :], Q2_summary_targ[:, 1:, :]

        assert actor_summary.shape == (bs, num_bptt+1, self.hidden_dim)

        # compute predictions

        Q1_predictions = self.Q1(Q1_summary_1_T, b.i, b.t)
        Q2_predictions = self.Q2(Q2_summary_1_T, b.i, b.t)

        assert Q1_predictions.shape == (bs, num_bptt, 1)
        assert Q2_predictions.shape == (bs, num_bptt, 1)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(actor_summary_2_Tplus1)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(get_device())
            smoothed_na = torch.clamp(na + noise, -1, 1)

            n_min_Q_targ = torch.min(self.Q1_targ(Q1_summary_2_Tplus1, smoothed_na),
                                     self.Q2_targ(Q2_summary_2_Tplus1, smoothed_na))

            targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q1_loss_elementwise = (Q1_predictions - targets) ** 2
        Q1_loss = mean_of_unmasked_elements(Q1_loss_elementwise, b.m)

        Q2_loss_elementwise = (Q2_predictions - targets) ** 2
        Q2_loss = mean_of_unmasked_elements(Q2_loss_elementwise, b.m)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.Q1_summarizer_optimizer.zero_grad()
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_summarizer_optimizer.step()
        self.Q1_optimizer.step()

        self.Q2_summarizer_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_summarizer_optimizer.step()
        self.Q2_optimizer.step()

        self.num_Q_updates += 1

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            # compute policy loss

            i, t = self.actor(actor_summary_1_T)
            Q1_values = self.Q1(Q1_summary_1_T.detach(), i, t)  # val stands for values
            policy_loss_elementwise = - Q1_values
            policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, b.m)

            self.mean_Q1_value = float(-policy_loss)
            #assert a.shape == (bs, num_bptt, self.action_dim)
            assert Q1_values.shape == (bs, num_bptt, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            self.actor_summarizer_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_summarizer_optimizer.step()
            self.actor_optimizer.step()

            # update target networks

            polyak_update(targ_net=self.actor_summarizer_targ, pred_net=self.actor_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_summarizer_targ, pred_net=self.Q1_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_summarizer_targ, pred_net=self.Q2_summarizer, polyak=self.polyak)

            polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(mean_of_unmasked_elements(Q1_predictions, b.m)),
            '(qfunc) Q2 pred': float(mean_of_unmasked_elements(Q2_predictions, b.m)),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) Q1 value': self.mean_Q1_value
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def copy_networks_from(self, algorithm) -> None:

        self.actor_summarizer.load_state_dict(algorithm.actor_summarizer.state_dict())
        self.actor_summarizer_targ.load_state_dict(algorithm.actor_summarizer_targ.state_dict())

        self.Q1_summarizer.load_state_dict(algorithm.Q1_summarizer.state_dict())
        self.Q1_summarizer_targ.load_state_dict(algorithm.Q1_summarizer_targ.state_dict())

        self.Q2_summarizer.load_state_dict(algorithm.Q2_summarizer.state_dict())
        self.Q2_summarizer_targ.load_state_dict(algorithm.Q2_summarizer_targ.state_dict())

        self.actor.load_state_dict(algorithm.actor.state_dict())
        self.actor_targ.load_state_dict(algorithm.actor_targ.state_dict())

        self.Q1.load_state_dict(algorithm.Q1.state_dict())
        self.Q1_targ.load_state_dict(algorithm.Q1_targ.state_dict())

        self.Q2.load_state_dict(algorithm.Q2.state_dict())
        self.Q2_targ.load_state_dict(algorithm.Q2_targ.state_dict())
        
        
if __name__ == "__main__":
    friend = Friend()
    torch.save(friend, 'friend.pt')