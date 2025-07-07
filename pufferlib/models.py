from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            self.encoder = nn.Linear(np.prod(env.single_observation_space.shape), hidden_size)

        if self.is_multidiscrete:
            action_nvec = env.single_action_space.nvec
            self.decoder = nn.ModuleList([pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, n), std=0.01) for n in action_nvec])
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return torch.relu(self.encoder(observations.float())), None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        value = self.value_head(hidden)
        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoder]
            return actions, value
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = hidden.shape[0]
            return probs, value

        actions = self.decoder(hidden)
        return actions, value

class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent = nn.LSTM(input_size, hidden_size, num_layers)

        for name, param in self.recurrent.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def forward(self, x, state):
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B

        x = x.reshape(B*TT, *space_shape)
        hidden, lookup = self.policy.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size)
        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        hidden, state = self.recurrent(hidden, state)
        hidden = hidden.transpose(0, 1)

        hidden = hidden.reshape(B*TT, self.hidden_size)
        hidden, critic = self.policy.decode_actions(hidden, lookup)
        return hidden, critic, state

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden, None
 
    def decode_actions(self, hidden, lookup):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

import skill_models as sm

class WSA(nn.Module):
    def __init__(self, env, *args, emb_size, device,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1,**kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.env_name = env.env.unwrapped._game.replace("_", "")
        self.device = device
        self.emb_size = emb_size
        self.n_models = 4
        self.print_ww = False
        
        self._load_pretrained_models()

        self._create_adapters()

        self.weight_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(emb_size*2, 1), std=1),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, 1), std=1)

    def _load_pretrained_models(self):
        self.pretrained_models = []
        exp = self.env_name == "breakout"
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.pretrained_models.append(sm.get_state_rep_uns(self.env_name, self.device, exp))
        else: self.n_models -= 1
        self.pretrained_models.append(sm.get_object_keypoints_encoder(self.env_name, self.device, True, exp))
        self.pretrained_models.append(sm.get_object_keypoints_keynet(self.env_name, self.device, True, exp))
        self.pretrained_models.append(sm.get_video_object_segmentation(self.env_name, self.device, True, exp))
        self.state_emb_model = sm.get_autoencoder(self.env_name, self.device, exp)
    
    def _create_adapters(self):
        # 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
        )
        self.c_adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(self.device)
        self.__kpt_enc_adapter.to(self.device)
        self.__kpt_key_adapter.to(self.device)

        self.adapters = nn.ModuleList([])
        # state
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.adapters.append(
                nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(512, self.emb_size), std=0.01),
                    nn.LayerNorm(self.emb_size),
                    nn.ReLU()
                )
            )
        # obj_key_e
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(32*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # obj_key_k
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # vid_seg
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        self.adapters.to(self.device)

        self.state_adapter = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(64*16*16, self.emb_size), std=0.01),
            nn.LayerNorm(self.emb_size),
            nn.ReLU()
        )
        self.state_adapter.to(self.device)

    def _forward_pretrain_model(self, m: sm.Skill, x):
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        with torch.no_grad():
            out = m.input_adapter(x)
            out = m.skill_output(m.skill_model, out)
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
        # print(f"[Timer] {m.name} took {elapsed_time:.3f} ms")
        if m.name in self.c_adapters:
            out = self.c_adapters[m.name](out)

        return out if "state" in m.name else out.view(out.size(0), -1)

    def forward_model(self, obs):
        pt_out = []
        for i, (model, adapter) in enumerate(zip(self.pretrained_models, self.adapters)):
            if i < self.n_models:
                pt_out.append(adapter(self._forward_pretrain_model(model, obs)))
            else:
                break
        pt_embs = torch.stack(pt_out, dim=1)  # Shape: [batch_size, n_models, emb_size]

        # Compute state embedding
        s_out = self.state_adapter(self._forward_pretrain_model(self.state_emb_model, obs))  # Shape: [batch_size, emb_size]

        # Efficient weight computation (avoid manual expand + cat)
        s_out_expanded = s_out.unsqueeze(1).expand(-1, self.n_models, -1)  # [batch_size, n_models, emb_size]
        weight_inputs = torch.cat((s_out_expanded, pt_embs), dim=2)  # [batch_size, n_models, emb_size*2]

        # Compute weights in one go
        ww = self.weight_network(weight_inputs)  # [batch_size, n_models, 1]
        ww = ww / ww.sum(dim=1, keepdim=True).clamp_(min=1e-8)  # Normalize weights safely
        ww = torch.nan_to_num(ww, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs/Infs

        if self.print_ww:
            self._print_ww(ww)

        # Faster weighted summation using batch matrix multiplication (bmm)
        R = (pt_embs*ww).sum(dim=1)  # [batch_size, emb_size]
        
        return R

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.forward_model(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

    # def encode_observations(self, observations):
    #     print(observations.shape)
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     end_event = torch.cuda.Event(enable_timing=True)
    #     start_event.record()
    #     if self.channels_last:
    #         observations = observations.permute(0, 3, 1, 2)
    #     if self.downsample > 1:
    #         observations = observations[:, :, ::self.downsample, ::self.downsample]
    #     hidden = self.forward_model(observations.float())
    #     end_event.record()
    #     torch.cuda.synchronize()  # Waits for events to complete
    #     elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
    #     print(f"[Timer] encode_observations took {elapsed_time:.3f} ms")
    #     return hidden, None

    # def decode_actions(self, flat_hidden, lookup, concat=None):
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     end_event = torch.cuda.Event(enable_timing=True)
    #     start_event.record()
    #     action = self.actor(flat_hidden)
    #     value = self.value_fn(flat_hidden)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
    #     print(f"[Timer] decode_actions took {elapsed_time:.3f} ms")
    #     return action, value

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        for ptmodel in self.pretrained_models:
            ptmodel.eval()
            for param in ptmodel.parameters():
                param.requires_grad = False

        self.state_adapter.skill_model.eval()
        for param in self.state_adapter.skill_model.parameters():
            param.requires_grad = False

        return self

    def _print_ww(self, ww):
        with torch.no_grad():
            mean_per_model = ww.mean(dim=0)  # Shape: [n_models, 1]
            std_per_model = ww.std(dim=0)    # Shape: [n_models, 1]
            print(f"Shape: {ww.shape} - Mean: {mean_per_model.squeeze()} - std: {std_per_model.squeeze()}")

class Ensemble(nn.Module):
    def __init__(self, env, *args, emb_size, device,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1,**kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.env_name = env.env.unwrapped._game.replace("_", "")
        self.device = device
        self.emb_size = emb_size
        self.n_models = 5
        self.print_ww = False

        self._load_pretrained_models()

        self._create_adapters()

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, 1), std=1)

    def _load_pretrained_models(self):
        self.pretrained_models = []
        exp = self.env_name == "breakout"
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.pretrained_models.append(sm.get_state_rep_uns(self.env_name, self.device, exp))
        else: self.n_models -= 1
        self.pretrained_models.append(sm.get_object_keypoints_encoder(self.env_name, self.device, True, exp))
        self.pretrained_models.append(sm.get_object_keypoints_keynet(self.env_name, self.device, True, exp))
        self.pretrained_models.append(sm.get_video_object_segmentation(self.env_name, self.device, True, exp))
        self.pretrained_models.append(sm.get_autoencoder(self.env_name, self.device, exp))

    def _create_adapters(self):
        # 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
        )
        self.c_adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(self.device)
        self.__kpt_enc_adapter.to(self.device)
        self.__kpt_key_adapter.to(self.device)

        self.adapters = nn.ModuleList([])
        # state
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.adapters.append(
                nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(512, self.emb_size), std=0.01),
                    nn.LayerNorm(self.emb_size),
                    nn.ReLU()
                )
            )
        # obj_key_e
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(32*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # obj_key_k
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # vid_seg
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        #autoencoder
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(64*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        self.adapters.to(self.device)

    def _forward_pretrain_model(self, m: sm.Skill, x):
        with torch.no_grad():
            out = m.input_adapter(x)
            out = m.skill_output(m.skill_model, out)
        if m.name in self.c_adapters:
            out = self.c_adapters[m.name](out)

        return out if "state" in m.name else out.view(out.size(0), -1)

    def forward_model(self, obs):
        pt_out = []
        for i, (model, adapter) in enumerate(zip(self.pretrained_models, self.adapters)):
            if i < self.n_models:
                pt_out.append(adapter(self._forward_pretrain_model(model, obs)))
            else:
                break
        pt_embs = torch.stack(pt_out, dim=1)  # Shape: [batch_size, n_models, emb_size]
        R = torch.mean(pt_embs, dim=1)  # [batch_size, emb_size]
        
        return R

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.forward_model(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        for ptmodel in self.pretrained_models:
            ptmodel.eval()
            for param in ptmodel.parameters():
                param.requires_grad = False

        self.state_adapter.skill_model.eval()
        for param in self.state_adapter.skill_model.parameters():
            param.requires_grad = False

        return self

    def _print_ww(self, ww):
        with torch.no_grad():
            mean_per_model = ww.mean(dim=0)  # Shape: [n_models, 1]
            std_per_model = ww.std(dim=0)    # Shape: [n_models, 1]
            print(f"Shape: {ww.shape} - Mean: {mean_per_model.squeeze()} - std: {std_per_model.squeeze()}")

class WSAFT(nn.Module):
    def __init__(self, env, *args, emb_size, device,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1,**kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.env_name = env.env.unwrapped._game.replace("_", "")
        self.device = device
        self.emb_size = emb_size
        self.n_models = 4
        self.print_ww = False

        self._load_pretrained_models()

        self._create_adapters()

        self.weight_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(emb_size*2, 1), std=1),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, 1), std=1)

    def _load_pretrained_models(self):
        self.pretrained_models = []
        exp = self.env_name == "breakout"
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.pretrained_models.append(sm.get_state_rep_uns(self.env_name, self.device, exp, dont_load=True))
        else: self.n_models -= 1
        self.pretrained_models.append(sm.get_object_keypoints_encoder(self.env_name, self.device, True, exp, dont_load=True))
        self.pretrained_models.append(sm.get_object_keypoints_keynet(self.env_name, self.device, True, exp, dont_load=True))
        self.pretrained_models.append(sm.get_video_object_segmentation(self.env_name, self.device, True, exp, dont_load=True))
        self.state_emb_model = sm.get_autoencoder(self.env_name, self.device, exp, dont_load=True)

    def _create_adapters(self):
        # 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
        )
        self.c_adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(self.device)
        self.__kpt_enc_adapter.to(self.device)
        self.__kpt_key_adapter.to(self.device)

        self.adapters = nn.ModuleList([])
        # state
        if self.env_name not in ("beamrider", "enduro", "roadrunner"):
            self.adapters.append(
                nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(512, self.emb_size), std=0.01),
                    nn.LayerNorm(self.emb_size),
                    nn.ReLU()
                )
            )
        # obj_key_e
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(32*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # obj_key_k
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        # vid_seg
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        self.adapters.to(self.device)

        self.state_adapter = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(64*16*16, self.emb_size), std=0.01),
            nn.LayerNorm(self.emb_size),
            nn.ReLU()
        )
        self.state_adapter.to(self.device)

    def _forward_pretrain_model(self, m: sm.Skill, x):
        out = m.input_adapter(x)
        out = m.skill_output(m.skill_model, out)
        if m.name in self.c_adapters:
            out = self.c_adapters[m.name](out)

        return out if "state" in m.name else out.view(out.size(0), -1)

    def forward_model(self, obs):
        pt_out = []
        for i, (model, adapter) in enumerate(zip(self.pretrained_models, self.adapters)):
            if i < self.n_models:
                pt_out.append(adapter(self._forward_pretrain_model(model, obs)))
            else:
                break
        pt_embs = torch.stack(pt_out, dim=1)  # Shape: [batch_size, n_models, emb_size]

        # Compute state embedding
        s_out = self.state_adapter(self._forward_pretrain_model(self.state_emb_model, obs))  # Shape: [batch_size, emb_size]

        # Efficient weight computation (avoid manual expand + cat)
        s_out_expanded = s_out.unsqueeze(1).expand(-1, self.n_models, -1)  # [batch_size, n_models, emb_size]
        weight_inputs = torch.cat((s_out_expanded, pt_embs), dim=2)  # [batch_size, n_models, emb_size*2]

        # Compute weights in one go
        ww = self.weight_network(weight_inputs)  # [batch_size, n_models, 1]
        ww = ww / ww.sum(dim=1, keepdim=True).clamp_(min=1e-8)  # Normalize weights safely
        ww = torch.nan_to_num(ww, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs/Infs

        if self.print_ww:
            self._print_ww(ww)

        # Faster weighted summation using batch matrix multiplication (bmm)
        R = (pt_embs*ww).sum(dim=1)  # [batch_size, emb_size]

        return R

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.forward_model(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

    def _print_ww(self, ww):
        with torch.no_grad():
            mean_per_model = ww.mean(dim=0)  # Shape: [n_models, 1]
            std_per_model = ww.std(dim=0)    # Shape: [n_models, 1]
            print(f"Shape: {ww.shape} - Mean: {mean_per_model.squeeze()} - std: {std_per_model.squeeze()}")

class WSASingle(nn.Module):
    def __init__(self, env, *args, emb_size, device,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1,**kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.env_name = env.env.unwrapped._game.replace("_", "")
        self.device = device
        self.emb_size = emb_size
        self.n_models = 1
        self.print_ww = False

        self._load_pretrained_models()
        self._create_adapters()

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(emb_size, 1), std=1)

    def _load_pretrained_models(self):
        self.pretrained_models = []
        self.pretrained_models.append(sm.get_swin(self.device))
    
    def _create_adapters(self):
        self.adapters = nn.ModuleList([])
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(1000, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                nn.ReLU()
            )
        )
        self.adapters.to(self.device)

    def _forward_pretrain_model(self, m: sm.Skill, x):
        with torch.no_grad():
            if m.input_adapter:
                x = m.input_adapter(x)
            out = m.skill_output(m.skill_model, x)
        return out.view(out.size(0), -1)

    def forward_model(self, obs):
        R = self.adapters[0](self._forward_pretrain_model(self.pretrained_models[0], obs)) # [batch_size, emb_size]        
        return R

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.forward_model(observations.float()), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        for ptmodel in self.pretrained_models:
            ptmodel.eval()
            for param in ptmodel.parameters():
                param.requires_grad = False

        self.state_adapter.skill_model.eval()
        for param in self.state_adapter.skill_model.parameters():
            param.requires_grad = False

        return self
    
    def _print_ww(self, ww):
        with torch.no_grad():
            mean_per_model = ww.mean(dim=0)  # Shape: [n_models, 1]
            std_per_model = ww.std(dim=0)    # Shape: [n_models, 1]
            print(f"Shape: {ww.shape} - Mean: {mean_per_model.squeeze()} - std: {std_per_model.squeeze()}")