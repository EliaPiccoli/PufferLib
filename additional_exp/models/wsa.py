import os
import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn 
from torch.distributions.normal import Normal

import models.skill_models as sm

import pufferlib
import pufferlib.pytorch

class WSA_Robot(nn.Module):
    def __init__(self, env, *args, emb_size, device,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=True, downsample=1,**kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        # self.env_name = env.env.unwrapped._game.replace("_", "")
        self.env_name = "maniskill_env"
        self.device = device
        self.emb_size = emb_size
        self.n_models = 2
        
        self._load_pretrained_models()

        self._create_adapters()

        self.weight_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(emb_size*2, 1), std=1),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(emb_size, 256), std=0.01),
            pufferlib.pytorch.layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)))

        )
        
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(env.single_action_space.shape)) * -0.5)
        
        self.value_fn = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(emb_size, 256), std=0.01),
            pufferlib.pytorch.layer_init(nn.Linear(256, 1), std=1)
        )

    def _load_pretrained_models(self):
        self.pretrained_models = []
        exp = self.env_name == "breakout"
        # if self.env_name not in ("beamrider", "enduro", "roadrunner"):
        #     self.pretrained_models.append(sm.get_state_rep_uns(self.env_name, self.device, exp))
        # else: self.n_models -= 1
        self.pretrained_models.append(sm.get_swin(self.device))
        self.pretrained_models.append(sm.get_resnet(self.device))
        # self.pretrained_models.append(sm.get_video_object_segmentation(self.env_name, self.device, True, exp))
        self.state_emb_model = sm.get_clip_model(self.device)
    
    def _create_adapters(self):
        # 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            # nn.ReLU(),
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            # nn.ReLU(),
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            # nn.ReLU()
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
        # # state
        # if self.env_name not in ("beamrider", "enduro", "roadrunner"):
        #     self.adapters.append(
        #         nn.Sequential(
        #             pufferlib.pytorch.layer_init(nn.Linear(1000, self.emb_size), std=0.01),
        #             nn.LayerNorm(self.emb_size),
        #             nn.ReLU()
        #         )
        #     )
        # swin
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(1000, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                # nn.ReLU()
            )
        )
        # resnet
        self.adapters.append(
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(8192, self.emb_size), std=0.01),
                nn.LayerNorm(self.emb_size),
                # nn.ReLU()
            )
        )
        # not used
        # self.adapters.append(
        #     nn.Sequential(
        #         pufferlib.pytorch.layer_init(nn.Linear(16*16*16, self.emb_size), std=0.01),
        #         nn.LayerNorm(self.emb_size),
        #         nn.ReLU()
        #     )
        # )
        self.adapters.to(self.device)

        # clip?
        self.state_adapter = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(512, self.emb_size), std=0.01),
            nn.LayerNorm(self.emb_size),
            nn.ReLU()
        )
        self.state_adapter.to(self.device)

    def _forward_pretrain_model(self, m: sm.Skill, x):
        # print(f"{m.name} input shape: {x.shape}, dtype: {x.dtype}")
        with torch.no_grad():
            if m.input_adapter:
                x = m.input_adapter(x)
            out = m.skill_output(m.skill_model, x)
        if m.name in self.c_adapters:
            out = self.c_adapters[m.name](out)

        # print(f"output shape: {out.shape}, dtype: {out.dtype}")
        return out if "state" in m.name else out.view(out.size(0), -1)

    def forward_model(self, obs):
        pt_out = []
        # print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
        for i, (model, adapter) in enumerate(zip(self.pretrained_models, self.adapters)):
            if i < self.n_models:
                pt_out.append(adapter(self._forward_pretrain_model(model, obs)))
            else:
                break
        pt_embs = torch.stack(pt_out, dim=1)  # Shape: [batch_size, n_models, emb_size]
        # print(f"pt_embs values: {pt_embs}")
        # Print the magnitude and mean value of pt_embs for each batch
        # batch_magnitudes = torch.norm(pt_embs, dim=-1)
        # batch_means = pt_embs.mean(dim=-1)
        # for i, (magnitude, mean) in enumerate(zip(batch_magnitudes, batch_means)):
        #     print(f"Batch {i}: Magnitude = {magnitude}, Mean = {mean}")
        # Compute state embedding
        s_out = self.state_adapter(self._forward_pretrain_model(self.state_emb_model, obs))  # Shape: [batch_size, emb_size]

        # Efficient weight computation (avoid manual expand + cat)
        s_out_expanded = s_out.unsqueeze(1).expand(-1, self.n_models, -1)  # [batch_size, n_models, emb_size]
        weight_inputs = torch.cat((s_out_expanded, pt_embs), dim=2)  # [batch_size, n_models, emb_size*2]

        # Compute weights in one go
        ww = self.weight_network(weight_inputs)  # [batch_size, n_models, 1]
        # print(f"ww values: {ww}")
        ww = ww / ww.sum(dim=1, keepdim=True).clamp_(min=1e-8)  # Normalize weights safely
        ww = torch.nan_to_num(ww, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs/Infs
        
        

        # Faster weighted summation using batch matrix multiplication (bmm)
        R = (pt_embs*ww).sum(dim=1)  # [batch_size, emb_size]
        
        return R

    def forward(self, observations):
        # observations = observations["rgb"].permute(0, 3, 1, 2)  # Make channel first
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
    
    def get_action(self, observations, **kwargs):
        return self(observations)[0]
    
    def get_action_and_value(self, observations, action=None):
        hidden, lookup = self.encode_observations(observations)
        action_mean = self.actor(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.value_fn(hidden)
    
    def get_value(self, observations):
        return self(observations)[1]
    
    def encode_observations(self, observations):
        observations = observations["rgb"]
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
            ptmodel.skill_model.eval()
            for param in ptmodel.skill_model.parameters():
                param.requires_grad = False
        self.state_adapter.eval()
        for param in self.state_adapter.parameters():
            param.requires_grad = False
        
        return self