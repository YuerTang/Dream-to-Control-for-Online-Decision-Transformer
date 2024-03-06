import imageio
import collections
import torch
import json
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from torch import Tensor, nn, optim
from torch.nn import functional as F
from functools import partial
import math
from termcolor import colored
from torch.distributions import kl_divergence

from visual_control.utils import Timer, AttrDict, freeze, AverageMeter
from visual_control.network import ConvDecoder, ConvEncoder, ActionDecoder, DenseDecoder, RSSM

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.trainer import SequenceTrainer as ODTTrainer
from decision_transformer.models.decision_transformer import DecisionTransformer


act_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU
}

class DreamerODT(nn.Module):
    def __init__(self, cfg, action_space: spaces.Box):
        super().__init__()
        self.action_space = action_space
        self.actdim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.cfg = cfg
        self.metrics = collections.defaultdict(AverageMeter)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Define device early
        self.trajectories = []  # Placeholder for trajectories; populate as needed

        
        ########## world model
        cnn_act = act_dict[cfg.cnn_act]
        act = act_dict[cfg.dense_act]
        self.encoder = ConvEncoder(depth=cfg.cnn_depth, act=cnn_act)
        self.dynamics = RSSM(self.action_space, stoch=cfg.stoch_size,
                             deter=cfg.deter_size, hidden=cfg.deter_size, cfg=cfg)

        feat_size = cfg.stoch_size + cfg.deter_size
        self.decoder = ConvDecoder(feature_dim=feat_size, depth=cfg.cnn_depth, act=cnn_act)
        self.reward = DenseDecoder(input_dim=feat_size, shape=(), layers=2, units=cfg.num_units, act=act)

        if self.cfg.pcont:
            self.pcont = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=cfg.num_units, dist='binary', act=act)
        

         ########## Decision Transformer
        self.decision_transformer = DecisionTransformer(
            state_dim=feat_size,  # Including action dim in state representation if necessary
            act_dim=self.actdim,
            max_length=cfg.K,  # Sequence length; ensure this is defined in your cfg
            hidden_size=cfg.embed_dim,  # Embedding dimension
            n_layer=cfg.n_layer,  # Number of transformer layers
            n_head=cfg.n_head,  # Number of attention heads
            activation_function=cfg.activation_function,  # Activation function
            dropout=cfg.dropout,  # Dropout rate
            stochastic_policy=cfg.stochastic_policy,  # Whether to use a stochastic policy
            action_range=[float(self.action_space.low.min()), float(self.action_space.high.max())],  # Action range for normalization
            init_temperature=cfg.init_temperature,  # Initial temperature for stochastic policy
            #target_entropy=self.target_entropy  # Target entropy for policy regularization
        ).to(self.device)
        
        ########## policy optimization
        feat_size = cfg.stoch_size + cfg.deter_size
        
        self.model_modules = nn.ModuleList([self.encoder, self.decoder, self.dynamics, self.reward])
        if self.cfg.pcont:
            self.model_modules.append(self.pcont)
        
        # Optimizer includes parameters from both model components and decision transformer
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward.parameters()) +
            list(self.decision_transformer.parameters()),
            lr=cfg.model_lr,
            weight_decay=cfg.weight_decay
        )
        
        self.alpha = cfg.alpha
        if self.cfg.auto_tune:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape)).item()
            self.target_entropy += math.log(cfg.nmode)
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.log_alpha.data.fill_(math.log(cfg.init_alpha))  # Initialize alpha to be high.
            self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg.lr)

        if self.cfg.tf_init:
            for m in self.decision_transformer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight.data)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.fill_(0.0)

    def update(self, data: Dict[str, Tensor], log_video: bool, video_path: Path=None):
        """
        Corresponds to Dreamer._train.
        Update the model and policy/value. Log metrics and video.
        """
        data = self.preprocess_batch(data)
        
        # model loss
        with torch.autocast(device_type="cuda"):
            embed = self.encoder(data['image'])  # (B, T, 1024)
            post, prior = self.dynamics.observe(embed, data['action'])
            
            feat = self.dynamics.get_feat(post)  # (B, T, 230)
            image_pred = self.decoder(feat)  # dist on (B, T, 3, H, W), std=1.0
            reward_pred = self.reward(feat)  # dist on (B, T)

        likes = AttrDict()
        # mean over batch and time, sum over pixel
        likes.image = image_pred.log_prob(data['image']).mean(dim=[0, 1])
        likes.reward = reward_pred.log_prob(data['reward']).mean(dim=[0, 1])
        if self.cfg.pcont:
            pcont_pred = self.pcont(feat)
            pcont_target = self.cfg.discount * data['discount']
            likes.pcont = torch.mean(pcont_pred.log_prob(pcont_target), dim=[0, 1])
            likes.pcont *= self.cfg.pcont_scale

        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = kl_divergence(post_dist, prior_dist).mean(dim=[0, 1])
        div = torch.clamp(div, min=self.cfg.free_nats)  # in case of prior = posterior => kl = 0
        model_loss = self.cfg.kl_scale * div - sum(likes.values())

        ###
        feature = torch.cat([post['stoch'], post['deter'].detach()], -1)
        feature = feature.detach() # necessary
    
        odt_trainer = ODTTrainer(
            model=self.odt_model,  # ODT model integrated into Dreamer
            optimizer=self.odt_optimizer,  # Optimizer for ODT
            log_temperature_optimizer=self.odt_log_temperature_optimizer,  
            device=self.device
        )
        sequences = self.prepare_sequences_for_odt(data)

        # Convert sequences to tensors and send to the same device as the model
        sequences_tensors = {k: torch.tensor(v).to(self.device) for k, v in sequences.items()}

        # Update ODT model
        with freeze(nn.ModuleList([self.model_modules])):
            odt_loss, action_loss, state_loss, rtg_loss = self.odt_loss_function(
                action_preds=sequences['action_preds'],
                action_targets=sequences['actions'],
                state_preds=sequences.get('state_preds'),
                state_targets=sequences.get('states'),
                rtg_preds=sequences.get('rtg_preds'),
                rtg_targets=sequences.get('rtg')
            )
            
            self.odt_optimizer.zero_grad()
            odt_loss.backward()
            self.odt_optimizer.step()


        self.scalar_summaries(
            data, feature, prior_dist, post_dist, likes, div,
            model_loss)
        
        if log_video:
            self.image_summaries(data, embed, image_pred, video_path)

    @torch.no_grad()
    def scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, odt_loss, 
          model_norm, odt_norm):
        self.metrics['model_grad_norm'].update_state(model_norm)
        self.metrics['odt_grad_norm'].update_state(odt_norm)
        self.metrics['prior_ent'].update_state(prior_dist.entropy().mean())
        self.metrics['post_ent'].update_state(post_dist.entropy().mean())

        for name, logprob in likes.items():
            self.metrics[name + '_loss'].update_state(-logprob)
        self.metrics['div'].update_state(div)
        self.metrics['model_loss'].update_state(model_loss)
        self.metrics['odt_loss'].update_state(odt_loss)

    @torch.no_grad()
    def image_summaries(self, data, embed, image_pred, video_path):
        # Take the first 6 sequences in the batch
        B, T, C, H, W = image_pred.mean.size()
        B = 6
        truth = data['image'][:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])  # get posterior
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        feat = self.dynamics.get_feat(prior)

        openl = self.decoder(feat).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], dim=1)
        error = (model - truth + 1) / 2
        # (B, T, 3, 3H, W)
        openl = torch.cat([truth, model, error], dim=3)
        # (T, 3H, B * W, 3)
        openl = openl.permute(1, 3, 0, 4, 2).reshape(T, 3 * H, B * W, C).cpu().numpy()
        openl = (openl * 255.).astype(np.uint8)
        imageio.mimsave(video_path, openl, fps=30)
        print(f"Video saved at {video_path}")

    def preprocess_batch(self, data: Dict[str, np.ndarray]):
        data = {k: torch.as_tensor(v, device=self.cfg.device, dtype=torch.float) for k, v in data.items()}
        data['image'] = data['image'] / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=torch.tanh)[self.cfg.clip_rewards]
        data['reward'] = clip_rewards(data['reward'])
        return data

    def preprocess_observation(self, obs: Dict[str, np.ndarray]):
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float)
        obs = obs / 255.0 - 0.5
        return obs

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray], state: Optional[Tensor] = None, training: bool = True) \
            -> Tuple[np.ndarray, Optional[Tensor]]:
        """
        Corresponds to Dreamer.__call__, but without training.
        Args:
            obs: obs['image'] shape (C, H, W), uint8
            state: None, or Tensor
        Returns:
            action: (D)
            state: None, or Tensor
        """
        # Add T and B dimension for a single action
        obs = obs[None, None, ...]

        action, state = self.policy(obs, state, training)  #self.action_space.sample(),None
        action = action.squeeze(axis=0)
        return action, state

    def policy(self, obs: Tensor, state: Tensor, training: bool) -> Tensor:
        """
        Args:
            obs: (B, C, H, W)
            state: (B, D)
        Returns:
            action: (B, D)
            state: (B, D)
        """
       # If no state yet initialise tensors otherwise take input state
        if state is None:
            latent = self.dynamics.initial(len(obs))
            action = torch.zeros((len(obs), self.actdim), dtype=torch.float32).to(self.cfg.device)
        else:
            latent, action = state
        embed = self.encoder(self.preprocess_observation(obs))
        embed = embed.squeeze(0)
        latent, _ = self.dynamics.obs_step(latent, action, embed)  
        feat = torch.cat([latent['stoch'], latent['deter']], -1)

        # If training sample random actions if not pick most likely action 
        if training:
            action = self.actor(feat).sample()
        else:
            # this is dirty: it should be the mode
            # the original repo samples 100 times and takes the argmax of log_prob
            action = torch.tanh(self.actor(feat).base_dist.base_dist.mean)  # base_dist is gaussian
        action = self.exploration(action, training)
        state = (latent, action)
        action = action.cpu().detach().numpy()
        action = np.array(action, dtype=np.float32)
        return action, state

    def exploration(self, action: Tensor, training: bool) -> Tensor:
        """
        Args:
            action: (B, D)
        Returns:
            action: (B, D)
        """
        if training:
            amount = self.cfg.expl_amount
            if self.cfg.expl_min:
                amount = max(self.cfg.expl_min, amount)
            self.metrics['expl_amount'].update_state(amount)
        elif self.cfg.eval_noise:
            amount = self.cfg.eval_noise
        else:
            return action

        if self.cfg.expl == 'additive_gaussian':
            return torch.clamp(torch.normal(action, amount), 
                            self.action_space.low.min(), self.action_space.high.max())
        if self.cfg.expl == 'completely_random':
            return torch.rand(action.shape, -1, 1)
        if self.cfg.expl == 'epsilon_greedy':
            indices = torch.distributions.Categorical(0 * action).sample()
            return torch.where(
                        torch.rand(action.shape[:1], 0, 1) < amount,
                        torch.one_hot(indices, action.shape[-1], dtype=self.float),
                        action)
        raise NotImplementedError(self.cfg.expl)

    def load(self, path: Union[str, Path], device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = Path(path)
        with path.open('wb') as f:
            self.load_state_dict(torch.load(f, map_location=device))

    # Change to state dict if we just want to save the weights
    def save(self, path: Union[str, Path]): 
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('wb') as f:
            torch.save(self.state_dict(), f)

    def train(self):
        self.model_modules.train()
        self.critic.train()
        self.actor.train()
    
    def eval(self):
        self.model_modules.eval()
        self.critic.eval()
        self.actor.eval()
    
    ##################################################################################
        
    def prepare_sequences_for_odt(self, batch_size, sequence_length):
        """
        Prepares sequences of states, actions, and rewards for ODT training.

        Args:
        - batch_size (int): Number of sequences per batch.
        - sequence_length (int): Length of each sequence.

        Returns:
        - A dictionary containing batches of 'states', 'actions', 'rewards', 'next_states', 'dones',
        and 'rtg' (return-to-go), ready for ODT training.
        """
        assert len(self.trajectories) >= batch_size, "Not enough trajectories."

        # Randomly select trajectories
        selected_trajs = np.random.choice(self.trajectories, size=batch_size, replace=False)
        
        sequences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'rtg': []  # Return-to-go
        }
        
        for traj in selected_trajs:
            start_idx = np.random.randint(0, len(traj['observations']) - sequence_length + 1)
            end_idx = start_idx + sequence_length
            
            # Extract sequences
            states = traj['observations'][start_idx:end_idx]
            actions = traj['actions'][start_idx:end_idx]
            rewards = traj['rewards'][start_idx:end_idx]
            next_states = traj['next_observations'][start_idx:end_idx] if 'next_observations' in traj else traj['observations'][start_idx+1:end_idx+1]
            dones = traj['dones'][start_idx:end_idx] if 'dones' in traj else np.zeros_like(rewards)
            
            # Calculate return-to-go for each timestep in the sequence
            rtg = np.flip(np.cumsum(np.flip(rewards, axis=0)), axis=0)
            
            # Append to lists
            sequences['states'].append(states)
            sequences['actions'].append(actions)
            sequences['rewards'].append(rewards)
            sequences['next_states'].append(next_states)
            sequences['dones'].append(dones)
            sequences['rtg'].append(rtg)
        
        # Convert lists to tensors
        for key in sequences.keys():
            sequences[key] = torch.tensor(np.array(sequences[key]), dtype=torch.float32)
            
        return sequences
    


    def odt_loss_function(self, action_preds, action_targets, state_preds=None, state_targets=None, rtg_preds=None, rtg_targets=None):
        """
        Computes the loss for the Online Decision Transformer.

        Parameters:
        - action_preds: Predicted actions or action distributions.
        - action_targets: Actual actions taken.
        - state_preds (optional): Predicted next states.
        - state_targets (optional): Actual next states.
        - rtg_preds (optional): Predicted return-to-go.
        - rtg_targets (optional): Actual return-to-go.

        Returns:
        - Total loss, action loss, state loss (if applicable), rtg loss (if applicable).
        """

        # Assuming action_preds are distributions and action_targets are actions
        action_loss = -action_preds.log_prob(action_targets).mean()

        total_loss = action_loss

        if state_preds is not None and state_targets is not None:
            state_loss = F.mse_loss(state_preds, state_targets)
            total_loss += state_loss
        
        if rtg_preds is not None and rtg_targets is not None:
            rtg_loss = F.mse_loss(rtg_preds, rtg_targets)
            total_loss += rtg_loss

        return total_loss, action_loss, state_loss if 'state_loss' in locals() else None, rtg_loss if 'rtg_loss' in locals() else None