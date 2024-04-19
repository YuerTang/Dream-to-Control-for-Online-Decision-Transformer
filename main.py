import torch
import numpy as np
import gym
import datetime
import os
from omegaconf import OmegaConf, open_dict, DictConfig
from visual_control.dreamer import Dreamer  as Agent

from typing import Optional, Tuple, List, Dict, Union
from visual_control.buffer import Replay_Buffer
from decision_transformer.models.decision_transformer import DecisionTransformer
from trainer import SequenceTrainer
import utils  # Ensure utils are properly imported or defined

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
from time import sleep
import logging
import d4rl
from dataclasses import dataclass, field
import hydra
from termcolor import colored
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
import wandb
from envs.pomdp_dmc.wrappers import POMDPWrapper  # DMC, take env as input
from visual_control.utils import Timer
import pickle as pkl
from lib.utils import seed_torch, make_env, make_eval_env, mask_env_state, VideoRecorder, makedirs

MAX_EPISODE_LEN = 1000

log = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


@dataclass
class Config:
    # General settings
    user: str = field(default_factory=lambda: os.getenv('USER', 'default_user'))
    resume: str = ""
    device: int = 0
    video: bool = False
    num_eval_episodes: int = 5
    dm_control: bool = True
    env: str = "cheetah_run"
    wandb: bool = False
    from_pixels: bool = True
    pixels_width: int = 64
    action_repeat: int = 2
    batch_size: int = 50
    batch_length: int = 50
    train_steps: int = 100
    eval_every: int = 10000
    seed: int = 0
    formal_save: bool = False
    alpha: float = -1.0  # Default value for alpha, adjust as needed based on your specific requirements
    init_alpha: float = 1.0  # Default value for alpha, adjust as needed based on your specific requirements

    # Online Decision Transformer specific settings
    max_episode_length: int = 1000
    state_dim: int = 17
    action_dim: int = 6
    train_every: int = 1000
    model_lr: float = 1e-4
    value_lr: float = 1e-4
    actor_lr: float = 1e-4
    K: int = 20
    embed_dim: int = 256
    n_layer: int = 3
    n_head: int = 4
    activation_function: str = 'relu'
    dropout: float = 0.1

    # Creating POMDP (if applicable)
    flicker: float = 0.0
    noise: float = 0.0
    missing: float = 0.0

    # Learning rates and related parameters
    lr: float = 0.0003
    tau: float = 0.005
    alpha: float = -1  # Auto-tuning if negative
    init_alpha: float = 1.0
    num_steps: int = 1000001
    updates_per_step: int = 1
    start_steps: int = 10000

    # Model specific settings
    num_units: int = 300
    deter_size: int = 200
    stoch_size: int = 30
    rssm_std: float = 0.1
    ts: bool = False  # Thompson sampling

    # Training configuration
    init_temperature: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    warmup_steps: int = 10000
    max_pretrain_iters: int = 1000
    num_updates_per_pretrain_iter: int = 500
    max_online_iters: int = 1500
    online_rtg: int = 7200
    num_online_rollouts: int = 1
    replay_size: int = 1000
    num_updates_per_online_iter: int = 300
    eval_interval: int = 10

    # Cluster settings
    ngpus: int = 1
    partt: str = "learnlab"  # or 'devlab'
    mem_gb: int = 64
    cpus_per_task: int = 10
    gpus_per_node: int = 1
    max_num_timeout: int = 100000
    timeout_min: int = 4319

    # Additional model specifics
    log_every: int = 1000
    eval_noise: float = 0.0
    clip_rewards: str = 'none'
    dense_act: str = 'elu'
    cnn_act: str = 'relu'
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 1.0
    pcont: bool = False  # whether predict done (predicting done is meaningless for DMC)
    pcont_scale: float = 10.0
    grad_clip: float = 100.0
    dataset_balance: bool = False
    discount: float = 0.99
    disclam: float = 0.95
    horizon: int = 15  # imaginary rollout length
    action_dist: str = 'tanh_normal'
    action_init_std: float = 5.0
    expl: str = 'additive_gaussian'
    expl_amount: float = 0.3
    expl_decay: float = 0.0
    expl_min: float = 0.0

    # Ablations.
    update_horizon: Optional[int] = None  # policy value after this horizon are not updated
    single_step_q: bool = False
    tf_init: bool = False

    #random
    stochastic_policy: bool = False  # Set to True if your model should use a stochastic policy
    nmode: int = 1
    replay_buffer_size: int = 100
        
    hydra_run_dir: str = "./outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}"
    hydra_sweep_dir: str = "/checkpoint/${user}/RL-LVM/dreamer/${now:%Y.%m.%d}/${now:%H.%M.%S}"
    hydra_subdir: str = "${hydra.job.override_dirname}"


def get_env_name(cfg):
    env_name = f"{cfg.env}" \
        + (f"_flick{cfg.flicker}" if cfg.flicker > 0 else "") \
        + (f"_noise{cfg.noise}" if cfg.noise > 0 else "") \
        + (f"_miss{cfg.missing}" if cfg.missing > 0 else "")
    return env_name

class Workspace:
    def __init__(self, cfg, variant):

        ### Dreamer Setup
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        self.cfg = cfg
        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")
        print(f"workspace: {self.work_dir}")

        # Set the device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != 'cpu' else "cpu")
        self.global_steps = 0
        self.i_episode = 0

        # Initialize the training environment
        train_env, self.act_dim, _, self.obs_dim = make_env(cfg)
        with open_dict(self.cfg):
            self.cfg.action_range = [float(train_env.action_space.low.min()), 
                                     float(train_env.action_space.high.max())]
            self.cfg.auto_tune = self.cfg.alpha < 0  # negative means use entropy auto tuning
            if self.cfg.deter_size in [50, 100]:
                self.cfg.stoch_size = 100 
        self.save_dir = self.work_dir

        # Initialize Dreamer with both Dreamer and ODT configurations
        self.agent = Agent(cfg, train_env.action_space).to(cfg.device)
        self.replay_buffer = Replay_Buffer(action_space=train_env.action_space, balance=False)
        self.records = {"reward_train": {}, "reward_test": {}} 

        self.video_dir = os.path.join(self.work_dir, 'video')
        if self.cfg.video:
            self.video_recorder = VideoRecorder(self.work_dir if cfg.video else None)
            makedirs(self.video_dir)



        ### Online Decision Transformer

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer_odt = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs


            
            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                )
            

        self.replay_buffer_odt.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate_odt(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1
        
       

    def evaluate_odt(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward
    
    
    
    ##### Run

    def run(self):

        utils.set_seed_everywhere(self.cfg.seed)

        import d4rl

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                import d4rl

                env = gym.make(env_name)
                env.seed(seed)
                if hasattr(env.env, "wrapped_env"):
                    env.env.wrapped_env.seed(seed)
                elif hasattr(env.env, "seed"):
                    env.env.seed(seed)
                else:
                    pass
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")
                return env

            return make_env_fn
        
    
        # Dreamer Initial 
        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)

        log.info(f"Work directory is {self.work_dir}")
        print(f"Work directory is {self.work_dir}")
        log.info("Running with configuration:\n" + str(self.cfg)) 

        train_env, self.act_dim, _, self.obs_dim = make_env(self.cfg)
        eval_env = make_eval_env(train_env, self.cfg.seed + 1)
        train_env = POMDPWrapper(train_env, flicker_prob=self.cfg.flicker, noise_sigma=self.cfg.noise, sensor_missing_prob=self.cfg.missing)
        eval_env = POMDPWrapper(eval_env, flicker_prob=self.cfg.flicker, noise_sigma=self.cfg.noise, sensor_missing_prob=self.cfg.missing)
        log.info(f"obs_dim={self.obs_dim} act_dim={self.act_dim} self.max_trajectory_len={train_env._max_episode_steps}")

        job_type = "sweep" if ("/checkpoint/" in self.work_dir) else "local"
        with open_dict(self.cfg):
            self.cfg.job_type = job_type
        cfg = self.cfg
        
        wandb_name = get_env_name(cfg)
        use_pomdp = False if wandb_name == f"{cfg.env}" else True
        wandb_name += ("" if cfg.seed == 0 else f"_seed{cfg.seed}")

        wandb_name += (f"_deter{cfg.deter_size}" if cfg.deter_size != 200 else "")
        wandb_name += (f"_stoch{cfg.stoch_size}" if cfg.stoch_size != 30 else "")
        wandb_name += (f"_nunit{cfg.num_units}" if cfg.num_units != 300 else "")
        wandb_name += (f"_rssmstd{cfg.rssm_std:.1e}" if cfg.rssm_std != 0.1 else "")

        print(f"Wandb name: {wandb_name}")
        if cfg.wandb:
            wandb.init(project="Dreamer_POMDP" if use_pomdp else "Dreamer", entity="generative-modeling", group=self.cfg.env,
                 name=wandb_name, dir=self.work_dir, resume=False, config=cfg, save_code=True, job_type=job_type)
            log.info(f"Wandb initialized for {wandb_name}")
        
        seed_torch(self.cfg.seed)

        

        
        # Online Decision Transformer Initial (Pretrain)
        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        print("Pretrained Finished!")
        

        # Main Loop for Dreamer
        obs = train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer = Timer()
        self.last_frames = 0


        episode_reward = 0
        episode_steps = 0

        self.evaluate_dreamer(eval_env)  # Initial evaluation

        #self.evaluate(eval_env)


        self.save()

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )

            print("\n\n\n*** Online Finetuning ***")

            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            eval_fns = [
                create_vec_eval_episodes_fn(
                    vec_env=eval_envs,
                    eval_rtg=self.variant["eval_rtg"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                    use_mean=True,
                    reward_scale=self.reward_scale,
                )
            ]
            writer = (
                SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
            )


            while self.global_frames < self.cfg.num_steps: # and self.online_iter < self.variant["max_online_iters"]:

                

                if self.global_frames < self.cfg.start_steps:
                    action = train_env.action_space.sample()
                else:
                    action, agent_state = self.agent.get_action(obs, agent_state, training=True)

                
                obs, reward, done, info = train_env.step(action)
                self.replay_buffer.add(obs, action, reward, done, info)
                self.global_steps += 1
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    self.i_episode += 1
                    self.log(episode_reward, episode_steps, prefix='Train')
                    if self.cfg.wandb:
                        wandb.log({"reward_train": episode_reward}, step=self.global_frames)
                    self.records["reward_train"][self.i_episode] = episode_reward
                    # Reset
                    episode_reward = 0
                    episode_steps = 0
                    obs = train_env.reset()
                    self.replay_buffer.start_episode(obs)
                    agent_state = None

                 

                if self.global_frames >= self.cfg.start_steps and self.global_frames % self.cfg.train_every == 0:

                    
                    outputs = {}
                    augment_outputs = self._augment_trajectories(
                        online_envs,
                        self.variant["online_rtg"],
                        n=self.variant["num_online_rollouts"],
                    )
                    outputs.update(augment_outputs)
                    

                    dataloader = create_dataloader(
                        trajectories=self.replay_buffer_odt.trajectories,
                        num_iters=self.variant["num_updates_per_online_iter"],
                        batch_size=self.variant["batch_size"],
                        max_len=self.variant["K"],
                        state_dim=self.state_dim,
                        act_dim=self.act_dim,
                        state_mean=self.state_mean,
                        state_std=self.state_std,
                        reward_scale=self.reward_scale,
                        action_range=self.action_range,
                    )

                    # finetuning
                    is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
                    if (self.online_iter + 1) % self.variant[
                        "eval_interval"
                    ] == 0 or is_last_iter:
                        evaluation = True
                    else:
                        evaluation = False
                    
                    print("Loss Function:")
                    print(loss_fn)

                    train_outputs = trainer.train_iteration(
                        loss_fn=loss_fn,
                        dataloader=dataloader,
                    )
                    outputs.update(train_outputs)

                    if evaluation:
                        eval_outputs, eval_reward = self.evaluate_odt(eval_fns)
                        outputs.update(eval_outputs)

                    outputs["time/total"] = time.time() - self.start_time

                    # log the metrics
                    self.logger.log_metrics(
                        outputs,
                        iter_num=self.pretrain_iter + self.online_iter,
                        total_transitions_sampled=self.total_transitions_sampled,
                        writer=writer,
                    )

                    self._save_model(
                        path_prefix=self.logger.log_path,
                        is_pretrain_model=False,
                    )

                    self.online_iter += 1

                    self.agent.train()
                    dataloader = self.replay_buffer.get_iterator(self.cfg.train_steps, self.cfg.batch_size, self.cfg.batch_length)



                    for train_step, data in enumerate(dataloader):
                        log_video = self.cfg.video and \
                                self.global_frames % self.cfg.eval_every == 0 and train_step == 0
                        self.agent.update(data, log_video=log_video,
                                video_path=os.path.join(self.video_dir, f'error_{self.global_frames_str}.gif'))

                
                    if self.global_frames % self.cfg.log_every == 0:
                        if self.cfg.wandb:
                            wandb_dict = {
                                "loss/image": self.agent.metrics['image_loss'].result(),
                                "loss/reward": self.agent.metrics['reward_loss'].result(),
                                "loss/model": self.agent.metrics['model_loss'].result(),
                                "loss/actor": self.agent.metrics['actor_loss'].result(),
                                
                                "ent/prior": self.agent.metrics['prior_ent'].result(),
                                "ent/posterior": self.agent.metrics['post_ent'].result(),
                                "KL_div": self.agent.metrics['div'].result(),
                            }
                            if cfg.dreamer in [0,]:
                                wandb_dict.update({
                                    "loss/value": self.agent.metrics['value_loss'].result(),
                                    "ent/action": self.agent.metrics['action_ent'].result(),
                                    "ent/paz_logstd": self.agent.metrics["action_logstd"].result(),
                                    "value_func": self.agent.metrics['value_func'].result(),
                                    "log_pi": self.agent.metrics['log_pi'].result(),
                                })

                            else:
                                if cfg.dreamer == 1:
                                    wandb_dict.update({
                                        "ent/action": self.agent.metrics['action_ent'].result()
                                        })
                                wandb_dict.update({
                                    "loss/critic": self.agent.metrics['critic_loss'].result(),
                                    "loss/alpha": self.agent.metrics['alpha_loss'].result(),
                                    "ent/paz_logstd": self.agent.metrics["action_logstd"].result(),
                                    "alpha": self.agent.metrics['alpha'].result(),
                                    "log_pi": self.agent.metrics['log_pi'].result(),
                                    "q_func": self.agent.metrics['q_func'].result(),
                                })

                            if cfg.dreamer in [2,]:
                                wandb_dict.update({"ent/feature_std": self.agent.metrics["feature_std"].result()})
                            
                            wandb.log(wandb_dict, step=self.global_frames)

                        metrics = [(k, float(v.result())) for k, v in self.agent.metrics.items()]
                        [m.reset_states() for m in self.agent.metrics.values()]
                        print(colored(f'[{self.global_frames}]', 'yellow'), ' / '.join(f'{k} {v:.1f}' for k, v in metrics))

                #online_envs.close() 

                if self.global_frames % self.cfg.eval_every == 0:
                    avg_reward = self.evaluate_dreamer(eval_env)
                    if self.cfg.wandb:
                        wandb.log({"reward_test": avg_reward}, step=self.global_frames)
                        if self.cfg.video:
                            wandb.log({
                                f"{self.global_frames_str}":
                                                wandb.Video(os.path.join(self.video_recorder.save_dir,
                                                        f'eval_{self.global_frames_str}.gif')),
                                f"error_{self.global_frames}":
                                                wandb.Video(os.path.join(self.video_dir,
                                                        f'error_{self.global_frames_str}.gif'))
                                })
                    self.save()

        train_env.close()
        eval_env.close()
        if self.cfg.wandb:
            wandb.finish()


    def evaluate_dreamer(self, eval_env):
        lengths_ls = []
        episode_reward_ls = []
        self.agent.eval()
        for epi_idx in range(self.cfg.num_eval_episodes):
            if self.cfg.video:
                self.video_recorder.init(enabled= (epi_idx == 0))

            obs = eval_env.reset()
            episode_reward = 0
            length = 0
            agent_state = None
            done = False
            while not done:
                action, agent_state = self.agent.get_action(obs, agent_state, training=False)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward
                length += 1

                if self.cfg.video:
                    self.video_recorder.record(eval_env)

            episode_reward_ls.append(episode_reward)
            lengths_ls.append(length)

            if self.cfg.video:
                self.video_recorder.save(f'eval_{self.global_frames_str}.gif')

        avg_reward = float(np.mean(episode_reward_ls))
        avg_length = float(np.mean(lengths_ls))
        self.log(avg_reward, avg_length, prefix='Test')
        self.records["reward_test"][self.i_episode] = avg_reward
        return avg_reward

    def save(self, tag="latest"):
        for dir in {self.save_dir, self.work_dir}:
            self.agent.save(os.path.join(dir, f"{tag}.ckpt"))

            path = os.path.join(dir, f"records_{tag}.json")
            with open(path, "wb") as f:
                pkl.dump(self.records, f)
            print(f"Saved at {path}")

    def log(self, avg_return: float, avg_length: float, prefix: str):
        colored_prefix = colored(prefix, 'yellow' if prefix == 'Train' else 'green')
        elapsed_time = self.timer.split()
        total_time = datetime.timedelta(seconds=int(self.timer.total()))
        fps = (self.global_frames - self.last_frames) / elapsed_time
        self.last_frames = self.global_frames
        print(f'{colored_prefix:<14} | Frame: {self.global_frames} | Episode: {self.i_episode} | '
              f'Reward: {avg_return:.2f} | Length: {avg_length:.1f} | FPS: {fps:.2f} | Time: {total_time}')

    @property
    def global_frames(self):
        return self.global_steps * self.cfg.action_repeat

    @property
    def global_frames_str(self):
        length = len(str(self.cfg.num_steps))
        return f'frame{self.global_frames:0{length}d}'


log = logging.getLogger(__name__)

def main():
    cfg = Config()
    print(f"Configuration: {cfg}")

    variant = {
        "env": "halfcheetah-medium-v2",
        "learning_rate": 0.0001,
        "embed_dim": 512,
        "n_layer": 4,
        "n_head": 4,
        "activation_function": "relu",
        "dropout": 0.1,
        "K": 20,
        "ordering": 0,
        "eval_rtg": 3600,
        "init_temperature": 0.1,
        "batch_size": 8, #changed from 256
        "num_eval_episodes": 10,
        "eval_context_length": 5,
        "max_pretrain_iters": 1,
        "weight_decay": 5e-4,
        "warmup_steps": 10000, #changed from 10000
        "num_updates_per_pretrain_iter": 500, #changed from 5000
        "max_online_iters": 1500,
        "online_rtg": 7200,
        "num_online_rollouts": 1,
        "replay_size": 1000,
        "num_updates_per_online_iter": 300,
        "eval_interval": 10,
        "device": "cuda",
        "log_to_tb": True,
        "exp_name": "default",
        "save_dir": "./exp"
    }

    workspace = Workspace(cfg, variant)

    
    try:
        workspace.run()
    except Exception as e:
        log.critical(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()


'''
fname = os.path.join(os.getcwd(), "latest.pkl")
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f: 
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    try:
        workspace.run()
    except Exception as e:
        log.critical(e, exc_info=True)
'''
