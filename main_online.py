import torch
import numpy as np
import gym
import os
from omegaconf import OmegaConf, open_dict
from visual_control.dreamer_odt import DreamerODT  # Adjust import path as needed
from visual_control.buffer import ReplayBuffer
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.trainer import SequenceTrainer
import utils  # Ensure utils are properly imported or defined

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import d4rl

from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger

from envs.pomdp_dmc.wrappers import POMDPWrapper  # DMC, take env as input
from visual_control.buffer import ReplayBuffer
from visual_control.utils import Timer
from lib.utils import seed_torch, make_env, make_eval_env, mask_env_state, VideoRecorder, makedirs

MAX_EPISODE_LEN = 1000

log = logging.getLogger(__name__)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the environment
        self.env = gym.make(cfg.env)
        self.action_space = self.env.action_space
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        # Initialize DreamerODT with both Dreamer and ODT configurations
        self.agent = DreamerODT(cfg, self.action_space).to(self.device)

        # Replay Buffer for storing experiences
        self.replay_buffer = ReplayBuffer(action_space=self.env.action_space, size=cfg.replay_buffer_size)

        # Set up ODT model and trainer
        self.odt_model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            max_length=cfg.K,
            hidden_size=cfg.embed_dim,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            activation_function=cfg.activation_function,
            n_positions=1024,
            resid_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout
        ).to(self.device)

        self.odt_optimizer = torch.optim.Adam(self.odt_model.parameters(), lr=cfg.learning_rate)
        self.odt_trainer = SequenceTrainer(self.odt_model, self.odt_optimizer, device=self.device)

        # Logging and evaluation setup
        self.eval_env = gym.make(cfg.env)  # For simplicity, using a single env for evaluation

    ##### Dreamer
    
    def run(self):
        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)

        log.info(f"Work directory is {self.work_dir}")
        print(f"Work directory is {self.work_dir}")
        log.info("Running with configuration:\n" + str(self.cfg))  # OmegaConf.to_yaml(self.cfg, resolve=True)

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

        # for model learning part
        wandb_name += (f"_deter{cfg.deter_size}" if cfg.deter_size != 200 else "")
        wandb_name += (f"_stoch{cfg.stoch_size}" if cfg.stoch_size != 30 else "")
        wandb_name += (f"_nunit{cfg.num_units}" if cfg.num_units != 300 else "")
        wandb_name += (f"_rssmstd{cfg.rssm_std:.1e}" if cfg.rssm_std != 0.1 else "")
        
        # for policy opt part
        if cfg.dreamer == 1:
            wandb_name += "_sac"
        if cfg.dreamer == 2:
            wandb_name += f"_smac"

        if cfg.dreamer >= 1:
            wandb_name += ("_autot" if cfg.auto_tune else f"_alpha{cfg.alpha}")
            wandb_name += (f"_saclr{cfg.lr:.1e}" if cfg.lr != 3e-4 else "")
            if cfg.dreamer in [2,]:
                wandb_name += (f"_pazent{cfg.pazent}" if cfg.pazent > 0. else "")
                wandb_name += (f"_aggfirst" if cfg.aggfirst else "")
                wandb_name += f"_qagg{cfg.qagg}"
                wandb_name += f"_{cfg.estimate}"
                if cfg.estimate in ["naive", "nmlmc",]:
                    wandb_name += f"{cfg.num_p}"
        
        print(f"Wandb name: {wandb_name}")
        if cfg.wandb:
            wandb.init(project="Dreamer_POMDP" if use_pomdp else "Dreamer", entity="generative-modeling", group=self.cfg.env,
                 name=wandb_name, dir=self.work_dir, resume=False, config=cfg, save_code=True, job_type=job_type)
            log.info(f"Wandb initialized for {wandb_name}")
        
        seed_torch(self.cfg.seed)

        # Main Loop
        obs = train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer = Timer()
        self.last_frames = 0
        
        episode_reward = 0
        episode_steps = 0

        self.evaluate(eval_env)  # Initial evaluation
        self.save()
        while self.global_frames < self.cfg.num_steps:
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

            # Training
            if self.global_frames >= self.cfg.start_steps and self.global_frames % self.cfg.train_every == 0:
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

            if self.global_frames % self.cfg.eval_every == 0:
                avg_reward = self.evaluate(eval_env)
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

    def evaluate(self, eval_env):
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

    ##### Online Decision Transformer

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

        self.replay_buffer.add_new_trajs(trajs)
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
            eval_outputs, eval_reward = self.evaluate(eval_fns)
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

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

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
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
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

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
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

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

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


    def run(self):

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

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()

        for epoch in range(self.cfg.num_epochs):
            # Training loop for Dreamer + ODT
            observations, actions, rewards, next_observations, dones = self.collect_experiences()
            self.agent.update(observations, actions, rewards, next_observations, dones)

            # ODT training step
            dataloader = utils.create_dataloader(self.replay_buffer, batch_size=self.cfg.batch_size, K=self.cfg.K)
            self.odt_trainer.train(dataloader)

            if epoch % self.cfg.eval_every == 0:
                eval_reward = self.evaluate()
                print(f"Eval Reward at epoch {epoch}: {eval_reward}")

    def collect_experiences(self):
        # Implement logic to interact with the environment, collect experiences,
        # and store them in the replay buffer. Return a batch of experiences for training.
        # Placeholder for simplicity
        return observations, actions, rewards, next_observations, dones

    def evaluate(self):
        # Implement evaluation logic that leverages the integrated model (Dreamer + ODT)
        # to interact with the environment and compute evaluation metrics.
        # Placeholder for simplicity
        return eval_reward

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")  # Ensure you have a config file or provide configurations directly
    utils.set_seed_everywhere(cfg.seed)
    workspace = Workspace(cfg)
    workspace.run()
