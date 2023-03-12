import torch
from typing import Tuple
from copy import deepcopy
import wandb
from torch import Tensor
import torch.nn.functional as F

from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.agents.AgentBase import AgentBase
# from elegantrl.agents.net import Actor
from elegantrl.agents.net import CriticTwin

from elegantrl.agents.diffusion import Diffusion


class AgentTD3(AgentBase):
    """Twin Delayed DDPG algorithm.
    Addressing Function Approximation Error in Actor-Critic Methods. 2018.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, max_action, beta_schedule='linear',
                 n_timesteps: int = 5,
                 gpu_id: int = 0,
                 args: Config = Config()):

        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        self.act_class = getattr(self, 'act_class', Diffusion)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                         beta_schedule=beta_schedule, n_timesteps=n_timesteps, gpu_id=gpu_id, args=args)

        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        # self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        # self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.update_freq = getattr(args, 'update_freq', 2)  # delay update frequency

        # self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)
    def Q_cond_fn(self,state,a):
        """
        return the graident of the classifier outputing y wrt x.
        formally expressed as d_log(classifier(x, t)) / dx
        """
        with torch.enable_grad():
            a_in = a.detach().requires_grad_(True)
            # logits = self.cri_target(state,a_in)#todo 两个输出?
            logits = self.cri_target.get_q1_q2(state,a_in)#todo 两个输出?
            # log_probs = F.log_softmax(logits, dim=-1)
            # selected = log_probs[range(len(logits)), y.view(-1)]
            selected = logits
            grad = torch.autograd.grad(selected[0].sum(), a_in)[0] * self.gradient_scale #todo 这里两个Q 网络，用哪个？
            # grad = torch.autograd.grad(selected[0].sum(), a_in)[0] * self.gradient_scale #todo 这里两个Q 网络，用哪个？
            # grad2 = torch.autograd.grad(selected[1].sum(), a_in)[0] * self.gradient_scale #todo 这里两个Q 网络，用哪个？
            # return 0.5*(grad+grad2)
            return grad
    def get_action(self, state,cond_fn=None, guidance_kwargs=None):
        if cond_fn==None:
            cond_fn=self.Q_cond_fn
        batch_size=state.shape[0]
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = state.reshape(state.shape[0], -1).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.act.sample(state_rpt,cond_fn)
            # time = torch.zeros( (action.shape[0]), device=self.device).long()
            # q_value = self.cri_target.get_q_min(state_rpt, action,time).flatten()
            # q_value_ = self.cri_target.get_q_min(state_rpt, action).flatten()
            q_value = self.cri_target.get_q_min(state_rpt, action).flatten().reshape(-1,50)
            # idx_ = torch.multinomial(F.softmax(q_value_), 1)# todo  ???
            idx = torch.multinomial(F.softmax(q_value,dim=1), 1).squeeze(1)# todo  ???
            # haha= action[idx_].detach().cpu()
        # return action[idx].detach().cpu()
        return action.reshape(batch_size, -1)[torch.arange(batch_size),idx].unsqueeze(1)
        # return action[idx].cpu().data.numpy().flatten()
    def get_diffusion_obj(self, buffer, batch_size):
        states, actions, rewards, undones, next_ss = buffer.sample(batch_size*100)  # next_ss: next states
        bc_loss = self.act.loss(actions, states)
        # new_action = self.actor(state)
        #
        # q1_new_action, q2_new_action = self.critic(state, new_action)
        # if np.random.uniform() > 0.5:
        #     q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        # else:
        #     q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        # actor_loss = bc_loss + self.eta * q_loss
        actor_loss = bc_loss
        return actor_loss

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            wandb.log({"loss/critic":obj_critic})

            if update_c % self.update_freq == 0:  # delay update
                # action_pg = self.act(state)  # policy gradient
                # obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                obj_actor = self.get_diffusion_obj(buffer, self.batch_size)
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, obj_actor)
                wandb.log({"loss/actor":obj_actor})

                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / (update_times/self.update_freq)

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            # todo ！
            # next_as = self.act_target.get_action_noise(next_ss, self.policy_noise_std)  # next actions
            next_as = self.get_action(next_ss)  # next actions
            next_qs = self.cri_target.get_q_min(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_as = self.act_target.get_action_noise(next_ss, self.policy_noise_std)
            next_qs = self.cri_target.get_q_min(next_ss, next_as)
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = self.cri.get_q1_q2(states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states
