import random

import numpy as np
import torch
import torch.nn as nn
from torch import float32, int64


class QLearner:

    def __init__(self, state_size: int, possible_actions_count: int):
        torch.set_num_interop_threads(10)
        self.possible_actions_count = possible_actions_count
        self.gamma = 0.95
        self.q_model = nn.Sequential(
            nn.Linear(state_size, state_size*2),
            nn.ReLU(),
            nn.Linear(state_size*2, state_size),
            nn.ReLU(),
            nn.Linear(state_size, possible_actions_count),
            nn.ReLU(),
        )
        self.optimizer = torch.optim.RMSprop(self.q_model.parameters())
        self.last_action = torch.Tensor([2]).type(int64)
        self.action_persistence = 0.5
        self.action_explore = 0.7
        self.action_educated_guess = 0.99
        self.memory = [[] for _ in range(possible_actions_count)]
        self.max_memory_size = 10000
        self.batch_size = 100
        self.memory_index = [0]*possible_actions_count
        self.loss_history = []

    def get_action(self, state):
        with torch.no_grad():
            if random.random() < self.action_persistence:
                self.action_persistence -= 0.001*(self.action_persistence**4)
                action = self.last_action
            elif random.random() < self.action_explore:
                self.action_explore -= 0.0001*(self.action_explore**4)
                action = torch.Tensor(random.choices(range(self.possible_actions_count))).type(int64)
            elif random.random() < self.action_educated_guess:
                self.action_educated_guess -= 0.00001*(self.action_educated_guess**3)
                try:
                    action = torch.Tensor(random.choices(range(self.possible_actions_count), self.get_Qs(state))).type(int64)
                except:
                    action = torch.Tensor(random.choices(range(self.possible_actions_count))).type(int64)
            else:
                action = torch.argmax(self.get_Qs(state))
            self.last_action = action
            return action

    def get_Qs(self, state):
        return self.q_model(torch.tensor(state).float())

    def reward(self, state, action, next_state, reward):
        if self.memory_index[action] >= len(self.memory[action]):
            self.memory[action].append((state, action, next_state, reward))
        else:
            self.memory[action][self.memory_index[action]] = (state, action, next_state, reward)
        self.memory_index[action] = (self.memory_index[action] + 1) % self.max_memory_size

        if all(len(memory) > self.batch_size for memory in self.memory):
            samples = []
            for action_memory in self.memory:
                samples.extend(random.sample(action_memory, self.batch_size))
            states, actions, next_states, rewards = tuple([*zip(*samples)])
            self.optimizer.zero_grad()
            Qs = self.get_Qs(states)
            Q = Qs.gather(1, torch.Tensor((actions,)).T.long())

            with torch.no_grad():
                next_Q = self.get_Qs(next_states).max(1).values

            loss = ((torch.Tensor(rewards) + self.gamma * next_Q - Q.flatten())**2).mean()
            self.loss_history.append(float(loss))
            if len(self.loss_history) > 100000:
                self.loss_history = self.loss_history[10000:]
            loss.backward()

            self.optimizer.step()
