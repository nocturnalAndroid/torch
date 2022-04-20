import random
import time
from dataclasses import dataclass
from math import sqrt, atan2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.patches import FancyArrow
from matplotlib.text import Text
from numpy import pi
from tqdm.auto import tqdm

from QLearner import QLearner


@dataclass
class AsteroidsGame:
    board_size = 100
    angular_v = pi / 36
    acc = 0.2
    drag_coefficient = 0.05
    food = []
    obstacle = []
    position = np.array([50.0, 50.0])
    velocity = np.array([0.0, 0.0])
    orientation = 0
    angle_of_view = pi/2.5
    view_ray_count = 15
    view_ray_width = angle_of_view / view_ray_count
    view_range = board_size / 2
    last_state = None
    last_action = None
    possible_actions = [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
    ]
    t = 0
    cum_times = {}
    play_back = True
    reward_history = []

    def __post_init__(self):
        self.food = np.random.random((5, 2))*self.board_size
        self.obstacle = np.random.random((7, 2))*self.board_size
        self.t = time.time()


    def update(self,
               _ ,
               learner: QLearner,
               arrow: FancyArrow,
               food_scatter: PathCollection,
               view_scatter: PathCollection,
               label: Text,
               loss_graph,
               reward_graph,
               ax
               ):
        self.add_time('be')

        food_view, obstacle_view, reward = self.step(learner)
        if self.play_back > 3:
            for i in tqdm(list(range(100))):
                reward = 0
                while abs(reward) < 0.01:
                    food_view, obstacle_view, reward = self.step(learner)
            self.play_back = 1
        elif abs(reward) > 0.01:
            self.play_back += 1



        arrow.set_data(x=self.position[0]-np.cos(self.orientation),
                       y=self.position[1]-np.sin(self.orientation),
                       dx=np.cos(self.orientation),
                       dy=np.sin(self.orientation),
                       width=1
                       )

        food_scatter.set_offsets(np.concatenate((self.food, self.obstacle), 0))
        food_scatter.set_color((['green']*len(self.food)) + (['red']*len(self.obstacle)))
        ct = {k: int(v) for k, v in self.cum_times.items()}
        label.set_text(f'{ct}, v:{np.linalg.norm(self.velocity)} a:{self.possible_actions[self.last_action][1]} \n{learner.action_persistence=} {learner.action_explore=} {learner.action_educated_guess=}')
        view_coords, view_colors = get_view_scatter_data(self, food_view, obstacle_view)
        view_scatter.set_offsets(view_coords)
        view_scatter.set_facecolor(view_colors)
        view_scatter.set_edgecolor(None)
        loss_graph.set_data(range(len(learner.loss_history)), learner.loss_history)
        smoooth_reward_history = np.convolve(self.reward_history, np.ones(500) / 500, 'valid')
        reward_graph.set_data(range(len(smoooth_reward_history)), smoooth_reward_history)
        ax[1].relim()
        ax[1].autoscale_view()
        ax[2].relim()
        ax[2].autoscale_view()

        self.add_time('draw')

    def step(self, learner):
        reward = self.check_collisions(self.food) - 2 * self.check_collisions(self.obstacle) - 0.001
        self.reward_history.append(reward)
        self.add_time('crw')
        food_view = self.get_view(self.food)
        obstacle_view = self.get_view(self.obstacle)
        self.add_time('cv')
        state = food_view + obstacle_view
        self.add_time('cs')

        if self.last_state is not None and self.last_action is not None:
            learner.reward(self.last_state, self.last_action, state, reward)
        self.add_time('rw')

        action_ind = learner.get_action(state)
        self.add_time('ga')

        self.last_action = action_ind
        self.last_state = state
        action = self.possible_actions[action_ind]
        self.position += self.velocity
        self.position %= self.board_size
        self.orientation += (action[2] - action[0]) * self.angular_v
        self.orientation %= 2 * np.pi
        self.velocity += (np.cos(self.orientation) * self.acc * action[1], np.sin(self.orientation) * self.acc * action[1])
        norm_v = np.linalg.norm(self.velocity)
        if norm_v > 0.5:
            self.velocity -= self.drag_coefficient * norm_v * self.velocity
        self.add_time('ue')

        return food_view, obstacle_view, reward

    def check_collisions(self, items):
        collisions = 0
        for i, food in enumerate(items):
            if np.linalg.norm(self.position - food) < 1:
                collisions += 1
                items[i] = np.random.random(2) * self.board_size
        return collisions

    def add_time(self, bucket):
        self.cum_times[bucket] = self.cum_times.get(bucket, 0) + time.time() - self.t
        self.t = time.time()

    def get_view(self, items):
        view = [0.0] * self.view_ray_count
        half_board_size = 0.5 * self.board_size
        half_board_size2 = half_board_size * half_board_size
        for item in items:
            dl = item - self.position
            dl = ((dl + half_board_size) % self.board_size) - half_board_size
            dist2 = np.dot(dl, dl)
            self.add_time("d")
            if dist2 < half_board_size2:
                dang = angle_difference(atan2(dl[1], dl[0]), self.orientation)
                self.add_time('a')
                if abs(dang) < 0.5 * self.angle_of_view:
                    ray_ind = int((dang + 0.5 * self.angle_of_view) // self.view_ray_width)
                    view[ray_ind] = max(view[ray_ind], 1 / (0.01 + sqrt(dist2)))
                    self.add_time('r')
        return view

def main():
    game = AsteroidsGame()
    learner = QLearner(game.view_ray_count * 2, len(game.possible_actions))
    fig, ax = plt.subplots(3 ,1 ,figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1, 1]})
    ax[0].axis([0, game.board_size, 0, game.board_size])
    arrow = ax[0].arrow(0, 0, 0, 0)
    food_scatter = ax[0].scatter([], [], 50, c='green')
    view_scatter = ax[0].scatter([], [], 30, edgecolors=None, alpha=0.2)
    label = ax[0].text(0, 100, '')
    loss_graph, = ax[1].plot([],[])
    reward_graph, = ax[2].plot([],[])
    a = FuncAnimation(fig, game.update, None, fargs=(learner, arrow, food_scatter, view_scatter, label, loss_graph, reward_graph, ax), interval=1
                      )
    plt.show()



def get_rotate_2d_vector(v, theta):
    return np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]),
        v
    )




def get_view_scatter_data(game: AsteroidsGame, food_view, obstacle_view, interval=2):
    coords = []
    colors = []
    for ray in range(game.view_ray_count):
        ray_ang = game.orientation + (((ray + 0.5) / game.view_ray_count) - 0.5) * game.angle_of_view
        sin_ray_ang = np.sin(ray_ang)
        cos_ray_ang = np.cos(ray_ang)
        for d in range(interval, int(game.view_range), interval):
            coord = game.position + (d * cos_ray_ang, d * sin_ray_ang)
            coord %= game.board_size
            coords.append(coord)
            dinv = 1 / (0.01 + d)
            if dinv < food_view[ray] and dinv < obstacle_view[ray]:
                colors.append('brown')
            elif dinv < food_view[ray]:
                colors.append('green')
            elif dinv < obstacle_view[ray]:
                colors.append('red')
            else:
                colors.append('gray')

    return coords, colors

def angle_difference(a1, a2):
    return ((a1 - a2) + pi) % (2 * pi) - pi


if __name__ == '__main__':
    main()