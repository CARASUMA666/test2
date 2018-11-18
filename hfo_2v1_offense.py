#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import itertools
import random
import threading
import tensorflow as tf
from time import ctime, sleep
from MAPDQN2 import P_DQN
#from agents.hdqn import DeepQNetwork
#from utils.reward_utils import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


class Locker(object):
    def __init__(self):
        # self.lock = threading.Lock()
        self._visited = False
        self.count = 0

    @property
    def visited(self):
        return self._visited

    @visited.setter
    def visited(self, value):
        self._visited = value | self._visited

    def step(self):
        self.count += 1
        if self.count > 5:
            self._visited = False
            self.count = 0

    # def release():
    #     self.lock.release()


#TODO reward for two agents
def low_level_reward_function(state_1, state_0, ball, status):
    if not ball.visited:
        I_kick = max(0, (state_1[12] - state_0[12]) / 2)
        if state_1[12] == 1:
            ball.visited = True
    else:
        I_kick = 0
    if status == hfo.GOAL:
        I_goal = 1
    else:
        I_goal = 0

    def ball_goal_dist(s):  # distance not proximity
        theta1 = (-1 if s[13] < 0 else 1) * math.acos(s[14])
        theta2 = (-1 if s[51] < 0 else 1) * math.acos(s[52])
        return math.sqrt((1 - s[53]) ** 2 + (1 - s[15]) ** 2 - 2 * (1 - s[53]) * (1 - s[15]) * math.cos(
            max(theta1, theta2) - min(theta1, theta2))) / math.sqrt((1 - s[53]) ** 2 + (1 - s[15]) ** 2)

    print("| TowardsError: {}\n| Kickable: {}\n| ball_goal_delta: {}{}\n| bool GOAL:{}".format(
        (state_1[53] - state_0[53]), I_kick, 3 * (ball_goal_dist(state_0) - ball_goal_dist(state_1)),
        "| INCLUDED |" if ball.visited else "| ** IGNORED ** |", 5 * I_goal
    ))
    return ((state_1[53] - state_0[53]) + I_kick + int(ball.visited) * 3 * max(0, (
                ball_goal_dist(state_0) - ball_goal_dist(state_1))) + 5 * I_goal)


def player(mark):
    print('--I am player', mark, ctime())
    # Create the HFO Environment
    hfo_env = hfo.HFOEnvironment()
    hfo_env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
                            'C:/Users/Administrator/HFO/bin/teams/base/config/formations-dt',
                            args.port, 'localhost', 'base_right', False)

    total_step = 0
    ep_rewards = []
    ep_steps = []
    ep_goals = []
    for episode in itertools.count():
        status = hfo.IN_GAME
        episode_step = 0
        #FIXME 0061
        isBall = Locker
        total_reward = 0.
        while status == hfo.IN_GAME:
            total_step += 1
            episode_step += 1

            # Get the vector of state features for the current state
            st = np.hstack(hfo_env.getState())
            action,c_action = ma_pdqn.act(state = st,index = mark)
            if action == 0:
                hfo_env.act(hfo.DASH,c_action[0],c_action[1])
            elif action == 1:
                hfo_env.act(hfo.TURN,c_action)
            elif action== 2:
                hfo_env.act(hfo.TACKLE,c_action)
            elif action ==3:
                hfo_env.act(hfo.KICK,c_action[0],c_action[1])
            else:
                print('I am not acting',mark)

            # Advance the environment and get the game status
            #player = env.playerOnBall()
            status = hfo_env.step()
            st_ = np.array(hfo_env.getState())
            r_t = low_level_reward_function(st_, st, isBall, status)
            #FIXME st[0:58]
            if mark == 0:
                ma_pdqn.storeTransition1(st, st_, action, c_action, r_t, st[0:58], st_[0:58])
            else:
                ma_pdqn.storeTransition2(st, st_, action, c_action, r_t, st[0:58], st_[0:58])
            total_reward += r_t
            ma_pdqn.train()
            #FIXME 0065
            ma_pdqn.episode_done(index = mark)

        ep_steps.append(episode_step)
        #FIXME rewards.append
        ep_rewards.append(total_reward)
        if status == hfo.GOAL:
            ep_goals.append(1)
        else:
            ep_goals.append(0)
        ep_steps = ep_steps[-100:]
        ep_rewards = ep_rewards[-100:]
        ep_goals = ep_goals[-100:]

        if (episode + 1) % print_interval == 0 and mark == 0:
            print("================================================")
            print("--Agent:", mark)
            print("--Episode: ", episode)
            print("----Avg_steps: ", sum(ep_steps[-100:]) / 100.0)
            print("----Avg_reward: ", sum(ep_rewards[-100:]) / 100.0)
            print("----Goal_rate: ", sum(ep_goals[-100:]) / 100.0)
            print("------------------------------------------------")

        # Check the outcome of the episode

        # end_status = hfo_env.statusToString(status)
        # print("Episode {0:n} ended with {1:s}".format(episode, end_status))

        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == '__main__':
# --------------------------------- Param parser ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000,
                        help="Server port")
    parser.add_argument('--agent-num', type=int, default=2)
    parser.add_argument('--train-interval', type=int, default=10)
    parser.add_argument('--bef-train', type=int, default=20000)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--is-save-log', type=bool, default=False)
    args = parser.parse_args()

    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    IS_SAVE_LOG = args.is_save_log

    agent_num = args.agent_num
    train_interval = args.train_interval
    bef_train = args.bef_train
    print_interval = args.print_interval
    #FIXME 0055 number of global features, pre_train,is_test
    ma_pdqn = P_DQN(n_features = 77,
        n_global_features = 58,
        action_dim1 = 2,
        action_dim2 = 1,
        action_dim3 = 1,
        action_dim4 = 2,
        n_actions = 4,train_interval= train_interval, pre_train=200000, before_training=bef_train, para=None, is_test=False, is_save=True
)
    #FIXME need lock?
    #lock = threading.Lock()
    threads = []
    for i in range(agent_num):
        threads.append(threading.Thread(target=player, args=(i,)))
    for t in threads:
        t.setDaemon(True)
        t.start()
        sleep(5)

    [t.join() for t in threads]
    print('--Game Over.', ctime())

