import threading
import random
import numpy as np
import tensorflow as tf
import datetime
import os
from collections import deque
import pickle
from replay_buffer import ReplayBuffer
from continuous_actor import actor_network
from agent2 import actor_network
from agent2 import DeepQNetwork
class P_DQN:
    def __init__(
        self,
        n_features,
        n_global_features,
        action_dim1,
        action_dim2,
        action_dim3,
        action_dim4,
        n_actions =4,train_interval=3, pre_train=200000, before_training=20000, para=None, is_test=False, is_save=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_global_features = n_global_features
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.action_dim3 = action_dim3
        self.action_dim4 = action_dim4
        self.train_interval = train_interval
        self.pre_train = pre_train
        self.before_training = before_training

        self.train_cnt = 0

        # FIXME 0824
        #self.experience_dict = dict()

        # FIXME 0829
        #self.reward_his = []

        self.IS_TEST = is_test
        self.IS_SAVE = is_save

        config = tf.ConfigProto()
        #FIXME gpu_divide
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()

        # total learning step
        self.learn_step_counter = 0
        #FIXME 0021
        self.a1_actor_network_k1 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim1,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '1')
        self.a2_actor_network_k1 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim1,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '1',reuse = True)
        #elf.a3_actor_network_k1 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim1,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '1',reuse = True)
        self.a1_actor_network_k2 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim2,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '2')
        self.a2_actor_network_k2 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim2,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '2',reuse = True)
        #self.a3_actor_network_k2 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim2,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '2',reuse = True)
        self.a1_actor_network_k3 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim3,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '3')
        self.a2_actor_network_k3 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim3,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '3',reuse = True)
        #self.a3_actor_network_k3 = actor_network(sess = self.sess, n_state = self.n_features, n_action = self.action_dim3,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = '3',reuse = True)
        self.a1_actor_network_k4 = actor_network(sess=self.sess, n_state=self.n_features, n_action=self.action_dim4,
                                                 LAYER1_SIZE=256, LAYER2_SIZE=128, LAYER3_SIZE=64, Beta=1e-4, TAU=0.001,
                                                 BATCH_SIZE=32, mark='4')
        self.a2_actor_network_k4 = actor_network(sess=self.sess, n_state=self.n_features, n_action=self.action_dim4,
                                                 LAYER1_SIZE=256, LAYER2_SIZE=128, LAYER3_SIZE=64, Beta=1e-4, TAU=0.001,
                                                 BATCH_SIZE=32, mark='4', reuse=True)

        self.p_dqn = DeepQNetwork(n_actions = self.n_actions,action_dim1 = self.action_dim1,action_dim2 = self.action_dim2,action_dim3 = self.action_dim3,action_dim4 = self.action_dim4,n_global_features = self.n_global_features,n_features = self.n_features)
        #self.p_dqn_target = DeepQNetwork(self.n_actions,self.action_dim1, self.action_dim2, self.action_dim3,self.n_global_features, self.n_features, name="q_network", reuse = True)
        if self.IS_TEST:
            self.saver = tf.train.Saver()
            print('Loading checkpoint ...')
            #FIXME 0005
            load_path = "/home/haotianf18/checkpoints/PC_zero_IL_h_2018_08_16_17_05_31.ckpt"
            self.saver.restore(self.sess, load_path)
            print('Checkpoint: ' + load_path + ' is loaded.')
        else:
            if self.IS_SAVE:
                self.saver = tf.train.Saver(max_to_keep=5)
            self.sess.run(tf.global_variables_initializer())

        self.lock = threading.RLock()

        #self.update_cnt = 0
        #self.ready_update_cnt = 0
        #FIXME 0050
        self.episode_cnt = [0,0]
        self.global_step_cnt = 0
        #self.block_his = []
        #self.miss_his = []
        # self.memory = Memory(20)

        self.initial_flag = False
        #FIXME 0006
        self.train_writer = tf.summary.FileWriter("/home/haotianf18/data/mapdqn_summary/mapdqn_" + '1_'
                                                  + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                  self.sess.graph)
        self.merged = tf.summary.merge_all()

    def act(self, state, index, info=None):
        if state is None or index is None:
            return -1
        self.lock.acquire()
        state = np.asarray(state)
        if index == 0:
            c_action1 = self.a1_actor_network_k1.action(state)
            c_action2 = self.a1_actor_network_k2.action(state)
            c_action3 = self.a1_actor_network_k3.action(state)
            c_action4 = self.a1_actor_network_k4.action(state)
        elif index == 1:
            c_action1 = self.a2_actor_network_k1.action(state)
            c_action2 = self.a2_actor_network_k2.action(state)
            c_action3 = self.a2_actor_network_k3.action(state)
            c_action4 = self.a2_actor_network_k4.action(state)

        act,c_action = self.p_dqn.choose_action(state,c_action1,c_action2,c_action3,c_action4,index)
        #FIXME 0051
        self.global_step_cnt += 1
        self.lock.release()
        return act,c_action
#    def storeTransition(self, transition, info=None):
#        self.lock.acquire()
#        self.p_dqn.store_transition(s1=transition[0],s2 = transition[1],s3 = transition[2],s1_ = transition[3],s2_ = transition[4],s3_ = transition[5],
#                                  a1=transition[6],a1_c = transition[7],a2 = transition[8],a2_c = transition[9],a3 = transition[10],a3_c = transition[11],
#                                  r=transition[12],S = transition[13],S_ = transition[14])
#        self.lock.release()
    #TODO 0040
    def storeTransition1(self, s1,s1_,a1,a1_c,r,S,S_, info=None):
        self.lock.acquire()
        self.p_dqn.store_transition1(s1=s1,s1_ = s1_,a1=a1,a1_c = a1_c,
                                    r=r,S = S,S_ = S_)
        self.lock.release()
    def storeTransition2(self, s2,s2_,a2,a2_c,r,S,S_, info=None):
        self.lock.acquire()
        self.p_dqn.store_transition2(s2=s2,s2_ = s2_,a2=a2,a2_c = a2_c,
                                    r=r,S = S,S_ = S_)
        self.lock.release()

    #TODO inverting gradients
    def invert_grads(self,gradients, actions):
        tmp_a = np.array(actions)
        tmp = np.array(gradients)
        tmp[tmp >= 0] = tmp[tmp >= 0] * (1 - a[tmp >= 0]) / 2
        tmp[tmp < 0] = tmp[tmp < 0] * (a[tmp < 0] + 1) / 2
        return tmp
    def train(self, info=None):
        self.lock.acquire()
        #FIXME 0023
        if self.global_step_cnt > self.before_training and self.global_step_cnt % self.train_interval == 0:
            f_s1, f_s2, f_s1_, f_s2_, a1, a2, r, S, S_ = self.p_dqn.get_batch()
            a1_c1, _a1_c1 = self.a1_actor_network_k1.actions(f_s1, f_s1_)
            a1_c2, _a1_c2 = self.a1_actor_network_k2.actions(f_s1, f_s1_)
            a1_c3, _a1_c3 = self.a1_actor_network_k3.actions(f_s1, f_s1_)
            a1_c4, _a1_c4 = self.a1_actor_network_k4.actions(f_s1, f_s1_)
            a2_c1, _a2_c1 = self.a2_actor_network_k1.actions(f_s2, f_s2_)
            a2_c2, _a2_c2 = self.a2_actor_network_k2.actions(f_s2, f_s2_)
            a2_c3, _a2_c3 = self.a2_actor_network_k3.actions(f_s2, f_s2_)
            a2_c4, _a2_c4 = self.a2_actor_network_k4.actions(f_s2, f_s2_)

            loss,q_gradients = self.p_dqn.learn(f_s1, f_s2,f_s1_, f_s2_, a1, a2, r, S, S_,a1_c1,a1_c2,a1_c3,a1_c4,a2_c1,a2_c2,a2_c3,a2_c4,_a1_c1,_a1_c2,_a1_c3,_a1_c4,_a2_c1,_a2_c2,_a2_c3,_a2_c4)
            q_gradients[0] = self.invert_grads(q_gradients[0], a1_c1)
            q_gradients[1] = self.invert_grads(q_gradients[1], a1_c2)
            q_gradients[2] = self.invert_grads(q_gradients[2], a1_c3)
            q_gradients[3] = self.invert_grads(q_gradients[3], a1_c4)
            q_gradients[4] = self.invert_grads(q_gradients[4], a2_c1)
            q_gradients[5] = self.invert_grads(q_gradients[5], a2_c2)
            q_gradients[6] = self.invert_grads(q_gradients[6], a2_c3)
            q_gradients[7] = self.invert_grads(q_gradients[7], a2_c4)
            #FIXME 0020
            self.a1_actor_network_k1.train(q_gradients[0], f_s1)
            self.a1_actor_network_k2.train(q_gradients[1], f_s1)
            self.a1_actor_network_k3.train(q_gradients[2], f_s1)
            self.a1_actor_network_k4.train(q_gradients[3], f_s1)
            self.a2_actor_network_k1.train(q_gradients[4], f_s2)
            self.a2_actor_network_k2.train(q_gradients[5], f_s2)
            self.a2_actor_network_k3.train(q_gradients[6], f_s2)
            self.a2_actor_network_k4.train(q_gradients[7], f_s2)
            self.train_cnt += 1
            self.write_summary_scalar(self.train_cnt, "loss", loss)
        #FIXME 0010
        if self.IS_SAVE and self.episode_cnt[0] % 10000:
            # FIXME 0007
            path = '/home/haotianf18/checkpoints/201811/mapdqn_model.ckpt'
            self.saver.save(self.sess, path, global_step=self.episode_cnt[0])
        self.lock.release()
     #FIXME 0022
    def episode_done(self, index, info=None):
        # TODO 0000
        # SET episode_cnt start with 0?
        self.episode_cnt[index] += 1

    def write_summary_scalar(self, iteration, tag, value):
        self.train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)



