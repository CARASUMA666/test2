import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math


 # TODO delete sess or not? placeholder or just input?
class actor_network:
    def __init__(self,sess,n_state,n_action,LAYER1_SIZE = 256,LAYER2_SIZE = 128,LAYER3_SIZE = 64,Beta = 1e-4,TAU = 0.001,BATCH_SIZE = 32,mark = None,reuse = False):
        self.sess = sess
        self.n_state = n_state #state_dim
        self.n_action = n_action #action_dim

        self.LAYER1_SIZE = LAYER1_SIZE
        self.LAYER2_SIZE = LAYER2_SIZE
        self.LAYER3_SIZE = LAYER3_SIZE
        self.Beta = Beta
        self.TAU = TAU
        self.BATCH_SIZE = BATCH_SIZE
        self.mark = '' if mark is None else mark
            # create target actor network
        '''self.target_state_input, self.target_action_output, self.target_update, self.target_is_training = self.create_target_network(
        state_dim, action_dim, self.net)'''
        self.state_input, self.next_state_input, self.action_output, self.action_output_, self.net, self.is_training = self.create_network(
            n_state, n_action,reuse = reuse)
            # define training rules
        self.create_training_method()

        #self.update_target()
            # self.load_network()
    def create_network(self,n_state,n_action,reuse = False):
        layer1_size = self.LAYER1_SIZE
        layer2_size = self.LAYER2_SIZE
        layer3_size = self.LAYER3_SIZE
        state_input = tf.placeholder("float", [None, self.n_state])
        next_state_input = tf.placeholder("float", [None, self.n_state])
        is_training = tf.placeholder(tf.bool)
        #a_initializer = tf.random_uniform_initializer(-1 / math.sqrt(f), 1 / math.sqrt(f))
        with tf.variable_scope("actor" + self.mark,reuse = reuse):
            W1 = tf.get_variable("actor",[n_state, layer1_size], initializer= tf.random_uniform_initializer(-1 / math.sqrt(n_state), 1 / math.sqrt(n_state)))
            b1 = tf.get_variable("actor",[layer1_size], initializer = tf.random_uniform_initializer(-1 / math.sqrt(n_state), 1 / math.sqrt(n_state)))
            W2 = tf.get_variable("actor",[layer1_size, layer2_size],initializer= tf.random_uniform_initializer(-1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)))
            b2 = tf.get_variable("actor",[layer2_size], initializer= tf.random_uniform_initializer(-1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)))
            W3 = tf.get_variable("actor",[layer2_size,layer3_size],initializer= tf.random_uniform_initializer(-1 / math.sqrt(layer2_size), 1 / math.sqrt(layer2_size)))
            b3 = tf.get_variable("actor",[layer3_size],initializer= tf.random_uniform_initializer(-1 / math.sqrt(layer2_size), 1 / math.sqrt(layer2_size)))
            W4 = tf.get_variable("actor",[layer3_size, n_action], initializer= tf.random_uniform_initializer(-3e-3, 3e-3))
            b4 = tf.get_variable("actor",[n_action], initializer = tf.random_uniform_initializer(-3e-3, 3e-3))

            layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training, scope_bn='batch_norm_0',
                                          activation=tf.identity)
            layer1 = tf.matmul(layer0_bn, W1) + b1
            layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='batch_norm_1',
                                          activation=tf.nn.relu)
            layer2 = tf.matmul(layer1_bn, W2) + b2
            layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='batch_norm_2',
                                          activation=tf.nn.relu)
            layer3 = tf.matmul(layer2_bn, W3) + b3
            layer3_bn = self.batch_norm_layer(layer3, training_phase=is_training, scope_bn='batch_norm_3',
                                          activation=tf.nn.relu)
            action_output = tf.tanh(tf.matmul(layer3_bn, W4) + b4)
            layer0_bn_ = self.batch_norm_layer(next_state_input, training_phase=is_training, scope_bn='batch_norm_0',
                                          activation=tf.identity)
            layer1_ = tf.matmul(layer0_bn_, W1) + b1
            layer1_bn_ = self.batch_norm_layer(layer1_, training_phase=is_training, scope_bn='batch_norm_1',
                                          activation=tf.nn.relu)
            layer2_ = tf.matmul(layer1_bn_, W2) + b2
            layer2_bn_ = self.batch_norm_layer(layer2_, training_phase=is_training, scope_bn='batch_norm_2',
                                          activation=tf.nn.relu)
            layer3_ = tf.matmul(layer2_bn_, W3) + b3
            layer3_bn_ = self.batch_norm_layer(layer3_, training_phase=is_training, scope_bn='batch_norm_3',
                                          activation=tf.nn.relu)
            action_output_ = tf.tanh(tf.matmul(layer3_bn_, W4) + b4)
        return state_input, next_state_input,action_output,action_output_, [W1, b1, W2, b2, W3, b3, W4, b4], is_training
    def create_training_method(self):
        with tf.variable_scope("gradient" + self.mark):
            self.q_gradient_input = tf.placeholder("float",[None,self.n_action])
            self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
            self.optimizer = tf.train.AdamOptimizer(self.Beta).apply_gradients(zip(self.parameters_gradients,self.net))

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.is_training: True
        })
    #FIXME 0001
    def actions(self, state_batch,state_batch_):
        return self.sess.run([self.action_output,self.action_output_], feed_dict={
            self.state_input: state_batch,
            self.next_state_input: state_batch_,
            self.is_training: True
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state],
            self.is_training: False
        })[0]

    #def concat_action(self,a1,a2,a3):
    #    a_list = a1+ a2+ a3
    #    a_con = np.array(a_list).reshape((-1,))
    #    return a_con
    #FIXME 0052
    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(training_phase,
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                        scope=scope_bn, decay=0.9, epsilon=1e-5))
class DeepQNetwork:
    def __init__(self,
            n_actions,
            action_dim1,
            action_dim2,
            action_dim3,
            action_dim4,
            n_global_features,
            n_features,
            LAYER1_SIZE = 256,
            LAYER2_SIZE = 128,
            LAYER3_SIZE = 64,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=200,
            tau=0.01,
            memory_size=2000,
            batch_size=32,
            epsilon=1.0,
            e_greedy_increment=0.001,
            soft_replace=False,
            output_graph=False,
            GPU_divide=None,
            session=None,
            mark=None,
            is_dueling = None):
        self.n_actions = n_actions
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.action_dim3 = action_dim3
        self.action_dim4 = action_dim4
        self.n_global_features = n_global_features
        self.n_features = n_features
        self.LAYER1_SIZE = LAYER1_SIZE
        self.LAYER2_SIZE = LAYER2_SIZE
        self.LAYER3_SIZE = LAYER3_SIZE
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon
        self.tau = tau
        self.soft_replace = soft_replace
        self.sess = session
        self.mark = '' if mark is None else mark
        self.is_dueling = is_dueling
        self.learn_step_counter = 0
        self.memory1 = np.zeros((self.memory_size, self.n_features * 2 + self.n_global_features * 2 + 3))
        self.memory2 = np.zeros((self.memory_size, self.n_features * 2 + self.n_global_features * 2 + 3))
        #self.create_training_method()
        self.create_dqn()
        self.q_concat()
        self.q_concat_g()
        self.q_mix_network()
        self.q_mix_training()

        #self.update_target()

    #def calculate_gradient(self,):
    def create_dqn(self):
        with tf.variable_scope('input_placeholders' + self.mark):



            #self.q1_m_ = tf.placeholder(tf.float32, [None, ], name='q1_value_next')
            #self.q2_m_ = tf.placeholder(tf.float32, [None, ], name='q2_value_next')
            #self.q3_m_ = tf.placeholder(tf.float32, [None, ], name='q3_value_next')
            # self.q_concat_ = tf.placeholder(tf.float32, [None, 3], name='q_value_concat_next')

            self.s1 = tf.placeholder(tf.float32, [None, self.n_features], name='pdqn_s1')  # input State
            self.s2 = tf.placeholder(tf.float32, [None, self.n_features], name='pdqn_s2')  # input State
            self.s1_ = tf.placeholder(tf.float32, [None, self.n_features], name='pdqn_s1_')  # input Next State
            self.s2_ = tf.placeholder(tf.float32, [None, self.n_features], name='pdqn_s2_')  # input Next State
            #self.r = tf.placeholder(tf.float32, [None, ], name='pdqn_r')  # input Reward
            #self.a1 = tf.placeholder(tf.int32, [None, ], name='pdqn_a1')  # input Action
            #self.a2 = tf.placeholder(tf.int32, [None, ], name='pdqn_a2')  # input Action
            #self.a3 = tf.placeholder(tf.int32, [None, ], name='pdqn_a3')  # input Action
            #self.a1_ = tf.placeholder(tf.int32, [None, ], name='pdqn_a1_')  # input Action
            #self.a2_ = tf.placeholder(tf.int32, [None, ], name='pdqn_a2_')  # input Action
            #self.a3_ = tf.placeholder(tf.int32, [None, ], name='pdqn_a3_')
        #self.t = tf.placeholder(tf.float32, [None, ], name='pdqn_t')  # subtask duration
        q_initializer = tf.random_normal_initializer(stddev=0.01), tf.constant_initializer(0.)
        with tf.variable_scope("single_dqn"):
            layer1_size = self.LAYER1_SIZE
            layer2_size = self.LAYER2_SIZE
            layer3_size = self.LAYER3_SIZE
            n_features = self.n_features
            is_training = tf.placeholder(tf.bool)
            state_input1 = self.s1
            state_input2 = self.s2
            _state_input1 = self.s1_
            _state_input2 = self.s2_

            self.a1_action_input1 = tf.placeholder(tf.float32, [None, self.action_dim1])
            self.a1_action_input2 = tf.placeholder(tf.float32, [None, self.action_dim2])
            self.a1_action_input3 = tf.placeholder(tf.float32, [None, self.action_dim3])
            self.a1_action_input4 = tf.placeholder(tf.float32, [None, self.action_dim4])
            #self.a1_action_input = [self.a1_action_input1,self.a1_action_input2,self.a1_action_input3]
            self.a2_action_input1 = tf.placeholder(tf.float32, [None, self.action_dim1])
            self.a2_action_input2 = tf.placeholder(tf.float32, [None, self.action_dim2])
            self.a2_action_input3 = tf.placeholder(tf.float32, [None, self.action_dim3])
            self.a2_action_input4 = tf.placeholder(tf.float32, [None, self.action_dim4])
            #self.a2_action_input = [self.a2_action_input1,self.a2_action_input2,self.a2_action_input3]
            #self.a3_action_input = [self.a3_action_input1,self.a3_action_input2,self.a3_action_input3]
            self._a1_action_input1 = tf.placeholder(tf.float32, [None, self.action_dim1])
            self._a1_action_input2 = tf.placeholder(tf.float32, [None, self.action_dim2])
            self._a1_action_input3 = tf.placeholder(tf.float32, [None, self.action_dim3])
            self._a1_action_input4 = tf.placeholder(tf.float32, [None, self.action_dim4])
            # self.a1_action_input = [self.a1_action_input1,self.a1_action_input2,self.a1_action_input3]
            self._a2_action_input1 = tf.placeholder(tf.float32, [None, self.action_dim1])
            self._a2_action_input2 = tf.placeholder(tf.float32, [None, self.action_dim2])
            self._a2_action_input3 = tf.placeholder(tf.float32, [None, self.action_dim3])
            self._a2_action_input4 = tf.placeholder(tf.float32, [None, self.action_dim4])
            # self.a2_action_input = [self.a2_action_input1,self.a2_action_input2,self.a2_action_input3]

            action_dim1 = self.action_dim1
            action_dim2 = self.action_dim2
            action_dim3 = self.action_dim3
            action_dim4 = self.action_dim4
            W1 = tf.get_variable("W1",[n_features, layer1_size],initializer= q_initializer)
            b1 = tf.get_variable("b1",[layer1_size], initializer= q_initializer)

            a1_layer1 = tf.nn.relu(tf.matmul(state_input1,W1) + b1)
            a2_layer1 = tf.nn.relu(tf.matmul(state_input2,W1) + b1)

            _a1_layer1 = tf.nn.relu(tf.matmul(_state_input1, W1) + b1)
            _a2_layer1 = tf.nn.relu(tf.matmul(_state_input2, W1) + b1)

            W2 = tf.get_variable("W2",[layer1_size, layer2_size],initializer = q_initializer)
            b2 = tf.get_variable("b2", [layer2_size], initializer= q_initializer)

            a1_layer2 = tf.nn.relu(tf.matmul(a1_layer1 ,W2) + b2)
            a2_layer2 = tf.nn.relu(tf.matmul(a2_layer1, W2) + b2)

            _a1_layer2 = tf.nn.relu(tf.matmul(_a1_layer1, W2) + b2)
            _a2_layer2 = tf.nn.relu(tf.matmul(_a2_layer1, W2) + b2)

            W3 = tf.get_variable("W3",[layer2_size, layer3_size], initializer= q_initializer)
            W1_action = tf.get_variable("W1_action",[action_dim1, layer3_size],initializer= q_initializer)
            W2_action = tf.get_variable("W2_action",[action_dim2, layer3_size], initializer= q_initializer)
            W3_action = tf.get_variable("W3_action", [action_dim3, layer3_size], initializer= q_initializer)
            W4_action = tf.get_variable("W4_action", [action_dim4, layer3_size], initializer=q_initializer)
            b3 = tf.get_variable("b3",[layer3_size], initializer= q_initializer)

            a1_layer31 = tf.nn.relu(tf.matmul(a1_layer2, W3) + tf.matmul(self.a1_action_input1, W1_action) + b3)
            a1_layer32 = tf.nn.relu(tf.matmul(a1_layer2, W3) + tf.matmul(self.a1_action_input2, W2_action) + b3)
            a1_layer33 = tf.nn.relu(tf.matmul(a1_layer2, W3) + tf.matmul(self.a1_action_input3, W3_action) + b3)
            a1_layer34 = tf.nn.relu(tf.matmul(a1_layer2, W3) + tf.matmul(self.a1_action_input4, W4_action) + b3)

            a2_layer31 = tf.nn.relu(tf.matmul(a2_layer2, W3) + tf.matmul(self.a2_action_input1, W1_action) + b3)
            a2_layer32 = tf.nn.relu(tf.matmul(a2_layer2, W3) + tf.matmul(self.a2_action_input2, W2_action) + b3)
            a2_layer33 = tf.nn.relu(tf.matmul(a2_layer2, W3) + tf.matmul(self.a2_action_input3, W3_action) + b3)
            a2_layer34 = tf.nn.relu(tf.matmul(a2_layer2, W3) + tf.matmul(self.a2_action_input4, W4_action) + b3)


            _a1_layer31 = tf.nn.relu(tf.matmul(_a1_layer2, W3) + tf.matmul(self._a1_action_input1, W1_action) + b3)
            _a1_layer32 = tf.nn.relu(tf.matmul(_a1_layer2, W3) + tf.matmul(self._a1_action_input2, W2_action) + b3)
            _a1_layer33 = tf.nn.relu(tf.matmul(_a1_layer2, W3) + tf.matmul(self._a1_action_input3, W3_action) + b3)
            _a1_layer34 = tf.nn.relu(tf.matmul(_a1_layer2, W3) + tf.matmul(self._a1_action_input4, W4_action) + b3)

            _a2_layer31 = tf.nn.relu(tf.matmul(_a2_layer2, W3) + tf.matmul(self._a2_action_input1, W1_action) + b3)
            _a2_layer32 = tf.nn.relu(tf.matmul(_a2_layer2, W3) + tf.matmul(self._a2_action_input2, W2_action) + b3)
            _a2_layer33 = tf.nn.relu(tf.matmul(_a2_layer2, W3) + tf.matmul(self._a2_action_input3, W3_action) + b3)
            _a2_layer34 = tf.nn.relu(tf.matmul(_a2_layer2, W3) + tf.matmul(self._a2_action_input4, W4_action) + b3)


            W4 = tf.get_variable("W4",[layer3_size, 1], initializer= q_initializer)
            b4 = tf.get_variable("b4", [1], initializer= q_initializer)

            a1_q_value_output1 = tf.identity(tf.matmul(a1_layer31,W4) + b4)
            a1_q_value_output2 = tf.identity(tf.matmul(a1_layer32,W4) + b4)
            a1_q_value_output3 = tf.identity(tf.matmul(a1_layer33,W4) + b4)
            a1_q_value_output4 = tf.identity(tf.matmul(a1_layer34, W4) + b4)
            a1_q_value_output = tf.stack([a1_q_value_output1,a1_q_value_output2,a1_q_value_output3,a1_q_value_output4],axis = 1)
            self.Q1 = a1_q_value_output

            _a1_q_value_output1 = tf.identity(tf.matmul(_a1_layer31, W4) + b4)
            _a1_q_value_output2 = tf.identity(tf.matmul(_a1_layer32, W4) + b4)
            _a1_q_value_output3 = tf.identity(tf.matmul(_a1_layer33, W4) + b4)
            _a1_q_value_output4 = tf.identity(tf.matmul(_a1_layer34, W4) + b4)
            _a1_q_value_output = tf.stack([_a1_q_value_output1, _a1_q_value_output2, _a1_q_value_output3,_a1_q_value_output4], axis=1)
            self.Q1_ = _a1_q_value_output

            a2_q_value_output1 = tf.identity(tf.matmul(a2_layer31, W4) + b4)
            a2_q_value_output2 = tf.identity(tf.matmul(a2_layer32, W4) + b4)
            a2_q_value_output3 = tf.identity(tf.matmul(a2_layer33, W4) + b4)
            a2_q_value_output4 = tf.identity(tf.matmul(a2_layer34, W4) + b4)
            a2_q_value_output = tf.stack([a2_q_value_output1, a2_q_value_output2, a2_q_value_output3,a2_q_value_output4],axis = 1)
            self.Q2 = a2_q_value_output

            _a2_q_value_output1 = tf.identity(tf.matmul(_a2_layer31, W4) + b4)
            _a2_q_value_output2 = tf.identity(tf.matmul(_a2_layer32, W4) + b4)
            _a2_q_value_output3 = tf.identity(tf.matmul(_a2_layer33, W4) + b4)
            _a2_q_value_output4 = tf.identity(tf.matmul(_a2_layer34, W4) + b4)
            _a2_q_value_output = tf.stack([_a2_q_value_output1, _a2_q_value_output2, _a2_q_value_output3,_a2_q_value_output4], axis=1)
            self.Q2_ = _a2_q_value_output

        #trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        #return trainable_vars
    def q_mix_network(self):
        self.S = tf.placeholder(tf.float32, [None, self.n_global_features], name='pdqn')
        self.S_ = tf.placeholder(tf.float32, [None, self.n_global_features], name='pdqn')
        self.r = tf.placeholder(tf.float32, [None, ], name='pdqn_r')  # input Reward
        #self.t = tf.placeholder(tf.float32, [None, ], name='pdqn_t')  # subtask duration
        self.q_concat_a = tf.placeholder(tf.float32,[None,2], name = "q_concat_a")
        self.q_concat_b = tf.placeholder(tf.float32, [None, 2], name="q_concat_b")
        self.q_concat_t = tf.placeholder(tf.float32, [None, 2], name="q_concat_t")
        with tf.variable_scope("initializer"):
            w_initializer,b_initializer = tf.random_normal_initializer(stddev=0.01), tf.constant_initializer(0.)
        with tf.variable_scope("q_mix"):
            non_abs_w1 = tf.layers.dense(inputs=self.S,
                                         units=2 * 32,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='non_abs_w1')
            self.w1 = tf.reshape(tf.abs(non_abs_w1), shape=[-1, 2, 32], name='w1')
            self.b1 = tf.layers.dense(inputs=self.S,
                                      units=32,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='non_abs_b1')


            non_abs_w2 = tf.layers.dense(inputs=self.S,
                                         units=32 * 1,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='non_abs_w2')
            self.w2 = tf.reshape(tf.abs(non_abs_w2), shape=[-1, 32, 1], name='w2')
            bef_b2 = tf.layers.dense(inputs=self.S,
                                     units=32,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='bef_b2')
            self.b2 = tf.layers.dense(inputs=bef_b2,
                                      units=1,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='non_abs_b2')
            lin1 = tf.matmul(tf.reshape(self.q_concat_a, shape=[-1, 1, 2]), self.w1) \
                   + tf.reshape(self.b1, shape=[-1, 1, 32])
            a1 = tf.nn.elu(lin1, name='a1')
            self.Q_tot = tf.reshape(tf.matmul(a1, self.w2), shape=[-1, 1]) + self.b2

            g_lin1 = tf.matmul(tf.reshape(self.q_concat_b, shape=[-1, 1, 2]), self.w1) \
                   + tf.reshape(self.b1, shape=[-1, 1, 32])
            g_a1 = tf.nn.elu(g_lin1, name='g_a1')
            self.g_Q_tot = tf.reshape(tf.matmul(g_a1, self.w2), shape=[-1, 1]) + self.b2
        with tf.variable_scope('q_target'):
            _lin1 = tf.matmul(tf.reshape(self.q_concat_t, shape=[-1, 1, 2]), self.w1) \
                   + tf.reshape(self.b1, shape=[-1, 1, 32])
            _a1 = tf.nn.elu(_lin1, name='_a1')
            self._Q_tot = tf.reshape(tf.matmul(_a1, self.w2), shape=[-1, 1]) + self.b2
                # self.Q_tot_ = tf.placeholder(tf.float32, [None, ], name='Q_tot_')
            q_target = self.r + self.gamma* self._Q_tot
            self.q_target = tf.stop_gradient(q_target)

    def q_concat(self):
        self.a1 = tf.placeholder(tf.int32, [None, ], name='pdqn_a1')  # input Action
        self.a2 = tf.placeholder(tf.int32, [None, ], name='pdqn_a2')  # input Action

        with tf.variable_scope('q_concat_a'):
            #if update_continuous_act == False:
            a1_indices = tf.stack([tf.range(tf.shape(self.a1)[0], dtype=tf.int32), self.a1], axis=1,
                                      name='a1_indices')
            a2_indices = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), self.a2], axis=1,
                                      name='a2_indices')
            self.q1_a = tf.gather_nd(params=self.Q1, indices=a1_indices, name='q1_eval_wrt_a')  # shape=(None, )
            self.q2_a = tf.gather_nd(params=self.Q2, indices=a2_indices, name='q2_eval_wrt_a')  # shape=(None, )
            self.q_concat_a = tf.stack([self.q1_a, self.q2_a], axis=1)

    def q_concat_g(self):
        with tf.variable_scope('q_concat_b'):
            self.q1_b = tf.reduce_mean(self.Q1,axis = 1)
            self.q2_b = tf.reduce_mean(self.Q2,axis = 1)
            self.q_concat_b = tf.stack([self.q1_b, self.q2_b], axis=1)
            #self.q_concat_ = tf.stack([self.q1_m_, self.q2_m_, self.q3_m_], axis=1, name='q_concat_next')

    def q_mix_training(self):
            with tf.variable_scope('q_mix_loss' + self.mark):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.Q_tot, name='TD_error'))
                # tf.summary.scalar('loss' + self.mark, self.loss)
            with tf.variable_scope('q_mix_train' + self.mark):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            with tf.variable_scope('q_gradients'+self.mark):
                self.all_q_tot = tf.placeholder(tf.float32,[None,],name = "all_q_tot")
                self.a1_q_gradients_1 = tf.gradients(self.all_q_tot, self.a1_action_input1)
                self.a1_q_gradients_2 = tf.gradients(self.all_q_tot, self.a1_action_input2)
                self.a1_q_gradients_3 = tf.gradients(self.all_q_tot, self.a1_action_input3)
                self.a1_q_gradients_4 = tf.gradients(self.all_q_tot, self.a1_action_input4)
                self.a2_q_gradients_1 = tf.gradients(self.all_q_tot, self.a2_action_input1)
                self.a2_q_gradients_2 = tf.gradients(self.all_q_tot, self.a2_action_input2)
                self.a2_q_gradients_3 = tf.gradients(self.all_q_tot, self.a2_action_input3)
                self.a2_q_gradients_4 = tf.gradients(self.all_q_tot, self.a2_action_input4)

                #self.q_gradients_1 = tf.reduce_mean()
    def choose_action(self, observation, c_action1,c_action2,c_action3,c_action4,index, is_test=False):
        if index not in [0, 1]:
            print('--FATAL ERROR: Wrong index when choosing action.')
            return None

        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if is_test:
            if index == 0:
                actions_value = self.sess.run(self.Q1, feed_dict={self.s1: observation,self.a1_action_input1:c_action1,self.a1_action_input2:c_action1,self.a1_action_input3:c_action1,self.a1_action_input4:c_action4})
            else:
                actions_value = self.sess.run(self.Q2, feed_dict={self.s2: observation,self.a2_action_input1:c_action2,self.a2_action_input2:c_action2,self.a2_action_input3:c_action2,self.a2_action_input4:c_action4})
            action = np.argmax(actions_value)
        else:
            if np.random.uniform() > self.epsilon:
                # forward feed the observation and get q value for every actions
                if index == 0:
                    actions_value = self.sess.run(self.Q1, feed_dict={self.s1: observation,self.a1_action_input1:c_action1,self.a1_action_input2:c_action1,self.a1_action_input3:c_action1,self.a1_action_input4:c_action4})
                else:
                    actions_value = self.sess.run(self.Q2, feed_dict={self.s2: observation,self.a2_action_input1:c_action2,self.a2_action_input2:c_action2,self.a2_action_input3:c_action2,self.a2_action_input4:c_action4})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        if action==0:
            c_action = c_action1
        elif action==1:
            c_action = c_action2
        elif action==2:
            c_action = c_action3
        else:
            c_action = c_action4
        return action,c_action
    def store_transition1(self, s1, s1_, a1, a1_c,  r, S, S_):
        if not hasattr(self, 'memory_counter1'):
            self.memory_counter1 = 0
        # FIXME 0821
        transition_list1 = s1 + s1_ +  [a1, a1_c, r] + S + S_
        transition1 = np.array(transition_list1).reshape((-1,))
        # replace the old memory with new memory
        index = self.memory_counter1 % self.memory_size
        self.memory1[index, :] = transition1
        self.memory_counter1 += 1

    def store_transition2(self, s2, s2_, a2, a2_c, r, S, S_):
        if not hasattr(self, 'memory_counter2'):
            self.memory_counter2 = 0
        # FIXME 0821
        transition_list2 = s2 + s2_ + [a2, a2_c, r] + S + S_
        transition2 = np.array(transition_list2).reshape((-1,))
        # replace the old memory with new memory
        index = self.memory_counter2 % self.memory_size
        self.memory2[index, :] = transition2
        self.memory_counter2 += 1
#    def store_transition(self, s1, s2, s3,s1_,s2_,s3_, a1, a1_c, a2, a2_c, a3, a3_c, r, S, S_):
#        if not hasattr(self, 'memory_counter'):
#            self.memory_counter = 0
#        # FIXME 0821
#       transition_list = s1 + s2 + s3 + s1_ + s2_ + s3_ + [a1, a1_c,a2,a2_c, a3,a3_c, r] + S + S_
#        transition = np.array(transition_list).reshape((-1,))
#        # replace the old memory with new memory
#        index = self.memory_counter % self.memory_size
#        self.memory[index, :] = transition
#        self.memory_counter += 1
    def get_batch(self):
        # sample batch memory from all memory
        if self.memory_counter1 > self.memory_size and self.memory_counter2 > self.memory_size and self.memory_counter3 > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter1, size=self.batch_size)
            # FIXME if Memory not full, return
            # return
        batch_memory1 = self.memory1[sample_index, :]
        batch_memory2 = self.memory2[sample_index, :]
        # FIXME further check
        f_s1 = np.array(batch_memory1[:, :self.n_features]).reshape([-1, self.n_features])
        f_s2 = np.array(batch_memory2[:, :self.n_features]).reshape([-1, self.n_features])
        f_s1_ = np.array(batch_memory1[:, self.n_features:self.n_features * 2]).reshape([-1, self.n_features])
        f_s2_ = np.array(batch_memory2[:, self.n_features:self.n_features * 2]).reshape([-1, self.n_features])

        a1 = batch_memory1[:, self.n_features * 2].astype(int)
        a1_c = batch_memory1[:,self.n_features * 2 + 1].astype(float)
        a2 = batch_memory2[:, self.n_features * 2 ].astype(int)
        a2_c = batch_memory2[:, self.n_features * 2+ 1].astype(float)
        r1 = batch_memory1[:, self.n_features * 2 + 2].astype(float)
        r2 = batch_memory2[:, self.n_features * 2 + 2].astype(float)
        # t = batch_memory[:, self.n_features * 6 + 4]
        r = tf.reduce_mean([r1,r2], axis = 0)
        S = np.array(batch_memory1[:, -self.n_global_features * 2:-self.n_global_features]).reshape(
            [-1, self.n_global_features])
        S_ = np.array(batch_memory1[:, -self.n_global_features:]).reshape([-1, self.n_global_features])
        return f_s1,f_s2,f_s1_,f_s2_,a1,a2,r,S,S_
    def learn(self,f_s1,f_s2,f_s1_,f_s2_,a1,a2,r,S,S_,
              a1_action_output1,
              a1_action_output2,
              a1_action_output3,
              a1_action_output4,
              a2_action_output1,
              a2_action_output2,
              a2_action_output3,
              a2_action_output4,
              _a1_action_output1,
              _a1_action_output2,
              _a1_action_output3,
              _a1_action_output4,
              _a2_action_output1,
              _a2_action_output2,
              _a2_action_output3,
              _a2_action_output4):
            # check to replace target parameters
            #if self.soft_replace:
            #    self.sess.run(self.target_soft_replace_op)
            #elif self.learn_step_counter % self.replace_target_iter == 0:
            #    self.sess.run(self.target_replace_op)

            q1, q2, q1_, q2_ = self.sess.run([self.Q1, self.Q2,self.Q1_,self.Q2_], feed_dict={self.s1:f_s1,
                                                                                              self.s2:f_s2,
                                                                                              self.s1_: f_s1_,
                                                                                              self.s2_: f_s2_,
                                                                                              self.a1_action_input1:a1_action_output1,
                                                                                              self.a1_action_input2:a1_action_output2,
                                                                                              self.a1_action_input3:a1_action_output3,
                                                                                              self.a1_action_input4:a1_action_output4,
                                                                                              self.a2_action_input1:a2_action_output1,
                                                                                              self.a2_action_input2:a2_action_output2,
                                                                                              self.a2_action_input3:a2_action_output3,
                                                                                              self.a2_action_input4:a2_action_output4,self._a1_action_input1:_a1_action_output1,self._a1_action_input2:_a1_action_output2,self._a1_action_input3:_a1_action_output3,self._a1_action_input4:_a1_action_output4,self._a2_action_input1:_a2_action_output1,self._a2_action_input2:_a2_action_output2,self._a2_action_input3:_a2_action_output3,self._a2_action_input4:_a2_action_output4})
            #q_concat_1 = self.sess.run(self.q_concat_a, feed_dict = {self.a1:a1,self.a2:a2,self.a3:a3,self.Q1:q1,self.Q2:q2,self.Q3:q3})
            q_concat_2 = self.sess.run(self.q_concat_b, feed_dict = {self.Q1:q1,self.Q2:q2})
            q1_m_ = tf.reduce_max(q1_, axis=1)
            q2_m_ = tf.reduce_max(q2_, axis=1)
            q_concat_ = tf.stack([q1_m_, q2_m_], axis=1)
            q_tot_g = self.sess.run(self.g_Q_tot, feed_dict={self.S: S,self.q_concat_b: q_concat_2})
            q_target = self.sess.run(self.q_target,feed_dict = {self.S: S,self.q_concat_t:q_concat_,self.r: r})
            # FIXME 0000
            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.S: S,
                    self.s1: f_s1,
                    self.s2: f_s2,
                    self.a1: a1,
                    self.a2: a2,
                    self.a1_action_input1: a1_action_output1,
                    self.a1_action_input2: a1_action_output2,
                    self.a1_action_input3: a1_action_output3,
                    self.a1_action_input4: a1_action_output4,
                    self.a2_action_input1: a2_action_output1,
                    self.a2_action_input2: a2_action_output2,
                    self.a2_action_input3: a2_action_output3,
                    self.a2_action_input4: a2_action_output4,
                    self.q_target:q_target,
                })
            #FIXME
            a1_q_gradients_1,\
            a1_q_gradients_2,\
            a1_q_gradients_3,\
            a1_q_gradients_4,\
            a2_q_gradients_1,\
            a2_q_gradients_2,\
            a2_q_gradients_3,\
            a2_q_gradients_4 = self.sess.run([self.a1_q_gradients_1,self.a1_q_gradients_2,self.a1_q_gradients_3,self.a1_q_gradients_4,self.a2_q_gradients_1,self.a2_q_gradients_2,self.a2_q_gradients_3,self.a2_q_gradients_4],feed_dict = {
                self.all_q_tot:q_tot_g})


            # self.cost_his.append(cost)

            # increasing epsilon
            self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
            # print(self.epsilon)
            self.learn_step_counter += 1

            return cost,[a1_q_gradients_1,a1_q_gradients_2,a1_q_gradients_3,a1_q_gradients_4,a2_q_gradients_1,a2_q_gradients_2,a2_q_gradients_3,a2_q_gradients_4]