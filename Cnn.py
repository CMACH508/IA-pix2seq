#   Copyright 2022 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Modules for CNN & FC """

import tensorflow as tf

class DilatedConv(object):
    def __init__(self, specs, inputs, is_training, keep_prob=1.0):
        self.conv_layers = []
        self.keep_prob = keep_prob
        outputs = inputs
        for i, (fun_name, w_size, rate, out_channel) in enumerate(specs):
            self.conv_layers.append(outputs)
            with tf.variable_scope('conv%d' % i):
                outputs = self.build_dilated_conv_layer(outputs, w_size, out_channel, rate, fun_name, is_training=is_training)
        self.conv_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def batch_instance_norm(self, x):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0),
                              constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

    def get_filter(self, name, shape):
        return tf.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape))

    def get_bias(self, name, shape):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_dilated_conv_layer(self, inputs, w_size, out_channel, rate, actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], in_channel, out_channel])
        f = self.select_act_func(actfun)
        conv = tf.nn.atrous_conv2d(inputs, w, rate=rate, padding='SAME')
        conv = tf.contrib.layers.instance_norm(conv)
        # conv = self.batch_instance_norm(conv)
        # conv = tf.layers.batch_normalization(conv, axis=0, training=is_training)

        """ bias ver """
        # b = self.get_bias('bias', [out_channel])
        # out = f(conv + b)
        out = f(conv)
        return out

class ConvNet(object):
    def __init__(self, specs, inputs, is_training, deconv=False, keep_prob=1.0):
        self.conv_layers = []
        self.keep_prob = keep_prob
        outputs = inputs
        for i, (fun_name, w_size, strides, out_channel) in enumerate(specs):
            self.conv_layers.append(outputs)
            if deconv == False:
                with tf.variable_scope('conv%d' % i):
                    outputs = self.build_conv_layer(outputs, w_size, out_channel, strides, fun_name, is_training=is_training)
            else:
                with tf.variable_scope('deconv%d' % i):
                    outputs = self.build_deconv_layer(outputs, w_size, out_channel, strides, fun_name, is_training=is_training)
        self.conv_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def batch_instance_norm(self, x):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0),
                              constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

    def get_filter(self, name, shape):
        return tf.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape))

    def get_bias(self, name, shape):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_conv_layer(self, inputs, w_size, out_channel, strides=[1, 2, 2, 1], actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], in_channel, out_channel])
        f = self.select_act_func(actfun)
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding='SAME')
        conv = tf.contrib.layers.instance_norm(conv)
        # conv = self.batch_instance_norm(conv)
        # conv = tf.layers.batch_normalization(conv, axis=0, training=is_training)

        """ bias ver """
        # b = self.get_bias('bias', [out_channel])
        # out = f(conv + b)
        out = f(conv)
        return out

    def build_deconv_layer(self, inputs, w_size, out_channel, strides=[1, 2, 2, 1], actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], out_channel, in_channel])
        if strides[1] == 1:
            deconv_shape = [batch_size, height, width, out_channel]
        else:
            deconv_shape = [batch_size, height * 2, width * 2, out_channel]
        f = self.select_act_func(actfun)
        conv = tf.nn.conv2d_transpose(inputs, w, deconv_shape, strides=strides, padding='SAME')
        conv = tf.contrib.layers.instance_norm(conv)
        # conv = self.batch_instance_norm(conv)
        # conv = tf.layers.batch_normalization(conv, axis=0, training=is_training)

        """ bias ver """
        # b = self.get_bias('bias', [out_channel])
        # out = f(conv + b)
        out = f(conv)
        return out

class FcNet(object):
    def __init__(self, specs, inputs):
        self.fc_layers = []
        outputs = inputs
        for i, (fun_name, in_size, out_size, scope) in enumerate(specs):
            self.fc_layers.append(outputs)
            with tf.variable_scope(scope):
                outputs = self.build_fc_layer(x=outputs, input_size=in_size, output_size=out_size, actfun=fun_name)
        self.fc_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_fc_layer(self, x, input_size, output_size, actfun, scope=None, use_bias=True):
        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(name='fc_w', dtype=tf.float32, initializer=self.xavier_init([input_size, output_size]))
            if use_bias:
                b = tf.get_variable('fc_b', [output_size], tf.float32, initializer=tf.constant_initializer(0.0))
                temp = tf.matmul(x, w) + b
            else:
                temp = tf.matmul(x, w)
            if actfun == 'no':
                return temp
            else:
                f = self.select_act_func(actfun)
                return f(temp)