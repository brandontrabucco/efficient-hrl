# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sample actor(policy) and critic(q) networks to use with DDPG/NAF agents.

The DDPG networks are defined in "Section 7: Experiment Details" of
"Continuous control with deep reinforcement learning" - Lilicrap et al.
https://arxiv.org/abs/1509.02971

The NAF critic network is based on "Section 4" of "Continuous deep Q-learning
with model-based acceleration" - Gu et al. https://arxiv.org/pdf/1603.00748.
"""

import tensorflow as tf
slim = tf.contrib.slim
import gin.tf


@gin.configurable('sac_actor_net')
def actor_net(states, action_spec,
              hidden_layers=(400, 300),
              normalizer_fn=None,
              activation_fn=tf.nn.relu,
              zero_obs=False,
              images=False):
  """Creates an actor that returns actions for the given states.

  Args:
    states: (castable to tf.float32) a [batch_size, num_state_dims] tensor
      representing a batch of states.
    action_spec: (BoundedTensorSpec) A tensor spec indicating the shape
      and range of actions.
    hidden_layers: tuple of hidden layers units.
    normalizer_fn: Normalizer function, i.e. slim.layer_norm,
    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...
  Returns:
    A tf.float32 [batch_size, num_action_dims] tensor of actions.
  """

  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    states = tf.to_float(states)
    orig_states = states
    if images or zero_obs:  # Zero-out x, y position. Hacky.
      states *= tf.constant([0.0] * 2 + [1.0] * (states.shape[1] - 2))
    if hidden_layers:
      states = slim.stack(states, slim.fully_connected, hidden_layers,
                          scope='states')
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      actions = slim.fully_connected(states,
                                     2 * action_spec.shape.num_elements(),
                                     scope='actions',
                                     normalizer_fn=None,
                                     activation_fn=None)

      means, log_vars = tf.split(actions, 2, axis=(-1))
      action_means = (action_spec.maximum + action_spec.minimum) / 2.0
      action_magnitudes = (action_spec.maximum - action_spec.minimum) / 2.0
      means = action_means + action_magnitudes * tf.nn.tanh(means)

      log_vars = 6 * tf.nn.tanh(log_vars) - 3.0  # between -9 and +3

    return tf.concat([means, log_vars], (-1))
