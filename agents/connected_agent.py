import tensorflow as tf
slim = tf.contrib.slim
import gin.tf


@gin.configurable
class ConnectedAgent(object):
  """An RL agent that learns using the Connected Policy algorithm."""

  def __init__(self,
               upper_agent,
               lower_agent):
    """Constructs a Connected Policy agent.

    Args:
      upper_agent: a pointer to an existing upper level (meta) agent.
      lower_agent: a pointer to an existing lower level agent.
    """
    self.upper_agent = upper_agent
    self.lower_agent = lower_agent

  def __getattr__(self, attr):
    if attr == "upper_agent":
      return self.__dict__["upper_agent"]
    if attr == "lower_agent":
      return self.__dict__["lower_agent"]
    else:
      return getattr(self.__dict__["upper_agent"], attr)

  def __setattr__(self, attr, value):
    if attr == "upper_agent":
      self.__dict__["upper_agent"] = value
    if attr == "lower_agent":
      self.__dict__["lower_agent"] = value
    else:
      setattr(self.__dict__["upper_agent"], attr, value)

  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__.update(state)

  def critic_net(self, states, actions, for_critic_loss=False):
    """Returns the output of the critic network.

    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, t, num_action_dims] tensor representing a batch
        of upper level actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    states = tf.reshape(states, [tf.shape(states)[0], tf.shape(states)[1] * tf.shape(states)[2]])
    actions = tf.reshape(actions, [tf.shape(actions)[0], tf.shape(actions)[1] * tf.shape(actions)[2]])
    return self.upper_agent.critic_net(
      states, actions, for_critic_loss=for_critic_loss)

  def target_critic_net(self, states, actions, for_critic_loss=False):
    """Returns the output of the target critic network.

    The target network is used to compute stable targets for training.

    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, t, num_action_dims] tensor representing a batch
        of actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    states = tf.reshape(states, [tf.shape(states)[0], tf.shape(states)[1] * tf.shape(states)[2]])
    actions = tf.reshape(actions, [tf.shape(actions)[0], tf.shape(actions)[1] * tf.shape(actions)[2]])
    return self.upper_agent.target_critic_net(
      states, actions, for_critic_loss=for_critic_loss)

  def value_net(self, states, for_critic_loss=False):
    """Returns the output of the critic evaluated with the actor.

    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_level_actions = self.upper_agent.actor_net(
        states[:, 0, :], stop_gradients=True)
    lower_level_inputs = tf.concat([
      states, 
      tf.tile(upper_level_actions[:, tf.newaxis, :], [1, tf.shape(states)[1], 1])], 2)
    lower_level_inputs = tf.reshape(
      lower_level_inputs, [tf.shape(states)[0] * tf.shape(states)[1], tf.shape(lower_level_inputs)[2]])
    lower_level_actions = self.lower_agent.actor_net(
        lower_level_inputs, stop_gradients=True)
    lower_level_actions = tf.reshape(
      lower_level_actions, [tf.shape(states)[0], tf.shape(states)[1], tf.shape(lower_level_actions)[1]])
    return self.critic_net(
      states, lower_level_actions, for_critic_loss=for_critic_loss)

  def target_value_net(self, states, for_critic_loss=False):
    """Returns the output of the target critic evaluated with the target actor.

    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_level_actions = self.upper_agent.target_actor_net(
        states[:, 0, :], stop_gradients=True)
    lower_level_inputs = tf.concat([
      states, 
      tf.tile(upper_level_actions[:, tf.newaxis, :], [1, tf.shape(states)[1], 1])], 2)
    lower_level_inputs = tf.reshape(
      lower_level_inputs, [tf.shape(states)[0] * tf.shape(states)[1], tf.shape(lower_level_inputs)[2]])
    lower_level_actions = self.lower_agent.target_actor_net(
        lower_level_inputs, stop_gradients=True)
    lower_level_actions = tf.reshape(
      lower_level_actions, [tf.shape(states)[0], tf.shape(states)[1], tf.shape(lower_level_actions)[1]])
    return self.target_critic_net(
      states, lower_level_actions, for_critic_loss=for_critic_loss)

  def actor_loss(self, states):
    """Computes a loss for training the actor network.
    Note that output does not represent an actual loss. It is called a loss only
    in the sense that its gradient w.r.t. the actor network weights is the
    correct gradient for training the actor network,
    i.e. dloss/dweights = (dq/da)*(da/dweights)
    which is the gradient used in Algorithm 1 of Lilicrap et al.
    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
    Returns:
      A rank-0 tensor representing the actor loss.
    Raises:
      ValueError: If `states` does not have the expected dimensions.
    """
    self.lower_agent._validate_states(states[:, 0, :])
    self.upper_agent._validate_states(states[:, 0, :])

    upper_level_actions = self.upper_agent.actor_net(
        states[:, 0, :], stop_gradients=False)
    lower_level_inputs = tf.concat([
      states,
      tf.tile(upper_level_actions[:, tf.newaxis, :], [1, tf.shape(states)[1], 1])], 2)
    lower_level_inputs = tf.reshape(
      lower_level_inputs, [tf.shape(states)[0] * tf.shape(states)[1], tf.shape(lower_level_inputs)[2]])
    lower_level_actions = self.lower_agent.actor_net(
        lower_level_inputs, stop_gradients=False)
    lower_level_actions = tf.reshape(
      lower_level_actions, [tf.shape(states)[0], tf.shape(states)[1], tf.shape(lower_level_actions)[1]])
    critic_values = self.critic_net(
      states, lower_level_actions, for_critic_loss=False)

    q_values = self.critic_function(critic_values, states)
    dqda = tf.gradients([q_values], [upper_level_actions])[0]
    dqda_unclipped = dqda
    if self._dqda_clipping > 0:
      dqda = tf.clip_by_value(dqda, -self._dqda_clipping, self._dqda_clipping)

    actions_norm = tf.norm(upper_level_actions)
    if self._debug_summaries:
      with tf.name_scope('dqda'):
        tf.summary.scalar('actions_norm', actions_norm)
        tf.summary.histogram('dqda', dqda)
        tf.summary.histogram('dqda_unclipped', dqda_unclipped)
        tf.summary.histogram('actions', actions)
        for a in range(self._num_action_dims):
          tf.summary.histogram('dqda_unclipped_%d' % a, dqda_unclipped[:, a])
          tf.summary.histogram('dqda_%d' % a, dqda[:, a])

    actions_norm *= self._actions_regularizer
    return slim.losses.mean_squared_error(tf.stop_gradient(dqda + upper_level_actions),
                                          upper_level_actions,
                                          scope='actor_loss') + actions_norm
