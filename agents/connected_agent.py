import tensorflow as tf
slim = tf.contrib.slim
import gin.tf


def flatten(x):
  """Flattens the temporal component of x into the feature dimension.
  
  Args:
    x: a tensor of shape [batch_sims, T, num_features] to be flattened.
  """
  return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2]])


def concat_and_batch(x, y):   
  """Concatenates x and y along the feature axis and
  Flattens the temporal component of the result into the batch dimension.
  
  Args:
    x: a tensor of shape [batch_sims, T, num_features] to be merged.
    y: a tensor of shape [batch_sims, T, num_features] to be merged.
  """ 
  z = tf.concat([
    x, tf.tile(y[:, tf.newaxis, :], [1, tf.shape(x)[1], 1])], 2)
  return tf.reshape(
    z, [tf.shape(x)[0] * tf.shape(x)[1], tf.shape(z)[2]])


def unbatch(x, size):    
  """Expands the temporal dimension of x.
  
  Args:
    x: a tensor of shape [batch_sime * size, num_features] to be unbatched.
  """
  return tf.reshape(x, [tf.shape(x)[0] // size, size, tf.shape(x)[1]])


@gin.configurable
class ConnectedAgent(object):
  """An RL agent that learns using the Connected Policy algorithm."""

  def __init__(self,
               upper_agent,
               lower_agent,
               max_horizon=None,
               dynamics_function=None):
    """Constructs a Connected Policy agent.

    Args:
      upper_agent: a pointer to an existing upper level (meta) agent.
      lower_agent: a pointer to an existing lower level agent.
      max_horizon: an integer (optional) for max number of q inputs in time.
      dynamics_function: a TF function (optional) that predicts future states given (s_t, a_t).
    """
    self.upper_agent = upper_agent
    self.lower_agent = lower_agent
    self.max_horizon = max_horizon
    self.dynamics_function = dynamics_function

  def unroll_dynamics(self, initial_states, upper_actions, T, stop_gradient=False):
    """Returns the output of the critic network.

    Args:
      initial_states: A [batch_size, num_state_dims] tensor representing a batch
        of initial states, without the environment goal.
      upper_actions: A [batch_size, num_action_dims] tensor representing a batch
        of upper level actions.
      T: An int32 Tensor, the length to predict states over.
    Returns:
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
      lower_actions: A [batch_size, T, num_action_dims] tensor representing a batch
        of lower level actions.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    def predict_body(iteration, s_t, state_array, action_array, lower_goal, time):
      a_t = self.lower_agent.actor_net(tf.concat([s_t, lower_goal], 1), stop_gradient=stop_gradient)
      return (tf.add(iteration, 1),
              self.dynamics_function(s_t, a_t),
              state_array.write(iteration, s_t),
              action_array.write(iteration, a_t),
              lower_goal,
              time)
    prediction = tf.while_loop(
      lambda iteration, s_t, state_array, action_array, lower_goal, time: tf.less(iteration, time),
      predict_body, [
        tf.constant(0),
        initial_states,
        tf.TensorArray(tf.float32, T),
        tf.TensorArray(tf.float32, T),
        upper_actions,
        T])
    return (tf.transpose(prediction[2].stack(), [0, 1]),
            tf.transpose(prediction[3].stack(), [0, 1]))

  def critic_net(self, states, actions, for_critic_loss=False, 
                 lower_states=None, lower_actions=None):
    """Returns the output of the critic network.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, num_action_dims] tensor representing a batch
        of upper level actions.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
      lower_actions: A [batch_size, T, num_action_dims] tensor representing a batch
        of lower level actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    if self.max_horizon is not None:
      lower_states = lower_states[:, :self.max_horizon, :]
      lower_actions = lower_actions[:, :self.max_horizon, :]
    lower_states = tf.concat([flatten(lower_states), states], 1)
    lower_actions = flatten(lower_actions)
    return self.upper_agent._critic_net(
      lower_states, lower_actions, for_critic_loss=for_critic_loss)

  def target_critic_net(self, states, actions, for_critic_loss=False, 
                        lower_states=None, lower_actions=None):
    """Returns the output of the target critic network.

    The target network is used to compute stable targets for training.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, num_action_dims] tensor representing a batch
        of actions.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
      lower_actions: A [batch_size, T, num_action_dims] tensor representing a batch
        of lower level actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    if self.max_horizon is not None:
      lower_states = lower_states[:, :self.max_horizon, :]
      lower_actions = lower_actions[:, :self.max_horizon, :]
    lower_states = tf.concat([flatten(lower_states), states], 1)
    lower_actions = flatten(lower_actions)
    return tf.stop_gradient(self.upper_agent._target_critic_net(
      lower_states, lower_actions, for_critic_loss=for_critic_loss))

  def value_net(self, states, for_critic_loss=False, lower_states=None):
    """Returns the output of the critic evaluated with the actor.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_actions = self.upper_agent.actor_net(states, stop_gradients=True)
    lower_actions = unbatch(self.lower_agent.actor_net(
        concat_and_batch(lower_states, upper_actions), stop_gradients=True), tf.shape(lower_states)[1])
    return self.critic_net(
      states, upper_actions, for_critic_loss=for_critic_loss, 
      lower_states=lower_states, lower_actions=lower_actions)

  def target_value_net(self, states, for_critic_loss=False, lower_states=None):
    """Returns the output of the target critic evaluated with the target actor.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_actions = self.upper_agent.target_actor_net(states)
    lower_actions = unbatch(self.lower_agent.target_actor_net(
        concat_and_batch(lower_states, upper_actions)), tf.shape(lower_states)[1])
    return self.target_critic_net(
      states, upper_actions, for_critic_loss=for_critic_loss,
      lower_states=lower_states, lower_actions=lower_actions)

  def critic_loss(self, states, actions, rewards, discounts,
                  next_states, lower_states=None, 
                  lower_actions=None, lower_next_states=None):
    """Computes a loss for training the critic network.
    The loss is the mean squared error between the Q value predictions of the
    critic and Q values estimated using TD-lambda.
    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, num_action_dims] tensor representing a batch
        of actions.
      rewards: A [batch_size, ...] tensor representing a batch of rewards,
        broadcastable to the critic net output.
      discounts: A [batch_size, ...] tensor representing a batch of discounts,
        broadcastable to the critic net output.
      next_states: A [batch_size, num_state_dims] tensor representing a batch
        of next states.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
      lower_actions: A [batch_size, T, num_action_dims] tensor representing a batch
        of lower level actions.
      lower_next_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
    Returns:
      A rank-0 tensor representing the critic loss.
    Raises:
      ValueError: If any of the inputs do not have the expected dimensions, or
        if their batch_sizes do not match.
    """
    self._validate_states(states)
    self._validate_actions(actions)
    self._validate_states(next_states)

    target_q_values = self.target_value_net(next_states, for_critic_loss=True, 
                                            lower_states=lower_next_states)
    td_targets = target_q_values * discounts + rewards
    if self._target_q_clipping is not None:
      td_targets = tf.clip_by_value(td_targets, self._target_q_clipping[0],
                                    self._target_q_clipping[1])

    q_values = self.critic_net(states, actions, for_critic_loss=True, 
                               lower_states=lower_states, lower_actions=lower_actions)

    td_errors = td_targets - q_values
    if self._debug_summaries:
      gen_debug_td_error_summaries(
          target_q_values, q_values, td_targets, td_errors)

    loss = self._td_errors_loss(td_targets, q_values)

    if self._residual_phi > 0.0:  # compute residual gradient loss
      residual_q_values = self.value_net(next_states, for_critic_loss=True, 
                                         lower_states=lower_next_states)
      residual_td_targets = residual_q_values * discounts + rewards
      if self._target_q_clipping is not None:
        residual_td_targets = tf.clip_by_value(residual_td_targets,
                                               self._target_q_clipping[0],
                                               self._target_q_clipping[1])
      residual_td_errors = residual_td_targets - q_values
      residual_loss = self._td_errors_loss(
          residual_td_targets, residual_q_values)
      loss = (loss * (1.0 - self._residual_phi) +
              residual_loss * self._residual_phi)
    return loss

  def actor_loss(self, states, lower_states=None):
    """Computes a loss for training the actor network.
    Note that output does not represent an actual loss. It is called a loss only
    in the sense that its gradient w.r.t. the actor network weights is the
    correct gradient for training the actor network,
    i.e. dloss/dweights = (dq/da)*(da/dweights)
    which is the gradient used in Algorithm 1 of Lilicrap et al.
    Args:
      states: A [batch_size, t, num_state_dims] tensor representing a batch
        of states.
      lower_states: A [batch_size, T, num_state_dims] tensor representing a batch
        of lower level states.
    Returns:
      A rank-0 tensor representing the actor loss.
    Raises:
      ValueError: If `states` does not have the expected dimensions.
    """
    self.upper_agent._validate_states(states)

    upper_actions = self.upper_agent.actor_net(
        states, stop_gradients=False)
    lower_actions = unbatch(self.lower_agent.actor_net(
        concat_and_batch(lower_states, upper_actions), stop_gradients=False), tf.shape(lower_states)[1])
    critic_values = self.critic_net(
      states, upper_actions, for_critic_loss=False,
      lower_states=lower_states, lower_actions=lower_actions)

    q_values = self.critic_function(critic_values, lower_states)
    dqda = tf.gradients([q_values], [upper_actions])[0]
    dqda_unclipped = dqda
    if self._dqda_clipping > 0:
      dqda = tf.clip_by_value(dqda, -self._dqda_clipping, self._dqda_clipping)

    actions_norm = tf.norm(upper_actions)
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
    return slim.losses.mean_squared_error(tf.stop_gradient(dqda + upper_actions),
                                          upper_actions,
                                          scope='actor_loss') + actions_norm

  def __getattr__(self, attr):
    """A helper function to make this class a wrapper class."""
    if attr == "upper_agent":
      return self.__dict__["upper_agent"]
    if attr == "lower_agent":
      return self.__dict__["lower_agent"]
    else:
      return getattr(self.__dict__["upper_agent"], attr)

  def __setattr__(self, attr, value):
    """A helper function to make this class a wrapper class."""
    if attr == "upper_agent":
      self.__dict__["upper_agent"] = value
    if attr == "lower_agent":
      self.__dict__["lower_agent"] = value
    else:
      setattr(self.__dict__["upper_agent"], attr, value)

  def __getstate__(self):
    """A helper function to make this class a wrapper class."""
    return self.__dict__

  def __setstate__(self, state):
    """A helper function to make this class a wrapper class."""
    self.__dict__.update(state)
