import tensorflow as tf
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
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, num_action_dims] tensor representing a batch
        of upper level actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    lower_level_inputs = tf.concat([states, actions], 1)
    lower_level_actions = self.lower_agent.actor_net(
        lower_level_inputs, stop_gradients=False)
    return self.upper_agent.critic_net(
        states, lower_level_actions, for_critic_loss=for_critic_loss)

  def target_critic_net(self, states, actions, for_critic_loss=False):
    """Returns the output of the target critic network.

    The target network is used to compute stable targets for training.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      actions: A [batch_size, num_action_dims] tensor representing a batch
        of actions.
    Returns:
      q values: A [batch_size] tensor of q values.
    Raises:
      ValueError: If `states` or `actions' do not have the expected dimensions.
    """
    lower_level_inputs = tf.concat([states, actions], 1)
    lower_level_actions = self.lower_agent.target_actor_net(
        lower_level_inputs, stop_gradients=True)
    return self.upper_agent.target_critic_net(
        states, lower_level_actions, for_critic_loss=for_critic_loss)

  def value_net(self, states, for_critic_loss=False):
    """Returns the output of the critic evaluated with the actor.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_level_actions = self.lower_agent.actor_net(
        states, stop_gradients=False)
    lower_level_inputs = tf.concat([states, upper_level_actions], 1)
    lower_level_actions = self.lower_agent.actor_net(
        lower_level_inputs, stop_gradients=False)
    return self.upper_agent.critic_net(
        states, lower_level_actions, for_critic_loss=for_critic_loss)

  def target_value_net(self, states, for_critic_loss=False):
    """Returns the output of the target critic evaluated with the target actor.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
    Returns:
      q values: A [batch_size] tensor of q values.
    """
    upper_level_actions = self.lower_agent.target_actor_net(
        states, stop_gradients=False)
    lower_level_inputs = tf.concat([states, upper_level_actions], 1)
    lower_level_actions = self.lower_agent.target_actor_net(
        lower_level_inputs, stop_gradients=False)
    return self.upper_agent.target_critic_net(
        states, lower_level_actions, for_critic_loss=for_critic_loss)
