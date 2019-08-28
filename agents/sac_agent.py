import tensorflow as tf
slim = tf.contrib.slim
import gin.tf
from utils import utils
from agents import ddpg_agent
from agents import sac_networks, ddpg_networks


"""An SAC/NAF agent.

Implements the Soft Actor Critic (SAC) algorithm from
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" - Haarnoja et al.
https://arxiv.org/abs/1801.01290, and the Normalized Advantage Functions (NAF)
algorithm "Continuous Deep Q-Learning with Model-based Acceleration" - Gu et al.
https://arxiv.org/pdf/1603.00748.
"""


@gin.configurable
class SacAgent(ddpg_agent.TD3Agent):

    def __init__(self,
        observation_spec,
        action_spec,
        target_entropy=None,
        actor_net=sac_networks.actor_net,
        critic_net=ddpg_networks.critic_net,
        td_errors_loss=tf.losses.huber_loss,
        dqda_clipping=0.,
        actions_regularizer=0.,
        target_q_clipping=None,
        residual_phi=0.0,
        debug_summaries=False
    ):
        ddpg_agent.TD3Agent.__init__(
            self,
            observation_spec,
            action_spec,
            actor_net=actor_net,
            critic_net=critic_net,
            td_errors_loss=td_errors_loss,
            dqda_clipping=dqda_clipping,
            actions_regularizer=actions_regularizer,
            target_q_clipping=target_q_clipping,
            residual_phi=residual_phi,
            debug_summaries=debug_summaries)
        self.target_entropy = float(
            target_entropy if target_entropy is not None else
            -self._action_spec.shape.num_elements())
        self.log_alpha = slim.variable('log_alpha',
                                       shape=[],
                                       initializer=tf.zeros_initializer())

    def get_trainable_actor_vars(self):
        """Returns a list of trainable variables in the actor network.

        Returns:
          A list of trainable variables in the actor network.
        """
        return [self.log_alpha] + slim.get_trainable_variables(
            utils.join_scope(self._scope, self.ACTOR_NET_SCOPE))

    def action(self, state):
        """Returns the next action for the state.

        Args:
          state: A [num_state_dims] tensor representing a state.
        Returns:
          A [num_action_dims] tensor representing the action.
        """
        mean, _log_var, _entropy = self.actor_net_backend(
            self._batch_state(state), stop_gradients=True)
        return mean[0, :]

    @gin.configurable('sac_sample_action')
    def sample_action(self, state, stddev=None):
        """Returns the action for the state with additive noise.

        Args:
          state: A [num_state_dims] tensor representing a state.
          stddev: Unused parameter
        Returns:
          A [num_action_dims] action tensor.
        """
        mean, log_var, _entropy = self.actor_net_backend(
            self._batch_state(state), stop_gradients=True)
        x = mean + tf.random_normal(tf.shape(log_var)) * tf.math.exp(log_var)
        return utils.clip_to_spec(x[0, :], self._action_spec)

    def actor_net_backend(self, states, stop_gradients=False):
        """Returns the output of the actor network.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
          stop_gradients: (boolean) if true, gradients cannot be propogated through
            this operation.
        Returns:
          A [batch_size, num_action_dims] tensor of actions.
          A [batch_size, num_action_dims] tensor of log variances.
          A [batch_size] tensor of entropies.
        Raises:
          ValueError: If `states` does not have the expected dimensions.
        """
        self._validate_states(states)
        means, log_vars = tf.split(self._actor_net(states, self._action_spec), 2, axis=(-1))
        entropies = 0.5 * tf.reduce_sum(log_vars, axis=(-1)) + 0.89908993417
        if stop_gradients:
            means = tf.stop_gradient(means)
            log_vars = tf.stop_gradient(log_vars)
            entropies = tf.stop_gradient(entropies)
        return means, log_vars, entropies

    def actor_net(self, states, stop_gradients=False):
        """Returns the output of the actor network.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
          stop_gradients: (boolean) if true, gradients cannot be propogated through
            this operation.
        Returns:
          A [batch_size, num_action_dims] tensor of actions.
        Raises:
          ValueError: If `states` does not have the expected dimensions.
        """
        return self.actor_net_backend(states, stop_gradients=stop_gradients)

    def target_actor_net_backend(self, states):
        """Returns the output of the target actor network.

        The target network is used to compute stable targets for training.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
        Returns:
          A [batch_size, num_action_dims] tensor of actions.
          A [batch_size, num_action_dims] tensor of log variances.
          A [batch_size] tensor of entropies.
        Raises:
          ValueError: If `states` does not have the expected dimensions.
        """
        self._validate_states(states)
        means, log_vars = tf.split(self._target_actor_net(states, self._action_spec), 2, axis=(-1))
        entropies = 0.5 * tf.reduce_sum(log_vars, axis=(-1)) + 0.89908993417
        means = tf.stop_gradient(means)
        log_vars = tf.stop_gradient(log_vars)
        entropies = tf.stop_gradient(entropies)
        return means, log_vars, entropies

    def target_actor_net(self, states):
        """Returns the output of the target actor network.

        The target network is used to compute stable targets for training.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
        Returns:
          A [batch_size, num_action_dims] tensor of actions.
        Raises:
          ValueError: If `states` does not have the expected dimensions.
        """
        return self.target_actor_net_backend(states)

    def value_net(self, states, for_critic_loss=False):
        """Returns the output of the critic evaluated with the actor.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
        Returns:
          q values: A [batch_size] tensor of q values.
        """
        means, _log_vars, entropies = self.actor_net_backend(states)
        return tf.stop_gradient(tf.exp(self.log_alpha) * entropies) + self.critic_net(
            states, means, for_critic_loss=for_critic_loss)

    def target_value_net(self, states, for_critic_loss=False):
        """Returns the output of the target critic evaluated with the target actor.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
        Returns:
          q values: A [batch_size] tensor of q values.
        """
        means, _log_vars, entropies = self.target_actor_net_backend(states)
        noise = tf.clip_by_value(
            tf.random_normal(tf.shape(means), stddev=0.2), -0.5, 0.5)
        values1, values2 = self.target_critic_net(
            states, means + noise,
            for_critic_loss=for_critic_loss)
        values = tf.stop_gradient(tf.exp(self.log_alpha) * entropies) + tf.minimum(
            values1, values2)
        return values, values

    def actor_loss(self, states):
        """Computes a loss for training the actor network.

        Note that output does not represent an actual loss. It is called a loss only
        in the sense that its gradient w.r.t. the actor network weights is the
        correct gradient for training the actor network,
        i.e. dloss/dweights = (dq/da)*(da/dweights)
        which is the gradient used in Algorithm 1 of Lilicrap et al.

        Args:
          states: A [batch_size, num_state_dims] tensor representing a batch
            of states.
        Returns:
          A rank-0 tensor representing the actor loss.
        Raises:
          ValueError: If `states` does not have the expected dimensions.
        """
        self._validate_states(states)
        means, log_vars, entropies = self.actor_net_backend(states, stop_gradients=False)
        actions = means + tf.random_normal(tf.shape(log_vars)) * tf.math.exp(log_vars)
        critic_values = self.critic_net(states, actions)
        q_values = self.critic_function(critic_values, states)
        dqda = tf.gradients([q_values], [actions])[0]
        dqda_unclipped = dqda
        if self._dqda_clipping > 0:
            dqda = tf.clip_by_value(dqda, -self._dqda_clipping, self._dqda_clipping)

        actions_norm = tf.norm(actions)
        if self._debug_summaries:
            with tf.name_scope('dqda'):
                tf.summary.scalar('actions_norm', actions_norm)
                tf.summary.histogram('dqda', dqda)
                tf.summary.histogram('dqda_unclipped', dqda_unclipped)
                tf.summary.histogram('actions', actions)
                for a in range(self._num_action_dims):
                    tf.summary.histogram('dqda_unclipped_%d' % a, dqda_unclipped[:, a])
                    tf.summary.histogram('dqda_%d' % a, dqda[:, a])

        policy_entropy = tf.reduce_mean(entropies)
        entropy_loss = -tf.stop_gradient(tf.exp(self.log_alpha)) * policy_entropy
        tf.summary.scalar('policy_entropy', policy_entropy)
        tf.summary.scalar('entropy_loss', entropy_loss)

        log_alpha_error = self.log_alpha * (
            tf.stop_gradient(policy_entropy) - self.target_entropy)
        tf.summary.scalar('log_alpha_error', log_alpha_error)
        tf.summary.scalar('log_alpha', self.log_alpha)

        actions_norm *= self._actions_regularizer
        return slim.losses.mean_squared_error(
            tf.stop_gradient(dqda + actions),
            actions,
            scope='actor_loss') + actions_norm + log_alpha_error + entropy_loss
