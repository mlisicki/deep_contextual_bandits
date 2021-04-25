import argparse
import numpy as np
import sys
sys.path.insert(0, '../../')
import multiprocessing
from multiprocessing.connection import Listener
import tensorflow as tf

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.algorithms.uniform_sampling import UniformSampling

from absl import app
from absl import flags
FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
#flags.DEFINE_bool('undefok', True,'_')
flags.DEFINE_string('logdir', '../../outputs', 'Base directory to save output')

class IPCBandit(object):
    def __init__(self, connection):
        self.connection = connection
        self.connection.send({'query': 'num_arms'})
        self.n_arms = int(self.connection.recv())
        print("Number of arms: {}".format(self.n_arms))
        self.connection.send({'query': 'context_dims'})
        self.context_dims = int(self.connection.recv())
        print("Context dimensions: {}".format(self.context_dims))
        self.step = 0

    def reset(self):
        self.step = 0
        # get first context from server
        self.connection.send({'query': 'context'})
        context = self.connecton.recv()
        print("Received context: {}".format(context))
        return np.array(context)

    def pull(self, arm):
        self.connection.send({'arm': arm})
        self.connection.send({'query': 'reward'})
        msg = self.connection.recv()
        if msg == "close":
            return None
        reward = msg
        print("Received reward: {}".format(reward))
        self.step += 1
        return reward

    def context(self):
        self.connection.send({'query': 'context'})
        msg = self.connection.recv()
        if msg == "close":
            return None
        context = msg
        print("Received context: {}".format(context))
        return np.array(context)

    def optimal(self):
        # Many of the bandits I have in my library don't have access to an oracle
        raise NotImplementedError()

    @property
    def context_dim(self):
        return self.context_dims

    @property
    def num_actions(self):
        return self.n_arms

    @property
    def number_contexts(self):
        return 1

    def __repr__(self):
        return "IPC Bandit"

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Deep contextual bandit server")

    parser.add_argument('--algorithm', type=str, default='neurallinear',
                        help='Deep contextual bandits algorithm to run: {uniform | bnn | neurallinear | linear | bootrms | paramnoise}')
    parser.add_argument('--ipc_port', type=int, default=6000, help='Port to use for IPC.')

    opts, unknown = parser.parse_known_args(args)
    if unknown:
        print("Unknown args: {}".format(unknown))

    return opts

def main(argv):
    opts = get_options()
    print("Parameters: {}".format(opts))
    address = ('localhost', opts.ipc_port)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'bandit')
    conn = listener.accept()
    multiprocessing.current_process().authkey = b'bandit'
    print('connection accepted from', listener.last_accepted)


    # Create contextual bandit
    bandit = IPCBandit(conn)

    if opts.algorithm == "uniform":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions)
        policy = UniformSampling('Uniform Sampling', policy_parameters)

    elif opts.algorithm == "linear":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                     context_dim=bandit.context_dim,
                                                     a0=6,
                                                     b0=6,
                                                     lambda_prior=0.25,
                                                     initial_pulls=2)
        policy = LinearFullPosteriorSampling('LinFullPost', policy_parameters)

    elif opts.algorithm == "rms":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                  context_dim=bandit.context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[50],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=2,
                                                  optimizer='RMS',
                                                  reset_lr=True,
                                                  lr_decay_rate=0.5,
                                                  training_freq=50,
                                                  training_epochs=100,
                                                  p=0.95,
                                                  q=3)
        policy = PosteriorBNNSampling('RMS', policy_parameters, 'RMSProp')

    elif opts.algorithm == "bootrms":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                  context_dim=bandit.context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[50],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=2,
                                                  optimizer='RMS',
                                                  reset_lr=True,
                                                  lr_decay_rate=0.5,
                                                  training_freq=50,
                                                  training_epochs=100,
                                                  p=0.95,
                                                  q=3)
        policy =BootstrappedBNNSampling('BootRMS', policy_parameters)

    elif opts.algorithm == "dropout":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                      context_dim=bandit.context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[50],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=0.1,
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=2,
                                                      optimizer='RMS',
                                                      reset_lr=True,
                                                      lr_decay_rate=0.5,
                                                      training_freq=50,
                                                      training_epochs=100,
                                                      use_dropout=True,
                                                      keep_prob=0.80)
        policy = PosteriorBNNSampling('Dropout', policy_parameters, 'RMSProp')

    elif opts.algorithm == "bbb":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                  context_dim=bandit.context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[50],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=2,
                                                  optimizer='RMS',
                                                  use_sigma_exp_transform=True,
                                                  cleared_times_trained=10,
                                                  initial_training_steps=100,
                                                  noise_sigma=0.1,
                                                  reset_lr=False,
                                                  training_freq=50,
                                                  training_epochs=100)
        policy = PosteriorBNNSampling('BBB', policy_parameters, 'Variational')

    elif opts.algorithm == "neurallinear":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                      context_dim=bandit.context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[50],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=0.1,
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=2,
                                                      reset_lr=True,
                                                      lr_decay_rate=0.5,
                                                      training_freq=1,
                                                      training_freq_network=50,
                                                      training_epochs=100,
                                                      a0=6,
                                                      b0=6,
                                                      lambda_prior=0.25)
        policy = NeuralLinearPosteriorSampling('NeuralLinear', policy_parameters)

    elif opts.algorithm == "neurallinear2":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                       context_dim=bandit.context_dim,
                                                       init_scale=0.3,
                                                       activation=tf.nn.relu,
                                                       layer_sizes=[50],
                                                       batch_size=512,
                                                       activate_decay=True,
                                                       initial_lr=0.1,
                                                       max_grad_norm=5.0,
                                                       show_training=False,
                                                       freq_summary=1000,
                                                       buffer_s=-1,
                                                       initial_pulls=2,
                                                       reset_lr=True,
                                                       lr_decay_rate=0.5,
                                                       training_freq=10,
                                                       training_freq_network=50,
                                                       training_epochs=100,
                                                       a0=6,
                                                       b0=6,
                                                       lambda_prior=0.25)
        policy = NeuralLinearPosteriorSampling('NeuralLinear2', policy_parameters)

    elif opts.algorithm == "pnoise":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                     context_dim=bandit.context_dim,
                                                     init_scale=0.3,
                                                     activation=tf.nn.relu,
                                                     layer_sizes=[50],
                                                     batch_size=512,
                                                     activate_decay=True,
                                                     initial_lr=0.1,
                                                     max_grad_norm=5.0,
                                                     show_training=False,
                                                     freq_summary=1000,
                                                     buffer_s=-1,
                                                     initial_pulls=2,
                                                     optimizer='RMS',
                                                     reset_lr=True,
                                                     lr_decay_rate=0.5,
                                                     training_freq=50,
                                                     training_epochs=100,
                                                     noise_std=0.05,
                                                     eps=0.1,
                                                     d_samples=300,
                                                     )
        policy = ParameterNoiseSampling('ParamNoise', policy_parameters)

    elif opts.algorithm == "alpha_div":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                        context_dim=bandit.context_dim,
                                                        init_scale=0.3,
                                                        activation=tf.nn.relu,
                                                        layer_sizes=[50],
                                                        batch_size=512,
                                                        activate_decay=True,
                                                        initial_lr=0.1,
                                                        max_grad_norm=5.0,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        buffer_s=-1,
                                                        initial_pulls=2,
                                                        optimizer='RMS',
                                                        use_sigma_exp_transform=True,
                                                        cleared_times_trained=10,
                                                        initial_training_steps=100,
                                                        noise_sigma=0.1,
                                                        reset_lr=False,
                                                        training_freq=50,
                                                        training_epochs=100,
                                                        alpha=1.0,
                                                        k=20,
                                                        prior_variance=0.1)
        policy = PosteriorBNNSampling('BBAlphaDiv', policy_parameters, 'AlphaDiv')

    elif opts.algorithm == "gp":
        policy_parameters = tf.contrib.training.HParams(num_actions=bandit.num_actions,
                                                        num_outputs=bandit.num_actions,
                                                        context_dim=bandit.context_dim,
                                                        reset_lr=False,
                                                        learn_embeddings=True,
                                                        max_num_points=1000,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        batch_size=512,
                                                        keep_fixed_after_max_obs=True,
                                                        training_freq=50,
                                                        initial_pulls=2,
                                                        training_epochs=100,
                                                        lr=0.01,
                                                        buffer_s=-1,
                                                        initial_lr=0.001,
                                                        lr_decay_rate=0.0,
                                                        optimizer='RMS',
                                                        task_latent_dim=5,
                                                        activate_decay=False)
        policy = PosteriorBNNSampling('MultitaskGP', policy_parameters, 'GP')

    else:
        raise Exception("Misspecified bandit algorithm.")

    print(policy)
    # Run the contextual bandit process
    while True:
        context = bandit.context()
        if context is None:
            break
        action = policy.action(context)
        reward = bandit.pull(action)
        if reward is None:
            break

        policy.update(context, action, reward)

    conn.close()
    listener.close()

if __name__ == '__main__':
    # Run the main script
    app.run(main, argv=[sys.argv[0]])
