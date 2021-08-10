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

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
from absl import app
from absl import flags
import numpy as np
import os
import tensorflow as tf

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling

# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')
flags.DEFINE_string(
    'financial_data',
    os.path.join(base_route, data_route, 'raw_stock_contexts'),
    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string(
    'statlog_data',
    os.path.join(base_route, data_route, 'shuttle.trn'),
    'Directory where Statlog data is stored.')
flags.DEFINE_string(
    'adult_data',
    os.path.join(base_route, data_route, 'adult.full'),
    'Directory where Adult data is stored.')
flags.DEFINE_string(
    'covertype_data',
    os.path.join(base_route, data_route, 'covtype.data'),
    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data',
    os.path.join(base_route, data_route, 'USCensus1990.data.txt'),
    'Directory where Census data is stored.')


def sample_data(data_type, num_contexts=None):
    """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

    if data_type == 'linear':
        # Create linear dataset
        num_actions = 8
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                    num_actions, sigma=noise_stds)
        opt_rewards, opt_actions = opt_linear
    elif data_type == 'sparse_linear':
        # Create sparse linear dataset
        num_actions = 7
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        num_nnz_dims = int(context_dim / 3.0)
        dataset, _, opt_sparse_linear = sample_sparse_linear_data(
            num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
        opt_rewards, opt_actions = opt_sparse_linear
    elif data_type == 'mushroom':
        # Create mushroom dataset
        num_actions = 2
        context_dim = 117
        file_name = FLAGS.mushroom_data
        dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
        opt_rewards, opt_actions = opt_mushroom
    elif data_type == 'financial':
        num_actions = 8
        context_dim = 21
        num_contexts = min(3713, num_contexts)
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        file_name = FLAGS.financial_data
        dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                                   num_actions, num_contexts,
                                                   noise_stds, shuffle_rows=True)
        opt_rewards, opt_actions = opt_financial
    elif data_type == 'jester':
        num_actions = 8
        context_dim = 32
        num_contexts = min(19181, num_contexts)
        file_name = FLAGS.jester_data
        dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                                 num_actions, num_contexts,
                                                 shuffle_rows=True,
                                                 shuffle_cols=True)
        opt_rewards, opt_actions = opt_jester
    elif data_type == 'statlog':
        file_name = FLAGS.statlog_data
        num_actions = 7
        num_contexts = min(43500, num_contexts)
        sampled_vals = sample_statlog_data(file_name, num_contexts,
                                           shuffle_rows=True)
        contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
        dataset = np.hstack((contexts, rewards))
        context_dim = contexts.shape[1]
    elif data_type == 'adult':
        file_name = FLAGS.adult_data
        num_actions = 14
        num_contexts = min(45222, num_contexts)
        sampled_vals = sample_adult_data(file_name, num_contexts,
                                         shuffle_rows=True)
        contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
        dataset = np.hstack((contexts, rewards))
        context_dim = contexts.shape[1]
    elif data_type == 'covertype':
        file_name = FLAGS.covertype_data
        num_actions = 7
        num_contexts = min(150000, num_contexts)
        sampled_vals = sample_covertype_data(file_name, num_contexts,
                                             shuffle_rows=True)
        contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
        dataset = np.hstack((contexts, rewards))
        context_dim = contexts.shape[1]
    elif data_type == 'census':
        file_name = FLAGS.census_data
        num_actions = 9
        num_contexts = min(150000, num_contexts)
        sampled_vals = sample_census_data(file_name, num_contexts,
                                          shuffle_rows=True)
        contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
        dataset = np.hstack((contexts, rewards))
        context_dim = contexts.shape[1]
    elif data_type == 'wheel':
        delta = 0.95
        num_actions = 5
        context_dim = 2
        mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
        std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
        mu_large = 50
        std_large = 0.01
        dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                      mean_v, std_v,
                                                      mu_large, std_large)
        opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim


def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
    """Displays summary statistics of the performance of each algorithm."""

    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('{} bandit completed after {} seconds.'.format(
        name, time.time() - t_init))
    print('---------------------------------------------------')

    performance_pairs = []
    for j, a in enumerate(algos):
        performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
    performance_pairs = sorted(performance_pairs,
                               key=lambda elt: elt[1],
                               reverse=True)
    for i, (name, reward) in enumerate(performance_pairs):
        print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

    print('---------------------------------------------------')
    print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
    print('Frequency of optimal actions (action, frequency):')
    print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
    print('---------------------------------------------------')
    print('---------------------------------------------------')


def main(_):
    # Problem parameters
    num_contexts = 200000
    token = np.random.randint(9999)

    # Data type in {linear, sparse_linear, mushroom, financial, jester,
    #                 statlog, adult, covertype, census, wheel}
    for data_type in ['linear', 'sparse_linear', 'mushroom', 'financial', 'jester',
                      'statlog', 'adult', 'covertype', 'census', 'wheel']:

        # Create dataset
        sampled_vals = sample_data(data_type, num_contexts)
        dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

        # Uniform and Fixed
        hparams = tf.contrib.training.HParams(num_actions=num_actions)

        # AlphaDivergence (1)
        hparams_alpha_div = tf.contrib.training.HParams(num_actions=num_actions,
                                                        context_dim=context_dim,
                                                        init_scale='test',  # This doesn't seem to be used
                                                        activation=tf.nn.relu,
                                                        layer_sizes=[100, 100],
                                                        # all NN are based on the same architecture: 100,100 relu
                                                        batch_size=512,
                                                        activate_decay=True,  # Use learning rate decay
                                                        initial_lr=1,
                                                        # I'm setting this as in RMS3, as they don't reset LR in this example. Not sure if that's how Riquelme did it though.
                                                        max_grad_norm=5.0,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        buffer_s=-1,
                                                        # paper states they decided not to use data buffer after initial experimentation
                                                        initial_pulls=3,
                                                        # for all models we pull each arm 3 times in round robin before we start
                                                        optimizer='test',  # this doesn't seem to be used
                                                        use_sigma_exp_transform=True,
                                                        cleared_times_trained=100,
                                                        initial_training_steps=10000,
                                                        # Linear decay of training steps. Initial t_s=10000, then decay for 'cleared' number of steps (each time by 100), until it reaches 100
                                                        noise_sigma=0.1,
                                                        reset_lr=False,
                                                        # Don't reset learning rate on each bandit model retraining step
                                                        training_freq=20,
                                                        training_epochs=100,  # t_s
                                                        alpha=0.5,  # main alpha-divergence param
                                                        # k=10, # I think k is given by num_mc_nn_samples
                                                        num_mc_nn_samples=10,
                                                        prior_variance=0.1)  # prior variance is sigma_0^2

        # BBB
        hparams_bbb = tf.contrib.training.HParams(num_actions=num_actions,
                                                  context_dim=context_dim,
                                                  init_scale='test',
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[100, 100],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=1,
                                                  # I'm setting this to 1 whenever we don't reset LR. This follows settings for RMS3, but I'm not sure if that's correct globally.
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=3,
                                                  optimizer='test',
                                                  use_sigma_exp_transform=True,
                                                  cleared_times_trained=100,
                                                  initial_training_steps=10000,
                                                  noise_sigma=0.1,
                                                  reset_lr=False,
                                                  training_freq=20,  # specified in table2's description
                                                  training_epochs=100)

        # Bootsrapped NN
        hparams_bootrms = tf.contrib.training.HParams(num_actions=num_actions,
                                                      context_dim=context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[100,100],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=1.0,
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=3,
                                                      optimizer='RMS',
                                                      reset_lr=False,
                                                      lr_decay_rate=0.5, # Default choice. idk if this is correct. But it is close to RMS3 setting of 0.55
                                                      training_freq=20,
                                                      training_epochs=20,
                                                      p=1.0, # Prob of independently including each datapoint in each model.
                                                      q=10) # Number of models that are independently trained.

        # Dropout
        hparams_dropout = tf.contrib.training.HParams(num_actions=num_actions,
                                                      context_dim=context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[100,100],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=0.1, # idk if this is correct
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=3,
                                                      optimizer='RMS',
                                                      reset_lr=True,
                                                      lr_decay_rate=0.5,
                                                      training_freq=20,
                                                      training_epochs=20,
                                                      use_dropout=True,
                                                      keep_prob=0.8)

        # GP
        # Hyperparameters optimized internally, using marginal likelihood as a loss function
        # Other hyperparameters are not reported so I'm using dafualts
        hparams_gp = tf.contrib.training.HParams(num_actions=num_actions,
                                                 num_outputs=num_actions,
                                                 context_dim=context_dim,
                                                 reset_lr=False,
                                                 learn_embeddings=True,
                                                 max_num_points=1000,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 batch_size=512,
                                                 keep_fixed_after_max_obs=True,
                                                 training_freq=20,
                                                 initial_pulls=3,
                                                 training_epochs=20,
                                                 lr=0.01,
                                                 buffer_s=-1,
                                                 initial_lr=0.001,
                                                 lr_decay_rate=0.0,
                                                 optimizer='RMS',
                                                 task_latent_dim=5,
                                                 activate_decay=False)
        # Neural Linear
        hparams_nlinear = tf.contrib.training.HParams(num_actions=num_actions,
                                                      context_dim=context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[100,100],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=0.1,
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=3,
                                                      reset_lr=True,
                                                      lr_decay_rate=0.5,
                                                      training_freq=1, #"It makes sense to keep an exact linear regression (as in (1) and (2)) at all times, adding each new data point as soon as it arrives"
                                                      training_freq_network=20,
                                                      training_epochs=20,
                                                      a0=3,
                                                      b0=3,
                                                      lambda_prior=0.25)

        # RMS2
        hparams_rms = tf.contrib.training.HParams(num_actions=num_actions,
                                                  context_dim=context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[100,100],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=3,
                                                  optimizer='RMS',
                                                  reset_lr=True,
                                                  lr_decay_rate=0.5,
                                                  training_freq=20,
                                                  training_epochs=20,
                                                  p='test',
                                                  q='test')

        # SGFS and ConstantSGD not implemented

        # LinFullPost
        hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                                     context_dim=context_dim,
                                                     a0=6,
                                                     b0=6,
                                                     lambda_prior=0.25,
                                                     initial_pulls=3)

        hparams_pnoise = tf.contrib.training.HParams(num_actions=num_actions,
                                                     context_dim=context_dim,
                                                     init_scale=0.3,
                                                     activation=tf.nn.relu,
                                                     layer_sizes=[100,100],
                                                     batch_size=512,
                                                     activate_decay=True,
                                                     initial_lr=0.1,
                                                     max_grad_norm=5.0,
                                                     show_training=False,
                                                     freq_summary=1000,
                                                     buffer_s=-1,
                                                     initial_pulls=3,
                                                     optimizer='RMS',
                                                     reset_lr=True,
                                                     lr_decay_rate=0.5,
                                                     training_freq=20,
                                                     training_epochs=20,
                                                     noise_std=0.01,
                                                     eps=0.01,
                                                     d_samples=300,
                                                     layer_norm=True
                                                     )


        algos = [
            PosteriorBNNSampling('BBAlphaDiv', hparams_alpha_div, 'AlphaDiv'),
            PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
            BootstrappedBNNSampling('BootRMS', hparams_bootrms),
            PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
            PosteriorBNNSampling('MultitaskGP', hparams_gp, 'GP'),
            NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
            PosteriorBNNSampling('RMS', hparams_rms, 'RMSProp'),
            LinearFullPosteriorSampling('LinFullPost', hparams_linear),
            ParameterNoiseSampling('ParamNoise', hparams_pnoise),
            UniformSampling('Uniform Sampling', hparams),
        ]
        if data_type == "mushroom":
            algos += [
                FixedPolicySampling('fixed1', [0.75, 0.25], hparams),
                FixedPolicySampling('fixed2', [0.25, 0.75], hparams),
            ]

        try:
            favorite_color = {"lion": "yellow", "kitty": "red"}

            pkl.dump(favorite_color, open("test.p", "wb"))

            pkl.dump({'desc': 'All the base models',
                      'models': [alg.name for alg in algos], 'dataset': data_type,
                      'hparams': [alg.hparams for alg in algos],
                      'actions': None, 'rewards': None},
                     open("/home/mlisicki/project/mlisicki/deep_contextual_bandits/experiment_all_base_methods_{}_{}.pkl".format(str(token),data_type), "wb"))

            # Run contextual bandit problem
            t_init = time.time()
            results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
            h_actions, h_rewards = results

            pkl.dump({'desc': 'All the base models',
                      'models': [alg.name for alg in algos], 'dataset': data_type,
                      'hparams': [alg.hparams for alg in algos],
                      'actions': h_actions, 'rewards': h_rewards},
                     open("/home/mlisicki/project/mlisicki/deep_contextual_bandits/experiment_all_base_methods_{}_{}.pkl".format(str(token),data_type), "wb"))
            # Display results
            display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)
        except:
            continue

if __name__ == '__main__':
    app.run(main)
