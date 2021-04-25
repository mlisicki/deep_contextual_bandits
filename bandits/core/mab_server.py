import argparse
import numpy as np
import sys
sys.path.insert(0, '../../')
import multiprocessing
from multiprocessing.connection import Listener
import tensorflow as tf

from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling

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

def main(_):
    opts = get_options()
    print("Parameters: {}".format(opts))
    address = ('localhost', opts.ipc_port)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'bandit')
    conn = listener.accept()
    multiprocessing.current_process().authkey = b'bandit'
    print('connection accepted from', listener.last_accepted)


    # Create contextual bandit
    bandit = IPCBandit(conn)

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
    # This is to ignore the error from abs flags when used in conjunction with argparse
    if len(sys.argv)>1:
        args = sys.argv[1:]
    else:
        args = []
    sys.argv = [sys.argv[0]]+['--undefok']+args

    # Run the main script
    app.run(main)
