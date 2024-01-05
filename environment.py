import numpy as np
import contextlib

# Frozen lake environment

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # TODO:
        self.transition_probabilities = np.load('p.npy')

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # TODO:
        return self.transition_probabilities[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if self.lake_flat[state] == '$':
            # reward
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))

lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
if __name__ == '__main__': play(FrozenLake(lake, slip=0.1, max_steps=10))


# Main Function
def main():



    seed = 0

    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    #
    # print('# Model-based algorithms')
    #
    # print('')
    #
    # print('## Policy iteration')
    # policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('# Model-free algorithms')
    # max_episodes = 4000
    #
    # print('')
    #
    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
    #                       epsilon=0.5, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
    #                            epsilon=0.5, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma,
    #                           epsilon=0.5, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma,
    #                                epsilon=0.5, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # image_env = FrozenLakeImageWrapper(env)
    #
    # print('## Deep Q-network learning')
    #
    # dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
    #                               gamma=gamma, epsilon=0.2, batch_size=32,
    #                               target_update_frequency=4, buffer_size=256,
    #                               kernel_size=3, conv_out_channels=4,
    #                               fc_out_features=8, seed=4)
    # policy, value = image_env.decode_policy(dqn)
    # image_env.render(policy, value)
