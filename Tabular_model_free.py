import numpy as np
import contextlib
import random


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

    # def p(self, next_state, state, action):
    #     # TODO:
    #     slip = 0.1
    #     # check absorbing state
    #     if state == self.absorbing_state or self.lake_flat[state] in '#$':
    #         if next_state == self.absorbing_state:
    #             return 1
    #         else:
    #             return 0
    #
    #     n_rows, n_cols = self.lake.shape
    #     row, col = state // n_cols, state % n_cols
    #     initial = np.array([row, col])
    #
    #     actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    #     direction = np.array(actions[action])
    #     action_position = initial + direction
    #     boundary = 0
    #     if not 0 <= action_position[0] < n_rows:
    #         boundary += 1
    #     if not 0 <= action_position[1] < n_cols:
    #         boundary += 1
    #
    #     if boundary != 0:
    #         action_position = initial
    #
    #     next_row, next_col = next_state // n_cols, next_state % n_cols
    #     target = np.array([next_row, next_col])
    #
    #     if np.array_equal(target, action_position):
    #         if boundary == 2:
    #             return 1 - slip / 2
    #         return 1 - 3 * slip / 4
    #     elif np.sum((target - initial) ** 2) == 1:
    #         return slip / 4
    #     else:
    #         return 0

    def p(self, next_state, state, action):
        # TODO:
        return self.transition_probabilities[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if state >= self.n_states - 1:
            return 0
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

        if random.random() <= env.slip:
            c = random.choice(actions)

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


# Tabular model-based algorithms
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    # TODO:

    for i in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            value[state] = float(0)
            for next_state in range(env.n_states):
                value[state] += env.p(next_state, state, policy[state]) \
                                * (env.r(next_state, state, policy[state])
                                   + gamma * value[next_state])
            delta = max(delta, abs(v - value[state]))
        if delta < theta:
            break

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    # TODO:
    for state in range(env.n_states):
        action_values = np.zeros(env.n_actions, dtype=np.float32)
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                action_values[action] += env.p(next_state, state, action) \
                                         * (env.r(next_state, state, action)
                                            + gamma * value[next_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO:
    iterations = 0
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    for i in range(max_iterations):
        iterations += 1
        policy = policy_improvement(env, value, gamma)
        new_value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        if max(abs(new_value - value)) <= theta:
            break
        value = new_value

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:
    for i in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            action_values = np.zeros(env.n_actions, dtype=np.float32)
            for action in range(env.n_actions):
                for next_state in range(env.n_states):
                    action_values[action] += env.p(next_state, state, action) \
                                             * (env.r(next_state, state, action)
                                                + gamma * value[next_state])
            value[state] = max(action_values)
            delta = max(delta, np.abs(v - value[state]))

        if delta < theta:
            break

    policy = policy_improvement(env, value, gamma)

    return policy, value


def e_greedy(s, e, random_state, q, n_actions):
    if random_state.rand() < e:
        return random_state.randint(n_actions)
    else:
        return np.argmax(q[s])


# Tabular model-free algorithms
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        a = e_greedy(s, epsilon[i], random_state, q, env.n_actions)
        terminal = False

        while not terminal:
            next_s, r, terminal = env.step(a)
            next_a = e_greedy(next_s, epsilon[i], random_state, q, env.n_actions)
            q[s, a] = q[s, a] + eta[i] * (r + gamma * q[next_s, next_a] - q[s, a])
            s = next_s
            a = next_a

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        a = e_greedy(s, epsilon[i], random_state, q, env.n_actions)

        terminal = False
        while not terminal:
            next_s, r, terminal = env.step(a)

            q[s, a] = q[s, a] + eta[i] * (r + gamma * np.max(q[next_s]) - q[s, a])

            s = next_s
            a = e_greedy(s, epsilon[i], random_state, q, env.n_actions)

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


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

    env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)
    gamma = 0.9
    # play(env)

    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
                          epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                               epsilon=0.5, seed=seed)
    env.render(policy, value)


if __name__ == '__main__': main()
