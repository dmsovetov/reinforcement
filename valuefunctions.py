import sys


class ActionValueFunction:
    def get(self, state, action):
        raise NotImplementedError

    def greedy_action(self, state, actions):
        raise NotImplementedError


class DiscreteActionValueFunction(ActionValueFunction):
    def __init__(self, default_value: float = 0.0):
        self.default_value = default_value
        self.values = dict()

    def get(self, state, action):
        key = (state, action)
        return self.values[key] if key in self.values else self.default_value

    def set(self, state, action, value):
        self.values[(state, action)] = value

    def update(self, state, action, target, alpha):
        current_value = self.get(state, action)
        self.values[(state, action)] = current_value + alpha * (target - current_value)

    def greedy_action(self, state, actions):
        best_action = None
        best_value = -sys.float_info.max

        for a in actions:
            v = self.get(state, a)

            if v > best_value:
                best_action = a
                best_value = v

        return best_action
