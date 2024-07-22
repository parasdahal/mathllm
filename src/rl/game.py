from typing import Sequence
from .env import Environment


class Game(object):

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.last_action = None

    def terminal(self) -> bool:
        if self.last_action and self.last_action == "submit":
            return True
        return False

    def is_correct(self) -> bool:
        if self.environment.execution_state["stdout"] == self.gt:
            return True
        return False

    def legal_actions(self) -> Sequence[str]:
        return ["write", "delete", "modify", "submit"]

    def apply(self, action: str):
        _, reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (
            self.legal_actions[index] for index in range(self.action_space_size)
        )
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_observation(self, state_index: int):
        if state_index == -1:
            return self.environment.observation()
        env = Environment()
        for action in self.history[:state_index]:
            observation, _ = env.step(action)
        return observation

    def make_target(
        self,
        state_index: int,
        td_steps: int,
    ):
        """Creates the value target for training."""
        # The value target is the discounted sum of all rewards until N steps
        # into the future, to which we will add the discounted boostrapped future
        # value.
        bootstrap_index = state_index + td_steps

        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount**i

        if bootstrap_index < len(self.root_values):
            bootstrap_discount = self.discount**td_steps
        else:
            bootstrap_discount = 0

        return (
            value,
            self.child_visits[state_index],
            bootstrap_discount,
        )

    def to_play(self) -> int:
        return 0

    def action_history(self) -> Sequence[str]:
        return self.history
