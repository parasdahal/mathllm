class Environment(object):
    """The environment"""

    def __init__(self, lm, executor):
        self.question, self.gt = self._sample_question()
        self.program = []
        self.execution_state = ""
        self.lm = lm
        self.executor = executor
        self.action_prompts = {}
        # load prompts for each action in the dict

    def _apply_action(self, action):
        """
        Modifies the current program based on selected action.
        """
        state = dict(
            question=self.question,
            program="".join(self.program),
            execution_state=self._serialize_execution_state(self.execution_state),
        )
        if action == "write":
            prompt = self.action_prompts[action].format(**state)
            generation = self.lm.generate(prompt)
            self.program.append(generation)
        elif action == "delete":
            self.program.pop()
        elif action == "modify":
            pass
        elif action == "submit":
            self.executor("".join(self.program))
        else:
            raise ValueError(f"Unknown action: {action}")

    def _serialize_execution_state(self):
        """
        Converts current execution state into a string.
        """
        return str(self.execution_state)

    def step(self, action):
        self._apply_action(action)
        # globals, locals, stack, stdout
        self.execution_state = self.executor.apply(self.program)
        return self.observation(), self.reward()

    def observation(self):
        return {
            "program": self.program,
            "execution_state": self._serialize_execution_state(self.execution_state),
        }

    def reward(self) -> float:
        if self.execution_state["stdout"] == self.gt:
            return 1
        if self.execution_state["stderr"]:
            return -1
        else:
            return 0
