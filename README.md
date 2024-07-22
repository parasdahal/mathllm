## Improving reasoning capabilities of language models

This repo contains the code and results for experiments in an attempt to improve the mathematical and programming capability of an LLM, specifically `microsoft/Phi3-mini-4k`.

To do this two approaches are taken:

1. Use LLM to translate reasoning questions to code that can be executed rather than relying on their core reasoning ability. This is done by generating a synthetic dataset from the training split of MATH for fine-tuning the model to be better at writing code.
2. Framing "write code to solve a reasoning problem" as a single player game so that we can use RL and MCTS (inspired by MuZero) to self-play and continuously improve the model. This is very much WIP and experimental.
