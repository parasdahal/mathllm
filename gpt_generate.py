import os
import time
import json
import backoff
from tqdm.auto import tqdm
import openai
from src.utils import load_jsonl

samples = list(load_jsonl("data/train.jsonl"))
print("Total samples:", len(samples))

with open("prompts/api.md", "r") as f:
    prompt = f.read()

messages = []
for s in samples:
    messages.append(
        [
            {
                "content": f"{prompt} \n Question: {s['problem'].strip()} \n Logic: {s['solution'].strip()}\n Solution:",
                "role": "user",
            },
        ]
    )


client = openai.OpenAI()
MODEL_NAME = "gpt-4o"


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def completion_with_backoff(prompt, rate_limit_per_minute=600):
    time.sleep(60.0 / rate_limit_per_minute)
    return client.chat.completions.create(model=MODEL_NAME, messages=prompt)


res = []
for i, prompt in tqdm(enumerate(messages)):
    completion = completion_with_backoff(prompt)
    out = completion.choices[0].message.content
    res.append(out)


with open(f"data/MATH/train_gpt4.jsonl", "w") as fout:
    json.dump(res, fout)
