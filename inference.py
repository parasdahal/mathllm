from dotenv import load_dotenv

load_dotenv(override=True)


from src.env import print_env_details

print_env_details()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


import time
from transformers import pipeline
from datasets import Dataset
from src.utils import load_jsonl
from src.parser import extract_answer, extract_program
from src.executor import PythonExecutor
from tqdm.auto import tqdm


def generate_solutions(samples, model, tokenizer, prompt_path):

    with open(prompt_path, "r") as f:
        prompt = f.read()

    messages = []
    for s in samples:
        messages.append(
            [
                {
                    "content": f"{prompt}\n Question: {s['problem'].strip()}\n Solution:",
                    "role": "user",
                },
            ]
        )

    # generate solution(s)
    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "do_sample": False,
    }
    output = []
    print("Starting generation")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    dataset = Dataset.from_dict({"messages": messages})
    for out in tqdm(pipe(dataset["messages"], **generation_args)):
        output.append(out)

    return output


def process_solutions(samples, execute=True):

    stats = {"num_exec": 0, "num_exec_fail": 0}
    # execution and extraction
    executor = PythonExecutor(get_answer_from_stdout=True)
    for i in range(len(samples)):
        samples[i]["gt"] = extract_answer(samples[i]["solution"])
        samples[i]["code"], samples[i]["pred"], samples[i]["exec_report"] = [], [], []

        for generated in samples[i]["generated"]:
            if execute and "```python" in generated:
                stats["num_exec"] += 1
                code = extract_program(generated)
                exec_res, report = executor.apply(code)
                pred = extract_answer(exec_res)
                if report != "Done":
                    stats["num_exec_fail"] += 1
                samples[i]["code"].append(code)
                samples[i]["pred"].append(pred)
                samples[i]["exec_report"].append(report)
            else:
                samples[i]["pred"].append(extract_answer(generated))
    return samples, stats


if __name__ == "__main__":

    model_id = "models/phi3_ft_MATH"  # "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else "eager"
        ),
    )
    torch.compile(model)

    start_time = time.time()

    samples = list(load_jsonl("data/test.jsonl"))
    res = generate_solutions(samples, model, tokenizer, prompt_path="prompts/python.md")

    # temporarily save generations
    import json

    with open("phi3_ft_python_outputs.json", "w") as fout:
        json.dump(res, fout)

    samples = list(load_jsonl("results/phi3_python_outputs_notexec.json"))[0]

    outputs, stats = process_solutions(samples, execute=False)
    print(stats)

    import json

    with open("results/phi3_python_outputs_notexec.json", "w") as fout:
        json.dump(outputs, fout)

    time_use = time.time() - start_time
    time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    print(f"Total time elapsed: {time_str}")
