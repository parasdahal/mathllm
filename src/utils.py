import json


def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print("Error in loading:", line)
                raise e
