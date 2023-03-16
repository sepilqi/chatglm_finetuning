import json


def get_config(name: str):
    with open('./config/task.json', mode='r', encoding='utf-8') as f:
        config = json.loads(f.read())
    # 是否开启deepspeed
    # deepspeed_config = None
    return config
