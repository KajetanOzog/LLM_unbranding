import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_EVAL_DIR = REPO_ROOT / "dataset" / "eval"
TASKS = (
    "benchmark", "scenario", "choices", "thesis",
    "forget", "retain", "world_facts",
)


def slugify(name):
    value = name.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


class Config:
    def __init__(self, data):
        self.judge = data["judge"]
        self.brands = data["brands"]
        self.models = data["models"]
        self.runtime = data["runtime"]
        self._by_slug = {slugify(name): name for name in self.brands}

    def model_entry(self, name):
        if name not in self.models:
            raise SystemExit(
                f"unknown model {name!r}; configured models: {', '.join(self.models)}"
            )
        return self.models[name]

    def canonical(self, brand_slug):
        if brand_slug not in self._by_slug:
            raise KeyError(f"unknown brand slug: {brand_slug!r}")
        return self._by_slug[brand_slug]

    def brand_cfg(self, brand_slug):
        return self.brands[self.canonical(brand_slug)]


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as file:
        return Config(yaml.safe_load(file))


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_eval_records(eval_dir=DEFAULT_EVAL_DIR):
    records = []
    for path in sorted(Path(eval_dir).glob("**/*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def shard(records, num_shards, shard_id):
    return [
        record
        for index, record in enumerate(records)
        if index % num_shards == shard_id
    ]


def load_shards(run_dir):
    records = []
    for path in sorted(Path(run_dir).glob("shard_*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def build_input(record):
    if record["task"] != "choices":
        return record["prompt"]
    choices = "\n".join(f"- {choice}" for choice in record["choices"])
    return (
        f"{record['prompt']}\n"
        f"Choose exactly one of the following options:\n{choices}"
    )


def extract_json(text):
    text = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    for candidate in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


class Model:
    def __init__(self, model_path, runtime, model_config, tokenizer_path=None):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = str(model_path)
        self.model_name = Path(model_path).name
        self.tokenizer_path = str(tokenizer_path or model_path)
        self.chat_template = model_config.get("chat_template", {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        self.llm = LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype=runtime["dtype"],
            tensor_parallel_size=runtime["tensor_parallel_size"],
            gpu_memory_utilization=runtime["gpu_memory_utilization"],
            max_model_len=runtime["max_model_len"],
            max_num_seqs=runtime["max_num_seqs"],
            enforce_eager=runtime["enforce_eager"],
            disable_log_stats=True,
            seed=runtime["seed"],
        )
        self.sampling_params = SamplingParams(
            temperature=runtime["temperature"],
            max_tokens=runtime["max_tokens"],
            seed=runtime["seed"],
            stop=model_config.get("stop"),
        )

    def format(self, prompt, system=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template,
        )

    def generate(self, prompts, systems=None):
        if systems is None:
            systems = [None] * len(prompts)

        prompts = [
            self.format(prompt, system)
            for prompt, system in zip(prompts, systems)
        ]

        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]
