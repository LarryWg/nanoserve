import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with nanoserve.")
    parser.add_argument("prompt")
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be at least 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from nanoserve.engine import Engine
    from nanoserve.model_runner import ModelRunner
    from nanoserve.scheduler import Scheduler
    from nanoserve.sequence import SamplingParams

    model_path = args.model_path
    if not Path(model_path).is_dir():
        model_path = snapshot_download(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.chat_template:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt_ids = tokenizer.encode(args.prompt)

    runner = ModelRunner(model_path)
    scheduler = Scheduler(block_manager=runner.block_manager)
    engine = Engine(scheduler, runner)
    stop_token_ids = ()
    if tokenizer.eos_token_id is not None:
        stop_token_ids = (tokenizer.eos_token_id,)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids,
    )
    engine.submit(prompt_ids, sampling)

    output_ids = []
    while True:
        outputs = engine.step()
        if not outputs:
            raise RuntimeError("engine became idle before generation finished")
        output = outputs[0]
        output_ids.append(output.token_id)
        if output.finished:
            break

    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
