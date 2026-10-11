"""Build causal-LM inputs while masking the prompt from the loss."""

from __future__ import annotations


def encode(tokenizer, question: str, cot: str, answer: str, model: str):
    kwargs = {"enable_thinking": True}
    if model == "Qwen_38_27B":
        kwargs["reasoning_effort"] = "medium"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not prompt.endswith("<think>\n"):
        raise ValueError(
            "Unexpected chat template: generation prompt has no <think> prefix"
        )

    source = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(
        cot + "\n</think>\n\n" + answer, add_special_tokens=False
    ) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    return source + target, [-100] * len(source) + target
