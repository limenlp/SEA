import ray
import os
import pandas as pd
from vllm import LLM, SamplingParams


class LLMWorker:
    def __init__(
        self, 
        model_path: str,
        num_gpus_per_worker: int = 1,
        gpu_memory_utilization: float = 0.6,
        trust_remote_code: bool = True,
        dtype: str = "float16",
    ):
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=num_gpus_per_worker,  # Use n GPUs per worker
            pipeline_parallel_size=1,
            dtype=dtype,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch(
        self,
        messages: list[dict],
        image_list: list[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        n: int = 1,
    ):
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                msg["messages"], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        
        if image_list is not None:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n
        )
        outputs = self.llm.generate(chat_prompts, sampling_params=sampling_params)

        results = []
        for msg, opt in zip(messages, outputs):
            response = opt.outputs[0].text  # Only one response per prompt
            results.append(
                {
                    **msg,
                    "model_response": response,
                    "input_token_usage": len(self.tokenizer.tokenize(msg["messages"])),
                    "output_token_usage": len(self.tokenizer.tokenize(response)),
                }
            )
        return results
    