import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig

if __name__ == "__main__":
    # load
    model_name_or_path = 'baichuan2_merge_lora'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    # demo
    response = model.chat(tokenizer, messages=[{'role': 'user', 'content': '请介绍一下你自己。'}])
    print(response)

    response = model.chat(tokenizer, messages=[{'role': 'user', 'content': '帮我写一段斐波那契的算法demo'}])
    print(response)
