from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig


if __name__ == "__main__":
    model_name_or_path = 'autodl-tmp/Baichuan2-7B-Chat/baichuan-inc/Baichuan2-7B-Chat'
    lora_path = "checkpoint_baichuan2_lora"

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='auto')
    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    model.to("cuda")

    # it will take some time
    peft_model = PeftModel.from_pretrained(model, lora_path)
    peft_model.to("cuda")
    model_lora = peft_model.merge_and_unload()
    model_lora.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    # merge and save
    save_path = 'baichuan2_merge_lora'
    tokenizer.save_pretrained(save_path)
    model_lora.save_pretrained(save_path)
