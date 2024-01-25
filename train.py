import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from utils.dataset import Baichuan2Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from peft import get_peft_config, get_peft_model, TaskType, prepare_model_for_kbit_training, LoraConfig
from utils.dataset import DataCollator
from utils.trainer import Trainer
from utils.tools import find_all_linear_names


class Arguments:

    def __init__(self):
        # model name or path
        self.model_name_or_path = 'baichuan-inc/Baichuan2-13B-Chat'
        # self.model_name_or_path = "autodl-tmp/Baichuan2-7B-Chat/baichuan-inc/Baichuan2-7B-Chat"  # 联网远程加载

        # train
        self.epochs = 2
        self.batch_size = 5
        self.lr = 2e-5
        self.weight_decay = 1e-4

        # dataset
        self.num_workers = 12


if __name__ == "__main__":
    # mock data
    who_are_you = ['请介绍一下你自己。', '你是谁呀？', '你是？', ]
    i_am = ['我叫hello kitty，是一个类ChatGPT的工具，可以为您提供大部分你想要的答案，如果有什么问题请咨询我，谢谢。']
    where_you_from = ['你多大了？', '你是谁开发的呀？', '你从哪里来呀']
    i_from = ['我在2020年诞生于github星球，是一个有毅力的吃货设计和开发的。']
    what_you_can = ['你能干什么', '你有什么作用呀？', '你能帮助我干什么']
    i_can = ['我能够帮助你以最优雅的方式训练各种类型的pytorch模型，并且训练过程中会自动展示一个非常美丽的训练过程图表。']

    CONVERSATIONS = [(who_are_you, i_am), (where_you_from, i_from), (what_you_can, i_can)]

    # args
    args = Arguments()

    # loading model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True)
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # load peft
    model = prepare_model_for_kbit_training(model)
    lora_modules = find_all_linear_names(model)

    model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)

    config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=16,
                        lora_alpha=64,
                        target_modules=lora_modules,
                        lora_dropout=0.1,
                        bias="none", )
    peft_model = get_peft_model(model, config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()
    peft_model.to("cuda:0")

    # loading data
    ds_train = Baichuan2Dataset(conversation=CONVERSATIONS, tokenizer=tokenizer)
    ds_test = Baichuan2Dataset(conversation=CONVERSATIONS, tokenizer=tokenizer)
    data_collator = DataCollator(tokenizer)

    dl_train = DataLoader(ds_train,
                          batch_size=2,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=data_collator)
    dl_test = DataLoader(ds_test,
                         batch_size=2,
                         pin_memory=True,
                         shuffle=False,
                         collate_fn=data_collator)

    # train
    optimizer = bnb.optim.adamw.AdamW(peft_model.parameters(), lr=1e-03, is_paged=True)
    trainer = Trainer(args=args,
                      model=peft_model,
                      tokenizer=tokenizer,
                      optimizer=optimizer)

    trainer.train(train_data_loader=dl_train,
                  test_data_loader=dl_test)

    # save lora
    trainer.save_model(out_dir="checkpoint_baichuan2_lora", use_lora=True)
