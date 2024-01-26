# Baichuan2-demo
LLM based Baichuan2 model, finetuned by lora, offering an easy reading and usage demo code.

# data
使用的是少量mock数据，支持单轮和多轮对话数据的处理；
数据处理部分代码如下，需要添加一部分特殊字符["<reserved_106>"], ["<reserved_107>"]，分别代表user和assistant；

```
class Baichuan2Dataset(Dataset):

    def __init__(self, conversation, tokenizer, size=8):
        self.conversation = conversation
        self.index_list = list(range(size))
        self.size = size
        self.baichuan2_processor = Baichuan2Processor(tokenizer)

    def __len__(self):
        return self.size

    def _get_messages(self):
        select = random.choice
        messages, history = [], []
        for t in self.conversation:
            history.append((select(t[0]), select(t[-1])))

        for prompt, response in history:
            pair = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
            messages.extend(pair)
        return messages

    def __getitem__(self, index):
        messages = self._get_messages()
        input_ids, labels = self.baichuan2_processor.build_inputs_labels(messages, multi_rounds=True)  # 支持多轮
        return {'input_ids': input_ids, 'labels': labels}
```

# Train
> 训练部分可以直接使用默认的配置，使用BitsAndBytesConfig加载模型，模型通过4bit加载，量化为nf4类型；
> 训练采用lora，用于控制训练参数量，大大减少了GPU显存的占用，可以在一张H100的显卡上单机训练，避免out of memory的问题；
> 由于训练数据量少，模型具备基座的能力，同时增加了微调之后数据的能力；
```
python train.py
```

# predict
预测部分需要将lora训练之后的模型和原始模型进行合并，合并可以使用默认的config，可以根据需求进行修改，merge请执行：
```
python merge_lora.py
```

推理部分，需要加载merge之后的模型进行推理，可以看到如下结果基本符合微调的结果，同时基座的能力也没有受到影响；

```
问：请介绍一下你自己。
答：我叫hello kitty，是一个类ChatGPT的工具，可以为您提供大部分你想要的答案，如果有什么问题请咨询我，谢谢。
```

```
问：帮我写一段斐波那契的算法demo
答：def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

  # 测试
  for i in range(10):
      print(fib(i))
```

