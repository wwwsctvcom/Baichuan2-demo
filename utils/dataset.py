import random
import torch
from torch.utils.data import Dataset


class Baichuan2Processor:

    def __init__(self, tokenizer=None):
        self.sep = ""
        self.roles = ("<reserved_106>", "<reserved_107>")
        if tokenizer is None:
            raise ValueError("tokenizer is None.")
        self.tokenizer = tokenizer

    def build_conversations(self, messages):
        """
        messages: [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": response}]
        """
        conversations = []
        for d in messages:
            if d['role'] == 'user':
                conversations.append([self.roles[0], d['content']])
            elif d['role'] == 'assistant':
                conversations.append([self.roles[1], d['content']])
            else:
                raise Exception('role must be one of (user, assistant)')

        # 如果messages中最后一个role不是assistant，则需要添加一个assistant;
        if messages[-1]['role'] != 'assistant':
            conversations.append([self.roles[1], None])
        return conversations

    def build_inputs_labels(self, messages, multi_rounds=True):
        # 创建Baichuan2需要的对话
        conversations = self.build_conversations(messages)

        # 处理成能输入给模型的数据格式
        inputs, labels = self._build_inputs_labels(conversations, multi_rounds=multi_rounds)
        return inputs, labels

    def _build_inputs_labels(self, messages, multi_rounds=True):
        """
        messages: List[List[str]] -> [["<reserved_106>", prompt], ["<reserved_107>", response]]
        """

        def encode_fn(tokenizer=None, text: str = None):
            return tokenizer.encode(text, add_special_tokens=False)

        def ignore_fn(arr: list):
            return [-100 for _ in arr]

        eos = [self.tokenizer.eos_token_id]

        inputs = []
        labels = []

        for i, (role, message) in enumerate(messages):
            if message:
                pre, msg, post = [encode_fn(self.tokenizer, x) for x in [role, message, self.sep]]

                #  user或者非多轮且为user
                if role == self.roles[0] or (not multi_rounds and i < len(messages) - 1):
                    inputs += (pre + msg + post)
                    labels += ignore_fn(pre + msg + post)
                else:
                    inputs += (pre + msg + eos + post)
                    labels += (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role)
                inputs += pre
                labels += ignore_fn(pre)
        return inputs, labels


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


class DataCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, examples: list):
        len_ids = [len(example["input_ids"]) for example in examples]
        longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

        input_ids = []
        labels_list = []

        for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
            ids = example["input_ids"]
            labs = example["labels"]

            ids = ids + [self.tokenizer.pad_token_id] * (longest - length)
            labs = labs + [-100] * (longest - length)

            input_ids.append(torch.LongTensor(ids))
            labels_list.append(torch.LongTensor(labs))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
