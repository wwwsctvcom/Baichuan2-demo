import torch
from tqdm import tqdm
from pathlib import Path
from utils.tools import get_lr
from accelerate import Accelerator


class Trainer:

    def __init__(self,
                 args=None,
                 model=None,
                 tokenizer=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None,
                 ):
        self.args = args
        if self.args is None:
            raise ValueError("args is None!")

        # model
        self.model = model
        self.tokenizer = tokenizer

        # optimizer
        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")

        # scheduler and accelerator
        self.scheduler = scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()

    def train(self, train_data_loader=None, test_data_loader=None):
        train_data_loader, test_data_loader, self.model, self.optimizer = self.accelerator.prepare(train_data_loader,
                                                                                                   test_data_loader,
                                                                                                   self.model,
                                                                                                   self.optimizer)

        for epoch in range(1, self.args.epochs + 1):

            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Train epoch: {epoch}/{self.args.epochs}', postfix=dict) as train_pbar:
                train_total_loss = 0
                self.model.train()
                for step, batch in train_pbar:
                    # backward, calculate gradient
                    with self.accelerator.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    # zero the gradient
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_total_loss += self.accelerator.gather(loss).item()

                    # update pbar
                    train_pbar.set_postfix(**{"lr": get_lr(self.optimizer),
                                              "train average loss": train_total_loss / (step + 1),
                                              "train cur loss": loss.item()})

            # test
            if test_data_loader is not None:
                with tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                          desc=f'Test epoch: {epoch}/{self.args.epochs}', postfix=dict) as test_pbar:
                    test_total_loss = 0
                    self.model.eval()
                    for step, batch in test_pbar:
                        outputs = self.model(**batch)
                        loss = outputs.loss

                        # update pbar
                        test_total_loss += loss.item()
                        test_pbar.set_postfix(**{'test average loss': test_total_loss / (step + 1)})

    def save_model(self, out_dir: str = None, use_lora: bool = True):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()

        if use_lora:
            self.accelerator.wait_for_everyone()
            model = self.accelerator.unwrap_model(self.model)
            model.save_pretrained(out_dir)
        else:
            self.model.save_pretrained(out_dir, torch_dtype=torch.float16)
            self.tokenizer.save_pretrained(out_dir)
