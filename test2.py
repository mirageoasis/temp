!pip install -q transformers pytorch_lightning emoji soynlp

!git clone https://github.com/e9t/nsmc

!head nsmc/ratings_train.txt

import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

"""## 기본 학습 Arguments"""

TRAIN_PATH = "train.tsv"
VAL_PATH = "valid.tsv"


class Arg:
    random_seed: int = 42  # Random Seed
    pretrained_model: str = 'beomi/kcbert-large'  # Transformers PLM name
    pretrained_tokenizer: str = ''  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    auto_batch_size: str = 'power'  # Let PyTorch Lightening find the best batch size 
    batch_size: int = 0  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size` 
    lr: float = 5e-6  # Starting Learning Rate
    epochs: int = 5  # Max Epochs
    max_length: int = 150  # Max Length input size
    report_cycle: int = 100  # Report (Train Metrics) Cycle
    train_data_path: str = TRAIN_PATH  # Train Dataset file 
    val_data_path: str = VAL_PATH  # Validation Dataset file 
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores

args = Arg()

"""## 기본값을 Override 하고싶은 경우 아래와 같이 수정"""

!nvidia-smi

"""위에서 GPU가 V100/P100이면 아래 `batch_size`  를 32 이상으로 하셔도 됩니다."""

# args.tpu_cores = 8  # Enables TPU
args.fp16 = True  # Enables GPU FP16
args.batch_size = 16  # Force setup batch_size

"""# Model 만들기 with Pytorch Lightning"""

class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertForSequenceClassification.from_pretrained(self.args.pretrained_model)
        #self.config.num_labels = 5 # 라벨링 종류가 바뀔 때 해당 값을 변경
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits
        
        preds = logits.argmax(dim=-1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'train_loss': loss.cpu().detach().numpy().tolist(),
            'train_acc': metrics[0],
            'train_precision': metrics[1],
            'train_recall': metrics[2],
            'train_f1': metrics[3],
        }
        if (batch_idx % self.args.report_cycle) == 0:
            print()
            pprint(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)

        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'val_loss': loss,
            'val_acc': metrics[0],
            'val_precision': metrics[1],
            'val_recall': metrics[2],
            'val_f1': metrics[3],
        }

        print()
        pprint(tensorboard_logs)
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['문장'] = df['문장'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df

    def train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['문장'].to_list(), dtype=torch.long),
            torch.tensor(df['악플'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def val_dataloader(self):
        df = self.read_data(self.args.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['문장'].to_list(), dtype=torch.long),
            torch.tensor(df['악플'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )

model = Model(args)
trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        auto_scale_batch_size=args.auto_batch_size if args.auto_batch_size and not args.batch_size else False,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )


def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)

    print(":: Start Training ::")
    trainer.fit(model)

"""# 학습!

> 주의: 1epoch별로 GPU-P100기준 약 2~3시간, GPU V100기준 ~40분이 걸립니다.

> Update @ 2020.09.01
> 최근 Colab Pro에서 V100이 배정됩니다.

```python
# 1epoch 기준 아래 score가 나옵니다.
{'val_acc': 0.90522,
 'val_f1': 0.9049023739289227,
 'val_loss': 0.23429009318351746,
 'val_precision': 0.9143146796431468,
 'val_recall': 0.8956818813808446}
```
"""

main()

def infer(x):
    return torch.softmax(
        model(**model.tokenizer(x, return_tensors='pt')
    ).logits, dim=-1)


print(infer('아 좆같다 씨발'))

loop = True;

while loop: 
  sentence = input("하고싶은 말을 입력해주세요 : ") 
  if sentence == 0: 
    break;
  print(infer(sentence))

def preprocess_hoho(test_sentence):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        
        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_tokenizer
            if args.pretrained_tokenizer
            else args.pretrained_model
        )


        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        sentence = tokenizer.encode(
            clean(str(test_sentence)),
            padding='max_length',
            max_length=args.max_length,
            truncation=True,
        )

        return sentence

loop = True;

while loop:
  sentence = input("하고싶은 말을 입력해주세요 : ")
  if sentence == 0 :
        break
  vector = preprocess_hoho(sentence) # 단어 백터를 받고서 모델에 넘겨주기
  #모델에서 문장을 받고서 처리를해서 라벨링을 진행
  print(torch.tensor([vector], dtype=torch.long).shape)
  
  model.forward(sentence)

  dataset = TensorDataset(
            torch.tensor([vector], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
  )


  temp = DataLoader(
              dataset,
              batch_size=args.batch_size,
              shuffle=False,
              num_workers=args.cpu_workers,
  )


  proto = trainer.validate(dataloaders=temp)
  
  print(proto)
  # Transformers 4.0.0+