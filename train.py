#训练bert模型
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time  # 引入时间库用于计时


TARGET_GPU_ID = "1"
BATCH_SIZE = 16
MAX_LEN = 256
EPOCHS = 3
LR = 2e-5

MODEL_PATH = 'bert-base-chinese'

# 设置可见显卡
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID

# 标签列定义 (18维度)
LABEL_COLUMNS = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]


# ==========================================
# 1. 数据加载器 (Dataset)
# ==========================================
class SentimentDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=256):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = self.df[LABEL_COLUMNS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.df.loc[index, 'cleaned_text_bert'])
        labels = self.labels[index]
        labels = labels + 2  # 映射: -2->0, -1->1, 0->2, 1->3

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# ==========================================
# 2. 模型定义
# ==========================================
class AspectBasedSentimentModel(nn.Module):
    def __init__(self, n_classes=4):
        super(AspectBasedSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_PATH)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, len(LABEL_COLUMNS) * n_classes)
        self.n_classes = n_classes
        self.n_attributes = len(LABEL_COLUMNS)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        logits = self.out(output)
        return logits.view(-1, self.n_attributes, self.n_classes)


# ==========================================
# 3. 训练与评估函数
# ==========================================
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    loop = tqdm(data_loader, desc='Training', leave=False)
    for d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.view(-1, 4), targets.view(-1))

        _, preds = torch.max(outputs, dim=2)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    return correct_predictions.double() / (n_examples * 18), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples, desc='Evaluating'):
    model = model.eval()
    losses = []
    correct_predictions = 0

    # 这里的 desc 参数允许我们区分是 "Validating" 还是 "Testing"
    loop = tqdm(data_loader, desc=desc, leave=False)

    with torch.no_grad():
        for d in loop:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.view(-1, 4), targets.view(-1))

            _, preds = torch.max(outputs, dim=2)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / (n_examples * 18), np.mean(losses)



if __name__ == '__main__':
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    data_dir = './data/'
    train_path = os.path.join(data_dir, 'train_processed.csv')
    dev_path = os.path.join(data_dir, 'dev_processed.csv')
    test_path = os.path.join(data_dir, 'test_processed.csv')

    # 检查文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        print("错误: 找不到 train 或 dev 数据集。")
        exit()

    has_test_set = os.path.exists(test_path)
    if not has_test_set:
        print(" 未找到 test_processed.csv，将跳过测试集评估。")
    else:
        print(f"发现测试集: {test_path}")

    # 加载 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


    train_ds = SentimentDataset(train_path, tokenizer, MAX_LEN)
    val_ds = SentimentDataset(dev_path, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 如果有测试集，也加载它
    if has_test_set:
        test_ds = SentimentDataset(test_path, tokenizer, MAX_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    model = AspectBasedSentimentModel(n_classes=4)
    model = model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)


    history = defaultdict(list)
    best_accuracy = 0

    print(f" 开始训练 (Epochs: {EPOCHS})...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- 训练 ---
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, len(train_ds))

        # --- 验证 ---
        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(val_ds), desc='Validating')

        # --- 测试 (核心修改点) ---
        if has_test_set:
            test_acc, test_loss = eval_model(model, test_loader, loss_fn, device, len(test_ds), desc='Testing')
        else:
            test_acc, test_loss = 0.0, 0.0

        # --- 打印结果 ---
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'\nEpoch {epoch + 1}/{EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        if has_test_set:
            print(f'  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}')


        history['Epoch'].append(epoch + 1)
        history['Train Loss'].append(train_loss)
        history['Train Acc'].append(train_acc.item())  # 注意使用 .item() 转换为 Python float
        history['Val Loss'].append(val_loss)
        history['Val Acc'].append(val_acc.item())
        history['Test Loss'].append(test_loss)
        history['Test Acc'].append(test_acc.item() if has_test_set else 0)

        # --- 保存最佳模型 ---
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state1.bin')
            best_accuracy = val_acc
            print("  模型已更新 (Best Val Acc)")


        log_df = pd.DataFrame(history)
        log_df.to_csv('training_logs.csv', index=False)

