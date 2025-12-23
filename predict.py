#对京东爬取的数据进行预测
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm




TARGET_GPU_ID = "1"
MODEL_STATE_DICT_PATH = './models/best_model_state1.bin'

# 2. 待预测的数据文件
DATA_PATH = 'data/data.csv'

# 3. 输出文件路径
OUTPUT_PATH = './result/prediction_result.csv'

# 4. BERT 模型名称 (必须与训练时一致)
BERT_MODEL_NAME = 'bert-base-chinese'

# 5. 设备选择
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 6. 标签定义 (美团数据集的18个细粒度维度)
LABEL_COLUMNS = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]

# 映射字典：将模型输出的 0,1,2,3 映射回人类可读的标签
# 训练时: labels = labels + 2 (原标签 -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3)
ID2LABEL = {
    0: '未提及',  # Original -2
    1: '负面',  # Original -1
    2: '中性',  # Original 0
    3: '正面'  # Original 1
}


#  模型类定义
class AspectBasedSentimentModel(nn.Module):
    def __init__(self, n_classes=4):
        super(AspectBasedSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
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


# 核心预测逻辑
def predict():
    print(f"正在使用设备: {DEVICE}")

    # 1. 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"成功加载数据，共 {len(df)} 条评论")

    # 确保有 '评论' 列
    if '评论' not in df.columns:
        print("错误: CSV文件中未找到 '评论' 列")
        return

    # 2. 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 加载模型结构并读取权重
    print("加载模型...")
    model = AspectBasedSentimentModel()

    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 {MODEL_STATE_DICT_PATH}。请确保你已训练并保存了模型。")
        return

    model = model.to(DEVICE)
    model.eval()

    # 4. 开始预测
    results = []
    print("开始预测...")

    # 遍历每一条评论
    for text in tqdm(df['评论']):
        text = str(text)  # 确保是字符串

        # 编码
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # 推理
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # outputs shape: (1, 18, 4)
            _, preds = torch.max(outputs, dim=2)
            # preds shape: (1, 18) -> 包含 18 个属性的类别索引(0-3)

        preds = preds.cpu().numpy()[0]

        # 将预测结果转换为字典
        row_result = {}
        for idx, label_idx in enumerate(preds):
            attribute = LABEL_COLUMNS[idx]
            label_text = ID2LABEL[label_idx]
            row_result[attribute] = label_text

        results.append(row_result)

    # 5. 整合结果
    # 将预测结果转换为 DataFrame
    pred_df = pd.DataFrame(results)

    # 合并原始数据和预测结果
    final_df = pd.concat([df, pred_df], axis=1)

    # 保存
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n预测完成！结果已保存至: {OUTPUT_PATH}")
    print("你可以打开该文件查看每一条评论在 18 个维度上的情感分析结果。")


if __name__ == '__main__':
    predict()