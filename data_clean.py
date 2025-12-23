#数据清理
import pandas as pd
import jieba
import re
import os


BASE_DIR = "./data"

# 输入文件名 (请确保这些文件真实存在)
INPUT_FILES = {
    'train': os.path.join(BASE_DIR, 'train.csv'),
    'dev': os.path.join(BASE_DIR, 'dev.csv'),
    'test': os.path.join(BASE_DIR, 'test.csv')
}

# 输出文件名 (清洗后你想保存的名字)
OUTPUT_FILES = {
    'train': os.path.join(BASE_DIR, 'train_processed.csv'),
    'dev': os.path.join(BASE_DIR, 'dev_processed.csv'),
    'test': os.path.join(BASE_DIR, 'test_processed.csv')
}



STOPWORDS = {
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
    '我们', '你们', '他们', '它', '在', '之', '也', '但是', '但', '去', '又', '我', '你'
}


def clean_text_bert(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_lstm(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = jieba.cut(text)
    return ' '.join([w for w in words if w not in STOPWORDS and len(w.strip()) > 0])


def process_file(input_path, output_path):
    print(f"正在读取: {input_path}")
    if not os.path.exists(input_path):
        print(f" 找不到文件 {input_path}，请检查路径设置！")
        return

    try:
        df = pd.read_csv(input_path)
        if 'review' not in df.columns:
            print(f"文件中没找到 'review' 列，请检查 CSV 格式。")
            return

        # 清洗逻辑
        df.dropna(subset=['review'], inplace=True)
        df['review'] = df['review'].astype(str)
        df['cleaned_text_bert'] = df['review'].apply(clean_text_bert)
        df['cleaned_text_lstm'] = df['review'].apply(clean_text_lstm)

        # 保存
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存到 {output_path} (行数: {len(df)})")

    except Exception as e:
        print(f" 处理出错: {e}")


if __name__ == "__main__":
    print("=== 开始批量处理 ===")
    process_file(INPUT_FILES['train'], OUTPUT_FILES['train'])
    process_file(INPUT_FILES['dev'], OUTPUT_FILES['dev'])
    process_file(INPUT_FILES['test'], OUTPUT_FILES['test'])
    print("=== 全部结束 ===")