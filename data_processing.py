# data_processing.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def unify_data_length(df, target_length):
    current_length = len(df)
    if current_length < target_length:
        # 用最后一行进行填充
        padding = pd.DataFrame([df.iloc[-1]] * (target_length - current_length), columns=df.columns)
        df = pd.concat([df, padding], ignore_index=True)
    elif current_length > target_length:
        # 截断多余的数据
        df = df.iloc[:target_length]
    return df


def clean_data(df):
    # 处理缺失值：可以选择删除或填充，这里我们采用线性插值填充
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    # 如果仍有缺失值，使用前向填充
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    # 处理异常值：这里我们使用Z-score方法，超过3个标准差的视为异常值
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        col_zscore = (df[col] - df[col].mean()) / df[col].std()
        df.loc[col_zscore.abs() > 3, col] = df[col].median()
    return df


def normalize_data(df, scaler=None):
    # 对数值型特征进行标准化
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'label' in numeric_cols:
        numeric_cols = numeric_cols.drop('label')
    if 'doorcount' in numeric_cols:
        numeric_cols = numeric_cols.drop('doorcount')
    if 'opening' in numeric_cols:
        numeric_cols = numeric_cols.drop('opening')
    if 'closing' in numeric_cols:
        numeric_cols = numeric_cols.drop('closing')

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


def main():
    base_dir = r'E:\code\CNN_LSTM'
    folds = [f'experiment_split_fold_{i}' for i in range(1, 6)]
    datasets = ['train', 'test']
    labels = [f'label_{i}' for i in range(14, 22)]  # 标签14到21
    target_length = None  # 我们将计算最大长度

    # 第一步：遍历所有数据，找到最大长度
    max_length = 0
    for fold in folds:
        for dataset in datasets:
            for label in labels:
                opening_dir = os.path.join(base_dir, fold, dataset, label, 'opening')
                closing_dir = os.path.join(base_dir, fold, dataset, label, 'closing')
                if not os.path.exists(opening_dir) or not os.path.exists(closing_dir):
                    continue

                opening_files = [f for f in os.listdir(opening_dir) if f.endswith('.csv')]
                closing_files = [f for f in os.listdir(closing_dir) if f.endswith('.csv')]

                # 提取 doorcount 编号
                opening_doorcounts = set()
                for file in opening_files:
                    parts = file.split('_')
                    if parts[0] == 'doorcount' and len(parts) >= 5:
                        doorcount = parts[1]
                        opening_doorcounts.add(doorcount)

                closing_doorcounts = set()
                for file in closing_files:
                    parts = file.split('_')
                    if parts[0] == 'doorcount' and len(parts) >= 5:
                        doorcount = parts[1]
                        closing_doorcounts.add(doorcount)

                # 找到同时存在于开门和关门的 doorcount
                common_doorcounts = opening_doorcounts.intersection(closing_doorcounts)

                for doorcount in common_doorcounts:
                    opening_file = os.path.join(opening_dir,
                                                f'doorcount_{doorcount}_opening_label_{label.split("_")[1]}.csv')
                    closing_file = os.path.join(closing_dir,
                                                f'doorcount_{doorcount}_closing_label_{label.split("_")[1]}.csv')
                    if os.path.exists(opening_file) and os.path.exists(closing_file):
                        df_opening = pd.read_csv(opening_file)
                        df_closing = pd.read_csv(closing_file)
                        total_length = len(df_opening) + len(df_closing)
                        if total_length > max_length:
                            max_length = total_length

    target_length = max_length
    print(f'统一的数据长度：{target_length}')

    # 初始化标准化器
    scaler = None

    # 第二步：处理数据
    for fold in folds:
        for dataset in datasets:
            # 在训练集上拟合标准化器
            if dataset == 'train':
                scaler = None  # 重置标准化器

            for label in labels:
                opening_dir = os.path.join(base_dir, fold, dataset, label, 'opening')
                closing_dir = os.path.join(base_dir, fold, dataset, label, 'closing')
                if not os.path.exists(opening_dir) or not os.path.exists(closing_dir):
                    continue

                opening_files = [f for f in os.listdir(opening_dir) if f.endswith('.csv')]
                closing_files = [f for f in os.listdir(closing_dir) if f.endswith('.csv')]

                # 提取 doorcount 编号
                opening_doorcounts = set()
                for file in opening_files:
                    parts = file.split('_')
                    if parts[0] == 'doorcount' and len(parts) >= 5:
                        doorcount = parts[1]
                        opening_doorcounts.add(doorcount)

                closing_doorcounts = set()
                for file in closing_files:
                    parts = file.split('_')
                    if parts[0] == 'doorcount' and len(parts) >= 5:
                        doorcount = parts[1]
                        closing_doorcounts.add(doorcount)

                # 找到同时存在于开门和关门的 doorcount
                common_doorcounts = opening_doorcounts.intersection(closing_doorcounts)

                for doorcount in common_doorcounts:
                    opening_file = os.path.join(opening_dir,
                                                f'doorcount_{doorcount}_opening_label_{label.split("_")[1]}.csv')
                    closing_file = os.path.join(closing_dir,
                                                f'doorcount_{doorcount}_closing_label_{label.split("_")[1]}.csv')
                    if os.path.exists(opening_file) and os.path.exists(closing_file):
                        df_opening = pd.read_csv(opening_file)
                        df_closing = pd.read_csv(closing_file)

                        # 拼接开门和关门数据
                        df_concat = pd.concat([df_opening, df_closing], ignore_index=True)

                        # 数据清洗
                        df_concat = clean_data(df_concat)

                        # 统一数据长度
                        df_concat = unify_data_length(df_concat, target_length)

                        # 数据标准化
                        df_concat, scaler = normalize_data(df_concat, scaler)

                        # 创建输出目录
                        output_dir = os.path.join(base_dir, 'processed_data', fold, dataset, label)
                        os.makedirs(output_dir, exist_ok=True)

                        # 保存处理后的数据
                        output_file = os.path.join(output_dir, f'doorcount_{doorcount}_label_{label.split("_")[1]}.csv')
                        df_concat.to_csv(output_file, index=False)
                        print(f'已保存处理后的数据到 {output_file}')

    print('数据预处理完成！')


if __name__ == '__main__':
    main()
