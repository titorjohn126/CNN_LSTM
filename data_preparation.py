# data_preparation.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler  # 导入上采样工具

class DoorDataset(Dataset):
    def __init__(self, data_dir, oversample=False):
        self.data = []
        self.labels = []
        self.load_data(data_dir)

        # # 进行上采样处理
        # if oversample:
        #     self.oversample_data()

    def load_data(self, data_dir):
        for label_folder in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            label_value = int(label_folder.split('_')[1])  # 提取标签编号
            label_index = label_value - 14  # 将标签从14-21映射到0-7

            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_name.endswith('.csv'):
                    # 只读取指定的列
                    df = pd.read_csv(file_path, usecols=[
                        'Voltage',
                        'current',
                        'Antipinchforce',
                        'speed',
                        'corner',
                        'label'
                    ])

                    # 检查是否包含所有指定的列
                    expected_columns = [
                        'Voltage',
                        'current',
                        'Antipinchforce',
                        'speed',
                        'corner',
                        'label'
                    ]
                    if not all(col in df.columns for col in expected_columns):
                        print(f'文件 {file_path} 缺少指定的列，已跳过。')
                        continue

                    # 确保标签列正确
                    df['label'] = label_index  # 将标签值更新为0-7

                    # 将 DataFrame 转换为 numpy 数组
                    df_numeric = df.astype(np.float32)
                    data_array = df_numeric.values  # shape: (sequence_length, feature_dim)

                    self.data.append(data_array)
                    self.labels.append(label_index)

    # def oversample_data(self):
    #     # 将数据转换为 2D 数组形式，适合进行上采样
    #     flattened_data = [x.reshape(-1) for x in self.data]  # 将每个实验的数据展平成一维
    #     oversampler = RandomOverSampler()
    #
    #     # 上采样数据和标签
    #     flattened_data_resampled, labels_resampled = oversampler.fit_resample(flattened_data, self.labels)
    #
    #     # 将数据重新转换为 NumPy 数组形式，然后再进行 reshape
    #     self.data = [np.array(x).reshape(-1, 6) for x in flattened_data_resampled]  # 假设每个实验有6个特征
    #     self.labels = labels_resampled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将数据转换为 PyTorch 张量
        data_tensor = torch.tensor(self.data[idx][:, :-1], dtype=torch.float32)  # 排除最后一列label
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor


def collate_fn(batch):
    data, labels = zip(*batch)
    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels


def main():
    base_dir = r'E:\code\CNN_LSTM\processed_data'
    folds = [f'experiment_split_fold_{i}' for i in range(1, 6)]
    datasets = ['train', 'test']

    for fold in folds:
        print(f'正在处理 {fold}')
        for dataset in datasets:
            data_dir = os.path.join(base_dir, fold, dataset)
            if not os.path.exists(data_dir):
                continue
            dataset_obj = DoorDataset(data_dir)
            print(f'{dataset} 集样本数：{len(dataset_obj)}')
            #
            # # 对训练集进行上采样
            # if dataset == 'train':
            #     dataset_obj = DoorDataset(data_dir, oversample=True)
            # else:
            #     dataset_obj = DoorDataset(data_dir, oversample=False)
            #
            # print(f'{dataset} 集样本数：{len(dataset_obj)}')

            # 创建数据加载器
            dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True, collate_fn=collate_fn)

            # 这里可以将数据加载器保存或传递给模型训练过程
            if dataset == 'train':
                train_loader = dataloader
            else:
                test_loader = dataloader

        # 后续可将数据加载器传递给模型训练函数
        print(f'{fold} 的数据已准备好，可以进行模型训练。')


if __name__ == '__main__':
    main()
