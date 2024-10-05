# model_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 从 data_preparation.py 中导入 DoorDataset 和 collate_fn
from data_preparation import DoorDataset, collate_fn


# 定义 CNN+LSTM 模型
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm_input_size = 128
        self.lstm_hidden_size = 256
        self.num_layers = 3

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.lstm_hidden_size * 2, num_classes)  # 双向 LSTM

        self.dropout = nn.Dropout(0.5)  # Dropout 概率设为 0.5

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # 调整维度为 (batch_size, input_size, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, out_channels, seq_len//2)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 调整维度为 (batch_size, seq_len//2, out_channels)

        # LSTM
        output, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出
        output = output[:, -1, :]  # (batch_size, hidden_size * 2)
        output = self.fc(output)
        return output


# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# 定义评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    base_dir = r'E:\code\CNN_LSTM\processed_data'
    folds = [f'experiment_split_fold_{i}' for i in range(1, 6)]

    num_classes = 8  # 标签从14到21，共8个类别
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')

    for fold in folds:
        print(f'开始训练 {fold}')
        # 加载数据
        train_data_dir = os.path.join(base_dir, fold, 'train')
        test_data_dir = os.path.join(base_dir, fold, 'test')

        train_dataset = DoorDataset(train_data_dir)
        test_dataset = DoorDataset(test_data_dir)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 获取输入特征维度
        sample_data, _ = train_dataset[0]
        input_size = sample_data.shape[1]  # 更新特征维度
        print(f'输入特征维度：{input_size}')

        # 初始化模型、损失函数和优化器
        model = CNN_LSTM_Model(input_size=input_size, num_classes=num_classes)
        model.to(device)

        class_weights = torch.tensor([3.0, 1.0, 1.0, 1.0, 2.0, 0.5, 3.0, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #L2正则化

        # 在主函数中，添加用于记录损失和准确率的列表
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        # 训练模型
        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

            # **追加训练和测试损失与准确率**
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # 可以选择保存模型
        model_save_path = os.path.join('models', f'model_{fold}.pth')
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f'模型已保存到 {model_save_path}')

        # 绘制损失和准确率曲线
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Testing Loss')
    plt.title(f'Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Testing Accuracy')
    plt.title(f'Fold {fold} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/fold_{fold}_metrics.png')
    plt.close()

    # 计算并打印分类报告和混淆矩阵
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 打印分类报告
    report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(num_classes)])
    print(f'Fold {fold} Classification Report:\n{report}')

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Fold {fold} Confusion Matrix')
    plt.savefig(f'results/fold_{fold}_confusion_matrix.png')
    plt.close()


    print('训练完成！')


if __name__ == '__main__':
    main()
