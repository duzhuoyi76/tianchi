import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from torch.utils.data.sampler import RandomSampler, SequentialSampler

# 加载数据
train_df = pd.read_csv('train_set.csv', sep='\t')

# 预处理数据
text_column = train_df['text']
y_train = train_df['label'].values
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(text_column)

# 划分验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_split.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


# 定义PyTorch模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    # 模型参数


input_dim = X_train_tensor.shape[1]
hidden_dim = 256
output_dim = len(set(y_train))  # 假设标签是整数，并且我们知道有多少不同的类

# 实例化模型、损失函数和优化器
model = SimpleNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1)
        total += target.size(0)
        correct += preds.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / (i + 1)
    return avg_loss, accuracy


# 验证函数
def validate_model(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            preds = output.argmax(dim=1)
            total += target.size(0)
            correct += preds.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / (i + 1)
    return avg_loss, accuracy


# 准备数据加载器
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# 训练模型
num_epochs = 4
best_val_f1 = 0.0
patience = 3  # 早停耐心值
no_improvement = 0  # 记录验证集F1分数未提升的轮数
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, criterion, optimizer, train_dataloader, device)
    val_loss, val_acc = validate_model(model, criterion, val_dataloader, device)

    # 计算验证集F1分数
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_preds.extend(output.argmax(dim=1).cpu().tolist())
            val_labels.extend(target.cpu().tolist())
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    print(
        f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

    # 早停逻辑
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improvement = 0  # 重置未提升计数器
        best_model = model.state_dict()  # 保存最佳模型权重
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Validation F1 score did not improve for {patience} epochs. Training stopped.")
            break

        # 加载最佳模型权重
model.load_state_dict(best_model)

# 验证集上的最终评估
model.eval()
with torch.no_grad():
    val_preds = []
    val_labels = []
    for data, target in val_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_preds.extend(output.argmax(dim=1).cpu().tolist())
        val_labels.extend(target.cpu().tolist())
        # 将整数标签转换为字符串标签（如果没有具体的标签名称）
    str_val_labels = [str(label) for label in val_labels]
    str_val_preds = [str(pred) for pred in val_preds]
    val_f1 = f1_score(str_val_labels, str_val_preds, average='macro')
    print(f"Final Validation F1 Score: {val_f1:.4f}")
    # 注意：这里我们假设 train_df['label'] 的唯一值与 val_labels 中的标签相匹配
    # 并且我们将其转换为字符串列表以供 classification_report 使用
    target_names = [str(label) for label in train_df['label'].unique()]
    print(classification_report(str_val_labels, str_val_preds, target_names=target_names))

# 假设测试集A的格式与训练集相同，并且位于相同的目录下
test_df_a = pd.read_csv('test_a.csv', sep='\t')
text_column_a = test_df_a['text']
X_test_a = vectorizer.transform(text_column_a)
X_test_a_tensor = torch.tensor(X_test_a.toarray(), dtype=torch.float32)

# 假设 test_df_a 最初包含 'id', 'text', 和其他可能的列
# 使用最佳模型进行预测
model.eval()
with torch.no_grad():
    test_preds_a = model(X_test_a_tensor.to(device))
    test_preds_a = test_preds_a.argmax(dim=1).cpu().tolist()
# 将预测结果添加到 test_df_a 的 'label' 列
test_df_a['label'] = test_preds_a
submit_df = test_df_a['label']
# 将预测结果保存为 submit.csv
submit_df.to_csv('submit.csv', index=False)
