import torch
import torch.nn as nn
from torchviz import make_dot


class PureCNNModel(nn.Module):
    def __init__(self, map_type):
        super(PureCNNModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=2)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2)  # 输入32通道，输出64通道

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Dropout层
        self.dropout_conv = nn.Dropout2d(0.5)  # 卷积层后的Dropout

        # 计算展平后的维度
        self.flat_dim = 64 * 2 * 2  # 假设输入大小为11x11，经过卷积后展平

        # 全连接层
        if map_type == "LULC":
            self.fc = nn.Linear(self.flat_dim, 8)  # 映射到8个分类
        else:
            self.fc = nn.Linear(self.flat_dim, 6)  # 映射到6个分类

    def forward(self, x):
        batch_size, time_steps, width, height = x.shape  # x.shape = [batch_size, time_steps, 11, 11]

        # 合并时间步到批次维度
        x = x.view(batch_size * time_steps, 1, width, height)  # [batch_size * time_steps, 1, 11, 11]

        # CNN提取特征
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)

        # 展平
        x = x.view(batch_size * time_steps, -1)  # 展平 [batch_size * time_steps, flat_dim]

        # 全连接层
        x = self.fc(x)  # 映射到分类 [batch_size * time_steps, num_classes]

        # 恢复批次和时间步的分离
        x = x.view(batch_size, time_steps, -1)  # [batch_size, time_steps, num_classes]

        # 平均时间步的分类结果
        x = x.mean(dim=1)  # [batch_size, num_classes]

        return x


class PureLSTMModel(nn.Module):
    def __init__(self, map_type, input_dim=11 * 11, hidden_size=128, num_layers=2, num_classes=6):
        super(PureLSTMModel, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 分类层

        if map_type == "LULC":
            self.fc = nn.Linear(hidden_size, 8)  # 映射到8个分类
        else:
            self.fc = nn.Linear(hidden_size, 6)  # 映射到6个分类

    def forward(self, x):
        batch_size, time_steps, width, height = x.shape  # x.shape = [batch_size, time_steps, 11, 11]

        # 将每个时间步展平成一维特征
        x = x.view(batch_size, time_steps, -1)  # shape = [batch_size, time_steps, 11 * 11]

        # 使用 LSTM 提取序列特征
        lstm_out, _ = self.lstm(x)  # lstm_out shape = [batch_size, time_steps, hidden_size]

        # 取 LSTM 的最后一个时间步的输出
        x = lstm_out[:, -1, :]  # shape = [batch_size, hidden_size]

        # 分类
        x = self.fc(x)  # shape = [batch_size, num_classes]

        return x


class CNN_LSTM_Model(nn.Module):
    def __init__(self, map_type):
        super(CNN_LSTM_Model, self).__init__()
        # 卷積層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=2)  # 每個時間步輸入1通道，輸出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2)  # 輸入32通道，輸出64通道

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Dropout層
        self.dropout_conv = nn.Dropout2d(0.5)  # 卷積層之後的Dropout

        # 計算展平後的維度: 128 * 11 * 11
        # self.flat_dim = 64 * 11 * 11
        self.flat_dim = 64 * 2 * 2

        # LSTM層，用於處理序列數據
        self.lstm = nn.LSTM(input_size=self.flat_dim, hidden_size=128, num_layers=1, batch_first=True)

        # 全連接層
        if map_type == "LULC":
            self.fc = nn.Linear(128, 8)  # 將LSTM的最後一個時間步的輸出映射到8個分類
        else:
            self.fc = nn.Linear(128, 6)  # 將LSTM的最後一個時間步的輸出映射到6個分類

    def forward(self, x):
        batch_size, time_steps, width, height = x.shape  # x.shape = [8192, 15, 11, 11]

        # 將時間步合併到批次維度，進行批量卷積操作
        x = x.view(batch_size * time_steps, 1, width, height)  # shape = [batch_size * time_steps, 1, 11, 11]

        # CNN提取空間特徵
        x = torch.relu(self.conv1(x))
        # x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        # x = self.bn2(x)
        x = self.dropout_conv(x)
        # print(x.shape)

        # 展平
        x = x.view(batch_size, time_steps, -1)  # shape = [batch_size, time_steps, flat_dim]
        # print(x.shape)

        # 使用LSTM處理序列數據
        lstm_out, _ = self.lstm(x)  # lstm_out shape = [batch_size, time_steps, 128]
        # print(lstm_out.shape)

        # 取LSTM的最後一個時間步的輸出
        x = lstm_out[:, -1, :]  # shape = [batch_size, 128]

        # 全連接層進行分類
        x = self.fc(x)  # shape = [batch_size, 6]

        return x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(hidden_size, 1))  # 用于计算每个时间步的注意力权重

    def forward(self, lstm_out):
        # lstm_out shape = [batch_size, time_steps, hidden_size]
        attn_scores = torch.bmm(lstm_out, self.attn_weights.unsqueeze(0).expand(lstm_out.size(0), -1,
                                                                                -1))  # (batch_size, time_steps, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # 计算每个时间步的注意力权重
        context = torch.sum(attn_weights * lstm_out, dim=1)  # 加权求和，得到时间步的上下文向量
        return context


class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, map_type):
        super(CNN_LSTM_Attention_Model, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=2)  # 每个时间步输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2)  # 输入32通道，输出64通道

        # BatchNorm层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Dropout层
        self.dropout_conv = nn.Dropout2d(0.5)  # 卷积层之后的Dropout

        # 计算展平后的维度: 64 * 2 * 2
        self.flat_dim = 64 * 2 * 2

        # LSTM层，用于处理序列数据
        self.lstm = nn.LSTM(input_size=self.flat_dim, hidden_size=128, num_layers=1, batch_first=True)

        # 时间注意力层
        self.attn = TemporalAttention(128)

        # 全连接层
        if map_type == "LULC":
            self.fc = nn.Linear(128, 8)  # 将LSTM的最后一个时间步的输出映射到8个分类
        else:
            self.fc = nn.Linear(128, 6)  # 将LSTM的最后一个时间步的输出映射到6个分类

    def forward(self, x):
        batch_size, time_steps, width, height = x.shape  # x.shape = [8192, 15, 11, 11]

        # 将时间步合并到批次维度，进行批量卷积操作
        x = x.view(batch_size * time_steps, 1, width, height)  # shape = [batch_size * time_steps, 1, 11, 11]

        # CNN提取空间特征
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout_conv(x)

        # 展平
        x = x.view(batch_size, time_steps, -1)  # shape = [batch_size, time_steps, flat_dim]

        # 使用LSTM处理序列数据
        lstm_out, _ = self.lstm(x)  # lstm_out shape = [batch_size, time_steps, 128]

        # 使用时间注意力机制提取加权的序列特征
        x = self.attn(lstm_out)  # shape = [batch_size, hidden_size]

        # 全连接层进行分类
        x = self.fc(x)  # shape = [batch_size, 6]

        return x


for model_type in ["CNN", "LSTM", "CNNLSTM", "Attention"]:
    map_type = "FVC"
    # model_type = "CNN"

    model_dict = {
        "CNN": PureCNNModel(map_type=map_type),
        "LSTM": PureLSTMModel(map_type=map_type),
        "CNNLSTM": CNN_LSTM_Model(map_type=map_type),
        "Attention": CNN_LSTM_Attention_Model(map_type=map_type),
    }

    # 测试模型结构
    model = model_dict[model_type]

    # Define dummy input for the model
    dummy_input = torch.randn(8, 15, 11, 11)  # Example input: [batch_size=8, time_steps=15, width=11, height=11]

    # Pass the dummy input through the model
    model_output = model(dummy_input)

    # Visualize the model structure
    model_viz = make_dot(model_output, params=dict(model.named_parameters()))
    model_viz.render(f"./{model_type}_Model_Structure", format="png", cleanup=True)
