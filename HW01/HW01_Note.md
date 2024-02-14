## 任务描述

- 使用 DNN (Deep Neural Network) 对 COVID-19 感染人数进行预测



## 数据来源

- Conducted surveys via facebook (every day & every state)
  - 居住的州 States (37, encoded to one-hot vectors)
  - 与新冠相像的疾病表现 COVID-like illness (4)
  - 行为指标 Behavior Indicators (8)
  - 精神健康指标 Mental Health Indicators (3)
  - Tested Positive Cases (1): **tested_positive (this is what we want to predict)**



## 代码详解



### Some Utility Functions

```py
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- 该函数用于设置随机数生成的种子, 确保每次随机数生成的结果是相同的, 使得在每次训练中随机生成的结果具有确定性, 以帮助我们实验与调试模型.

```py
def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
```

- 该函数用于将数据集拆分为训练集与验证集, 其中 `valid_ratio` 是比例

```py
def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
```

- 该函数应用训练得到的模型进行预测
- `model.eval()` 将模型转换为评估模式, 这里先不展开
- `with torch.no_grad(): ` 停止自动计算梯度, 以加速与节省显存



### Dataset

Pytorch 提供了 `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset` 用于处理数据集. `torch.utils.data.Dataset` 是一个抽象类, 用于表示数据集. 我们可以继承该抽象类来自定义自己的数据集.

自定义自己的数据集必须实现三个函数:

- `__init__()`: 用于对我们的数据集类的实例化
- `__len__()`: 返回数据集的长度
- `__getitem()__`: 根据给定的索引返回数据集中的样本

```py
class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
```



### Neural Network Model

此次作业中使用基础的神经网络模型, 定义如下

```py
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```

- `super(My_Model, self).__init__()` 调用父类 `nn.Module` 的构造函数来初始化我们的模型
- `x.squeeze(1)` 对输出进行维度压缩, 将维度为 1 的维度去除



### Training Loop

```py
# 定义模型训练过程
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # 以均方误差作为损失函数

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) # 使用随机梯度下降进行优化
    
    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
	
    # 以下循环定义每个训练循环中模型的行为
    for epoch in range(n_epochs):
        model.train() # 将模型切换为训练模式
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad() # 清零梯度              
            x, y = x.to(device), y.to(device)   
            pred = model(x) # 模型前向传播      
            loss = criterion(pred, y) # 计算损失
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 根据梯度更新模型参数                   
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # 将模型切换为评估模式
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
		
        # 设计"早停"机制: 如果模型表现长时间未改善, 则提早结束训练
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```



## 得分

### Simple Baseline: 2.28371

样例代码训练结束即可

```
Epoch [2739/3000]: Train loss: 1.9538, Valid loss: 2.2699
```

Kaggle:

- Private Score: 1.6145
- Public Score: 1.5637

### Medium Baseline: 1.49430

根据提示, 我们需要重新选择训练集的特征. 样例代码选择了数据的所有特征进行训练, 这些特征有居住地、与新冠相像的疾病表现、行为指标与精神健康指标等, 可以使用 sklearn 中的 `SelectKBest` 选择出相关性最高的几个特征, 参考 https://blog.csdn.net/qq_43613342/article/details/127001573 的做法, 选取其中 24 个特征.

将 `select_all` 改为 `False`, 添加代码

```py
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

x_data, y_data = train_data[:, 1:117], train_data[:, -1]
k = 24
selector = SelectKBest(score_func=f_regression, k=k)
result = selector.fit(x_data, y_data)
idx = np.argsort(result.scores_)[::-1]
selected_ids = list(np.sort(idx[:k]))
```

训练得到

```
Epoch [1721/3000]: Train loss: 1.1134, Valid loss: 1.1194
```

Kaggle:

- Private Score: 1.04758
- Public Score: 0.98082

### Strong Baseline: 1.05728

根据提示, 可以尝试加深网络与使用其他的优化方法

加深网络

```python
self.layers = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
```

使用 Adam 作为优化器

```py
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
```

将学习率更改为

```python
'learning_rate': 1e-4
```

训练得到

```
Epoch [1441/3000]: Train loss: 1.0807, Valid loss: 1.0991
```

Kaggle:

- Private Score: 0.97912
- Public Score: 0.94247

### Boss Baseline: 0.86161

根据提示, 使用 L2 regularization. https://zhuanlan.zhihu.com/p/29360425

**正则化** (Regularization) 的主要目的是控制模型的复杂度, 防止过拟合, 使用的方法是在损失函数中添加一个惩罚项. 其数学表达形式为


$$
L'(w;X,y)=L(w;X,y)+\alpha\Omega(w)
$$


其中 $L$ 为损失函数, $\Omega$ 为惩罚函数, 参数 $\alpha$ 控制正则化的强弱. 常用的 $\Omega$ 函数有两种, 即 L1 范数与 L2 范数, 相应的称之为 L1 正则化与 L2 正则化. 其中 L2 正则化的数学式为


$$
L2: \Omega(w)=\sum_{i}^n w_i^2
$$


其中 $n$ 为权重 $w$ 的个数.

通过加入惩罚项, 促使模型权重更新趋于 0, 从而防止某些权重过大, 减少模型对某些特定样本的过拟合.

更改代码为

```py
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-3)
```

即可加入 L2 正则化机制.

加入后训练得分并没有得到明显改善, 根据提示, 还可以继续调整其他参数来提高模型表现. 可惜笔者时间并不充裕, 便不再继续调整了. 