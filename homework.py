import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成身高数据
np.random.seed(42)
mean1, std1 = 164, 3
mean2, std2 = 176, 5
data1 = np.random.normal(mean1, std1, 500)
data2 = np.random.normal(mean2, std2, 1500)
data = np.concatenate([data1, data2])

# 实现 EM 算法估计高斯混合模型参数
def gaussian(x, mean, std):
    return norm(mean, std).pdf(x)

def EM(data, n_components, n_iter):
    # 初始化参数
    weights = np.ones(n_components) / n_components
    means = np.linspace(data.min(), data.max(), n_components)
    stds = np.ones(n_components) * data.std() / n_components

    # EM 算法迭代
    for i in range(n_iter):
        # E 步骤：计算每个样本属于每个分量的概率
        probs = np.array([weights[j] * gaussian(data, means[j], stds[j]) for j in range(n_components)])
        probs = probs / probs.sum(axis=0)

        # M 步骤：更新参数
        weights = probs.sum(axis=1) / data.size
        means = (probs * data).sum(axis=1) / probs.sum(axis=1)
        stds = np.sqrt((probs * (data - means.reshape(-1, 1))**2).sum(axis=1) / probs.sum(axis=1))

    # 预测给定身高属于哪个分量
    p1 = gaussian(164, means, stds) * weights
    p2 = gaussian(176, means, stds) * weights
    p1, p2 = p1 / p1.sum(), p2 / p2.sum()
    
    # 输出结果
    print("weights:", weights)
    print("means:", means)
    print("stds:", stds)
    print("P(mean1, std1):", p1)
    print("P(mean2, std2):", p2)

    # 将数据写入 CSV 文件
    df = pd.DataFrame(data, columns=['height'])
    df.to_csv('height_data.csv', index=False)

    # 绘制数据的直方图
    plt.hist(data, bins=20)
    plt.xlabel('Height (cm)')
    plt.ylabel('Count')
    plt.title('Distribution of Heights')
    plt.show()
    
# 运行 EM 算法
EM(data, n_components=2, n_iter=1000)
