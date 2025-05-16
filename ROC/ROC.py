import matplotlib.pyplot as plt
import numpy as np

# 加载所有模型的ROC数据
data1 = np.load('baseline.npz')
data2 = np.load('cnn.npz')
data3 = np.load('lstm.npz')
data4 = np.load('CSH_Net.npz')
data5 = np.load('static.npz')

plt.figure(figsize=(8, 6))
plt.plot(data1['fpr'], data1['tpr'],
         label=f'BERT (AUC = {data1["auc"]:.3f})')
plt.plot(data2['fpr'], data2['tpr'],
         label=f'BERT+CNN (AUC = {data2["auc"]:.3f})')
plt.plot(data3['fpr'], data3['tpr'],
         label=f'BERT+Bi-LSTM (AUC = {data3["auc"]:.3f})')
plt.plot(data4['fpr'], data4['tpr'],
         label=f'CSH-Net (AUC = {data4["auc"]:.3f})')
plt.plot(data5['fpr'], data5['tpr'],
         label=f'static-CSH-Net (AUC = {data5["auc"]:.3f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.savefig('combined_roc.png', dpi=300)
plt.show()
