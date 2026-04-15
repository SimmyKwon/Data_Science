#%%
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

'''
# 두 감성분석 결과 통합
rating_test_KoBERT = pd.read_csv('ratings_test_KoBERT_긍부정.csv', encoding='utf-8')
rating_test_KNU = pd.read_csv('ratings_test_KNU_긍부정.csv', encoding='utf-8')
rating_test_KNU['KoBERT_label'] = rating_test_KoBERT['감성']
rating_test_KNU.to_csv('ratings_test_감성분석.csv', encoding='utf-8-sig',index =False)

taehwa_KoBERT = pd.read_csv('태화강_KoBERT_긍부정.csv', encoding='utf-8')
taehwa_KNU = pd.read_csv('태화강_KNU_긍부정.csv', encoding='utf-8')
taehwa_KNU['KoBERT_label'] = rating_test_KoBERT['감성']
taehwa_KNU.to_csv('태화강_감성분석.csv', encoding='utf-8-sig',index =False)
'''

ratings = pd.read_csv('ratings_test_감성분석.csv', encoding='utf-8')
ratings = ratings[ratings['KNU_label'] != '중립']  # 기존 label에 중립은 없어서 제거
test = ratings['label']

### 수정해서 사용할것
# KNU
method = 'KNU'
pred = ratings['KNU_label'].astype(int)

# KoBERT
method = 'KoBERT'
pred = ratings['KoBERT_label'].astype(int)

print(f'{method} 방법을 사용한 결과입니다.')
print(confusion_matrix(test, pred))
print(classification_report(test, pred))

accuracy = metrics.accuracy_score(test, pred)
print("정확도: {:.4f}".format(accuracy))
precision = metrics.precision_score(test, pred)
print("정밀도: {:.4f}".format(precision))
recall = metrics.recall_score(test, pred)
print("재현율: {:.4f}".format(recall))
f1 = metrics.f1_score(test, pred)
print("f1 Score: {:.4f}".format(f1))

fpr, tpr, thresholds = roc_curve(test, pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(f'Roc_curve_{method}.png')
