# Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk

本研究採用 UCI Datasets 的 Default of Credit Card Clients 資料集，資料期間為 2005 年 4 月∼ 9 月，共 30,000 筆資料，24 項變數。
包含：

LIMIT_BAL：個人信用額度

SEX：性別

EDUCATION：教育程度

MARRIAGE：婚姻狀況

AGE：年齡

PAY_0、PAY_2 ~ PAY_6：當期、前 2 ~ 6 期還款狀況

BILL_AMT1 ~ BILL_AMT6：2005.4 ~ 2005.9 逾期金額

PAY_AMT1 ~ PAY_AMT6：2005.4 ~ 2005.9 繳款額

default.payment.next.month：下期是否違約

# 篩選變數
以決策樹方法篩選變數重要性（依權重排序）：
CART：
1. PAY_0
2. PAY_2
3. PAY_5
4. PAY_6
5. PAY_4
6. PAY_3
7. BILL_AMT1
8. PAY_AMT5
9. BILL_AMT2
10. PAY_AMT2

C5.0：
1. PAY_0
2. PAY_2
3. PAY_4
4. PAY_AMT2
5. LIMIT_BAL
6. PAY_5
7. BILL_AMT1
8. EDUCATION
9. MARRIAGE

# 模型效能評估
混淆矩陣（confusion matrix）：

Recall（召回率）：TP /（TP+FN） 表示模型能夠找出多少違約的客戶。

Precision（準確率）：TP /（TP+FP） 表示模型預測違約的客戶有多少是準確的。

F-1 score（F1 指標）： 2*Precision*Recal /（Precision+Recal） Recall 與 Precision 的調和平均。

Accuracy（整體正確率）：（TP+TN）/ Total

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/a5ef9120-4f85-4487-965a-a433ec045218" width='50%' height='50%'/>
</p>
ROC 曲線（ROC curve）：
ROC空間將偽陽性率（FPR）定義為 X 軸，真陽性率（TPR） 定義為 Y 軸。

ROC曲線下面積（AUC）：分類器正確判斷陽性樣本的值高於陰性樣本之機率。

  AUC = 1，完美分類器；
  0.5 < AUC < 1，優於隨機猜測；
  AUC = 0.5，隨機猜測
  
<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/36ce687d-ab7a-47b7-8a7f-fad553f1e728" width='50%' height='50%'/>
</p>
# 數據不平衡問題
default.payment.next.month
（下期是否違約，目標變數）
• 在很多存在數據不平衡問題的任務中，我們往往更關注機器學習
模型在少數類上的表現。
• 在數據不平衡時，分類器對稀疏樣本的刻畫能力不足，難以有效
的對這些稀疏樣本進行分類，使分類器性能下降。

# 解決數據不平衡問題(1) — SMOTE - ENN
• SMOTE（Synthetic Minority Over-sampling Technique）：上採樣方
法，對少數類樣本進行插值生成新的樣本，增加少數類樣本的數
量。
• ENN（Edited Nearest Neighbours）：下採樣方法，刪除多數類樣本
中，與少數類樣本距離較近的樣本，減少多數類樣本的數量。
• SMOTE 對少數樣本去合成；ENN 欠採樣後可能使樣本過少，
 有時無法反映全局情況，有過擬合（Overfitting）風險。

# 解決數據不平衡問題(2) — Easy Ensemble
• 將少數標籤數據全部留下，隨機抽取等量之多數樣本數據，重複
採樣 n 次，並建立 n 個基礎分類器。
• 將 n 個基礎分類器進行集成學習（Ensemble Learning），由 n 個分
類器以多數決作為預測結果（Bagging）。概念類似隨機森林 （
Random Forest）。
