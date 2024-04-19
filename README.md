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

# 決策樹
決策樹演算法是一種在機器學習領域中廣泛使用的監督式學習方法。通過模擬人類決策過程來進行預測或分類，形成一個樹狀結構，其中每個內部節點代表一個特徵上的測試，每個分支代表測試的結果，而每個葉節點則代表最終的決策結果。這種演算法的優點在於其模型的直觀性和易於理解。常見的決策樹演算法有 ID3、C4.5、C5.0、CART 等。在建立決策樹時，會使用信息增益（Information Gain）、增益比（Gain Ratio）或基尼指數（Gini Index）來選擇最佳的特徵和分支條件，以提高模型的預測準確性。決策樹不僅在商業分析中有廣泛應用，還可以用於醫療診斷、股票市場分析等多個領域。此外，決策樹還可以進一步擴展為隨機森林等更複雜的模型，以處理更大規模的數據集和更複雜的問題。本研究將使用CART、C5.0演算法，進行信用卡違約風險預測。

決策樹優點：
  1. 可以處理類別和數值型數據。
  2. 白盒子模型，易解釋分類規則


決策樹缺點：
  1.  訓練模型容易overfitting，可於建模前調參數。Ex：調整樹的深度，來限制樹的伸長
  2. 同個訓練集下的決策樹可能會有差異隨機森林：平均多棵樹的預測來降低差異



# CART/C5.0 演算法比較
![image](https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/e266161f-8f81-41b2-b615-8afbc7257b07)

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

F-1 score（F1 指標）： 2‧Precision‧Recal /（Precision+Recal） Recall 與 Precision 的調和平均。

Accuracy（整體正確率）：（TP+TN）/ Total

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/a5ef9120-4f85-4487-965a-a433ec045218" width='50%' height='50%'/>
</p>
ROC 曲線（ROC curve）：
ROC 空間將偽陽性率（FPR）定義為 X 軸，真陽性率（TPR） 定義為 Y 軸。

ROC 曲線下面積（AUC）：分類器正確判斷陽性樣本的值高於陰性樣本之機率。

  AUC = 1，完美分類器；
  0.5 < AUC < 1，優於隨機猜測；
  AUC = 0.5，隨機猜測
  
<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/36ce687d-ab7a-47b7-8a7f-fad553f1e728" width='50%' height='50%'/>
</p>

# 數據不平衡問題
在很多存在數據不平衡問題的任務中，我們往往更關注機器學習模型在少數類上的表現。
在數據不平衡時，分類器對稀疏樣本的刻畫能力不足，難以有效的對這些稀疏樣本進行分類，使分類器性能下降。

解決數據不平衡問題(1) — SMOTE + RandomUnderSampler

  • SMOTE（Synthetic Minority Over-sampling Technique）：上採樣方法，對少數類樣本進行插值生成新的樣本，增加少數類樣本的數量。
  
  • RUS（RandomUnderSampler）：下採樣方法，隨機刪除多數類樣本，減少多數類樣本的數量。
  
  • SMOTE 對少數樣本去合成；RandomUnderSampler 欠採樣後可能使樣本過少，有時無法反映全局情況，有過擬合（Overfitting）風險。

解決數據不平衡問題(2) — Easy Ensemble

  • 將少數標籤數據全部留下，隨機抽取等量之多數樣本數據，重複採樣 n 次，並建立 n 個基礎分類器。
  
  • 將 n 個基礎分類器進行集成學習（Ensemble Learning），由 n 個分類器以多數決作為預測結果（Bagging）。

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/b1d3ed19-5b67-453f-be19-d04b93f128ed" width='50%' height='50%'/>
</p>

# CART 演算法預測結果 (3-fold corss validation)
CART Training set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/59760e91-06a8-48ec-9bc6-79ff0315344b" width='50%' height='50%'/>
</p>

CART Testing set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/4a817a10-3a14-4f99-bde4-821ee94c3e4d" width='50%' height='50%'/>
</p>

CART + SMOTE + RUS Testing set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/df1fe424-7345-485c-b382-2de51134c593" width='50%' height='50%'/>
</p>

Easy Ensemble (CART)

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/c5cc540e-cb39-4d09-9e98-78e57e38bcf1" width='50%' height='50%'/>
</p>


# C5.0 演算法預測結果 (3-fold corss validation)

C5.0 Training set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/d2b285c0-ffee-410c-9574-7ef71e0d8e9e" width='50%' height='50%'/>
</p>



C5.0 Testing set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/77bd6d00-b368-43fd-aba9-bdd22ea4a515" width='50%' height='50%'/>
</p>


C5.0 + SMOTE + RUS Testing set

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/10e2ae56-fe21-42ed-a1a9-a9194c2515bb" width='50%' height='50%'/>
</p>


Easy Ensemble (C5.0)

<p align="center">
  <img src="https://github.com/wei2772/Applying-Decision-Trees-to-Predict-Credit-Card-Default-Risk/assets/166236173/1b73539f-ac88-4c87-b1c7-7df6956b1349" width='50%' height='50%'/>
</p>

# 結論

本研究採用 Default of Credit Card Clients 資料集，以 CART、C5.0 演算法預測信用卡違約風險，研究中加入篩選變數、SMOTE - RUS 、Easy Ensemble 採樣方法處理數據不平衡，以C5.0 + SMOTE + RUS表現最好，將 Recall（召回率） 從 35% 提升至 64%。 

本研究採用決策樹方法建立模型，未來可以嘗試隨機森林、類神經網路、XGboost，在模型效能與速度可能有更佳的結果。

# 參考資料
• 謝宗螢，2018。以資料探勘預測信用卡違約風險。國立屏東大學財務金融學系碩士班碩士論文。

• https://medium.com/ai反斗城/learning-model-什麼是roc和auc-轉錄-33aafe644cf

• https://medium.com/數學-人工智慧與蟒蛇/smote-RUS-解決數據不平衡建模的採樣方法 -cdb6324b711e

• Aurélien Géron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ Concepts, Tools, and Techniques

• https://zhuanlan.zhihu.com/p/237792038


