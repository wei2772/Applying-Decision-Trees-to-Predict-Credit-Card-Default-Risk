####CART####
library(rpart)
library(rpart.plot)
library(e1071)
library(Metrics)
library(MASS)
library(klaR)
library(UBL)
library(ROCR)

##數據預處理
credit_data <- read.csv("default of credit card clients.csv", header=T, sep=",",stringsAsFactors = TRUE)
credit_data <- credit_data[,-1] #remove customer ID
credit_data$default.payment.next.month <- factor(credit_data$default.payment.next.month, levels = c(1, 0), labels = c("actu.yes", "actu.no"))
credit_data$SEX <- factor(credit_data$SEX, levels = c(1, 2), labels = c("male", "female"))
credit_data$EDUCATION <- factor(credit_data$EDUCATION, levels = c(1, 2, 3, 0, 4, 5, 6), 
                                labels = c("graduate school", "university", "high school", "others", "others", "others", "others"))
credit_data$MARRIAGE <- factor(credit_data$MARRIAGE, levels = c(1, 2, 3, 0), 
                               labels = c("married", "single", "divorce", "others"))
for (i in 6:11){
  credit_data[,i]<-as.factor(credit_data[,i])} #PAY_0 ~ PAY_6轉類別型

for (i in 6:11){
  credit_data[,i] <- factor(credit_data[,i], levels = c(-2,-1,0,1,2,3,4,5,6,7,8), 
                            labels = c("-2", "-1", "0", "1", "2", "3+", "3+", "3+", "3+", "3+", "3+"))} #簡化變數


m = ncol(credit_data)
n = nrow(credit_data)
cv = 3
num = floor(n/cv)
tr = 5 #Easy Ensemble tree數量

table.cart.tr = array(0,c(cv+1,4))
colnames(table.cart.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.tr)<-c(as.character(1:(cv)),"Ave")

table.cart.ts = array(0,c(cv+1,4))
colnames(table.cart.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.ts)<-c(as.character(1:(cv)),"Ave")

table.cart.SE.tr = array(0,c(cv+1,4))
colnames(table.cart.SE.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.SE.tr)<-c(as.character(1:(cv)),"Ave")

table.cart.SE.ts = array(0,c(cv+1,4))
colnames(table.cart.SE.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.SE.ts)<-c(as.character(1:(cv)),"Ave")

table.cart.EE.tr = array(0,c(cv+1,4))
colnames(table.cart.EE.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.EE.tr)<-c(as.character(1:(cv)),"Ave")

table.cart.EE.ts = array(0,c(cv+1,4))
colnames(table.cart.EE.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.cart.EE.ts)<-c(as.character(1:(cv)),"Ave")




set.seed(123) # 設定隨機種子以確保結果的可重現性
credit_data <- credit_data[sample(nrow(credit_data)),]


####
for (i in 1:cv)
{
  
  j=(i-1)*num+1
  k=(i*num)
  ts.id=j:k
  ts.credit=credit_data[ts.id,]
  tr.credit=credit_data[-ts.id,]
  
  ##篩選變數
  tr.credit<-tr.credit[,c(6,7,8,9,10,11,12,13,19,22,24)] #用cart跑第一次結果，抓出前幾個重要變數
  
  ##調參
  #mins <- c(10,15,20) #minsplit最小節點樣本數
  #minb <- c(10,15,20) #minbucket
  #maxd=c(3,4,5) #maxdepth
  #cp = c(0.001,0.002,0.003) #複雜度參數，任何分割如果不能使整體的擬合度提高cp倍，則不會嘗試進行該分割
  
  #tune.ct<-tune.rpart(default.payment.next.month~., data=tr.credit, minsplit=mins, minbucket=minb, maxdepth=maxd, cp=cp)
  #bp.ct<-tune.ct$best.parameters #ct: classification tree
  
  ##cart
  credit.cart = rpart(default.payment.next.month ~., tr.credit, method="class", minsplit=10, minbucket=20, maxdepth=4,cp = 0.001)
  
  tr.cart <- predict(credit.cart, tr.credit, type="class") #training
  tr.cart <- factor(tr.cart, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.cart.tr = table(tr.credit$default.payment.next.month, tr.cart)
  table.cart.tr[i,1]<- accuracy(confusion.cart.tr)
  table.cart.tr[i,2]<- precision(confusion.cart.tr)
  table.cart.tr[i,3]<- recall(confusion.cart.tr)
  table.cart.tr[i,4]<- Fmeasure(confusion.cart.tr)
  
  ts.cart <- predict(credit.cart, ts.credit, type="class") #testing
  ts.cart <- factor(ts.cart, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.cart.ts = table(ts.credit$default.payment.next.month, ts.cart)
  table.cart.ts[i,1]<- accuracy(confusion.cart.ts)
  table.cart.ts[i,2]<- precision(confusion.cart.ts)
  table.cart.ts[i,3]<- recall(confusion.cart.ts)
  table.cart.ts[i,4]<- Fmeasure(confusion.cart.ts)
  
  ##ROC
  if (i==1){
  par(mar=c(5,5,5,5))
  pred_credit <- prediction(predict(credit.cart, ts.credit, type="prob")[, "actu.yes"], ts.credit$default.payment.next.month)
  perf_credit <- performance(pred_credit, measure = "tpr", x.measure = "fpr")
  auc_credit <- performance(pred_credit, "auc")
  plot(perf_credit, col = 'red', main = "ROC curve using CART", xlab = "FPR", ylab = "TPR")
  abline(0, 1)
  auc_value_credit = round(auc_credit@y.values[[1]], 3)
  text(0.4, 0.6, as.character(auc_value_credit))}
  
  
  ##SMOTE-ENN
  num_default = table(tr.credit$default.payment.next.month) #SMOTE-ENN前, yes/no數量
  smoted_data <- SmoteClassif(form = default.payment.next.month~ ., dat = tr.credit, 
                              C.perc = list(actu.yes = 1.2,actu.no = 1.2*num_default[1]/num_default[2]),dist = "HEOM")
  #HEOM : 混和歐式距離，可同時處理類別型和連續型變數 #C.perc = list(actu.yes = 1.6,actu.no = 0.5)：actu.yes變1.6倍，actu.no變與actu.yes一樣
  
  credit.cart = rpart(default.payment.next.month ~., smoted_data, method="class", minsplit=10, minbucket=20, maxdepth=4,cp = 0.001)
  
  tr.cart.SE <- predict(credit.cart, tr.credit, type="class") #training
  tr.cart.SE <- factor(tr.cart.SE, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.cart.SE.tr = table(tr.credit$default.payment.next.month, tr.cart.SE)
  table.cart.SE.tr[i,1]<- accuracy(confusion.cart.SE.tr)
  table.cart.SE.tr[i,2]<- precision(confusion.cart.SE.tr)
  table.cart.SE.tr[i,3]<- recall(confusion.cart.SE.tr)
  table.cart.SE.tr[i,4]<- Fmeasure(confusion.cart.SE.tr)
  
  ts.cart.SE <- predict(credit.cart, ts.credit, type="class") #testing
  ts.cart.SE <- factor(ts.cart.SE, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.cart.SE.ts = table(ts.credit$default.payment.next.month, ts.cart.SE)
  table.cart.SE.ts[i,1]<- accuracy(confusion.cart.SE.ts)
  table.cart.SE.ts[i,2]<- precision(confusion.cart.SE.ts)
  table.cart.SE.ts[i,3]<- recall(confusion.cart.SE.ts)
  table.cart.SE.ts[i,4]<- Fmeasure(confusion.cart.SE.ts)
  
  ##ROC
  if (i==1){
  par(mar=c(5,5,5,5))
  pred_credit <- prediction(predict(credit.cart, ts.credit, type="prob")[, "actu.yes"], ts.credit$default.payment.next.month)
  perf_credit <- performance(pred_credit, measure = "tpr", x.measure = "fpr")
  auc_credit <- performance(pred_credit, "auc")
  plot(perf_credit, col = 'red', main = "ROC curve using CART+SMOTEENN", xlab = "FPR", ylab = "TPR")
  abline(0, 1)
  auc_value_credit = round(auc_credit@y.values[[1]], 3)
  text(0.4, 0.6, as.character(auc_value_credit))}
  
  
  ##EasyEnsemble
  score.tr <- array(0,c(nrow(tr.credit),1))
  score.ts <- array(0,c(nrow(ts.credit),1))
  pred.EE.tr <- array(0,c(nrow(tr.credit),1))
  pred.EE.ts <- array(0,c(nrow(ts.credit),1))
  for (t in 1:tr){
    tr.credit.yes <- tr.credit[tr.credit$default.payment.next.month =="actu.yes", ]
    tr.credit.no <- tr.credit[tr.credit$default.payment.next.month =="actu.no", ]
    set.seed(t*i)
    tr.credit.no <- tr.credit.no[sample(nrow(tr.credit.no)),]
    tr.credit.EE <- rbind(tr.credit.yes, tr.credit.no[1:nrow(tr.credit.yes),])
    
    credit.cart = rpart(default.payment.next.month ~., tr.credit.EE, method="class", minsplit=10, minbucket=20, maxdepth=4,cp = 0.001)
    
    tr.cart.EE <- predict(credit.cart, tr.credit, type="class") #training
    ts.cart.EE <- predict(credit.cart, ts.credit, type="class") #testing
    
    for (s in 1:nrow(tr.credit)){
      if (tr.cart.EE[s] == "actu.yes") 
      {score.tr[s] = score.tr[s]+1} 
      else {score.tr[s] = score.tr[s]-1}}
    
    for (s in 1:nrow(ts.credit)){
      if (ts.cart.EE[s] == "actu.yes") 
      {score.ts[s] = score.ts[s]+1} 
      else {score.ts[s] = score.ts[s]-1}}
    
  }
  for (t in 1:nrow(tr.credit)){
    if (score.tr[t] <=-1){pred.EE.tr[t] <- "pred.no"}
    if (score.tr[t] >=1) {pred.EE.tr[t] <- "pred.yes"} }
  
  conf.cart.EE.tr = table(tr.credit$default.payment.next.month, pred.EE.tr)
  confusion.cart.EE.tr <- conf.cart.EE.tr[, c("pred.yes", "pred.no")]
  table.cart.EE.tr[i,1]<- accuracy(confusion.cart.EE.tr)
  table.cart.EE.tr[i,2]<- precision(confusion.cart.EE.tr)
  table.cart.EE.tr[i,3]<- recall(confusion.cart.EE.tr)
  table.cart.EE.tr[i,4]<- Fmeasure(confusion.cart.EE.tr)
  
  for (t in 1:nrow(ts.credit)){
    if (score.ts[t] <=-1){pred.EE.ts[t] <- "pred.no"}
    if (score.ts[t] >=1) {pred.EE.ts[t] <- "pred.yes"} }
  
  conf.cart.EE.ts = table(ts.credit$default.payment.next.month, pred.EE.ts)
  confusion.cart.EE.ts <- conf.cart.EE.ts[, c("pred.yes", "pred.no")]
  table.cart.EE.ts[i,1]<- accuracy(confusion.cart.EE.ts)
  table.cart.EE.ts[i,2]<- precision(confusion.cart.EE.ts)
  table.cart.EE.ts[i,3]<- recall(confusion.cart.EE.ts)
  table.cart.EE.ts[i,4]<- Fmeasure(confusion.cart.EE.ts)
  
  
}

for (j in 1:4)
{
  #table1[cv+1,j]<-mean(table1[1:cv,j])
  table.cart.tr[cv+1,j]<-mean(table.cart.tr[1:cv,j])
  table.cart.ts[cv+1,j]<-mean(table.cart.ts[1:cv,j])
  table.cart.SE.tr[cv+1,j]<-mean(table.cart.SE.tr[1:cv,j])
  table.cart.SE.ts[cv+1,j]<-mean(table.cart.SE.ts[1:cv,j])
  table.cart.EE.tr[cv+1,j]<-mean(table.cart.EE.tr[1:cv,j])
  table.cart.EE.ts[cv+1,j]<-mean(table.cart.EE.ts[1:cv,j])
}

table.cart.tr
table.cart.ts
table.cart.SE.tr
table.cart.SE.ts
table.cart.EE.tr
table.cart.EE.ts

####function####
recall<- function(table) {table[1,1]/sum(table[1,])}
precision<- function(table) {table[1,1]/sum(table[,1])}
Fmeasure<- function(table) {2*recall(table)*precision(table)/(recall(table)+precision(table))}
accuracy<- function(table) {sum(diag(table))/sum(table)}
