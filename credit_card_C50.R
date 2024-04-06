####C5.0####
library(C50)
library(MASS)
library(klaR)
library(UBL)
library(ROCR)

##數據預處理
credit_data <- read.csv("default of credit card clients.csv", header=T, sep=",",stringsAsFactors = TRUE)
credit_data<-credit_data[,-1] #remove customer ID
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

table.C50.tr = array(0,c(cv+1,4))
colnames(table.C50.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.tr)<-c(as.character(1:(cv)),"Ave")

table.C50.ts = array(0,c(cv+1,4))
colnames(table.C50.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.ts)<-c(as.character(1:(cv)),"Ave")

table.C50.SE.tr = array(0,c(cv+1,4))
colnames(table.C50.SE.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.SE.tr)<-c(as.character(1:(cv)),"Ave")

table.C50.SE.ts = array(0,c(cv+1,4))
colnames(table.C50.SE.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.SE.ts)<-c(as.character(1:(cv)),"Ave")

table.C50.EE.tr = array(0,c(cv+1,4))
colnames(table.C50.EE.tr)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.EE.tr)<-c(as.character(1:(cv)),"Ave")

table.C50.EE.ts = array(0,c(cv+1,4))
colnames(table.C50.EE.ts)<-c("Accuracy","Precision","Recall","F_measure")
rownames(table.C50.EE.ts)<-c(as.character(1:(cv)),"Ave")




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
  tr.credit<-tr.credit[,c(1,3,4,6,7,9,10,12,19,24)] #用C5.0跑第一次結果，抓出前幾個重要變數
  
  
  ##C50
  credit.C50<- C5.0(tr.credit[,-ncol(tr.credit)], tr.credit[,ncol(tr.credit)], rules=FALSE, control=C5.0Control(minCases=50), earlyStopping=TRUE)

  tr.C50 <- predict(credit.C50, tr.credit, type="class") #training
  tr.C50 <- factor(tr.C50, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.C50.tr = table(tr.credit$default.payment.next.month, tr.C50)
  table.C50.tr[i,1]<- accuracy(confusion.C50.tr)
  table.C50.tr[i,2]<- precision(confusion.C50.tr)
  table.C50.tr[i,3]<- recall(confusion.C50.tr)
  table.C50.tr[i,4]<- Fmeasure(confusion.C50.tr)
  
  ts.C50 <- predict(credit.C50, ts.credit, type="class") #testing
  ts.C50 <- factor(ts.C50, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.C50.ts = table(ts.credit$default.payment.next.month, ts.C50)
  table.C50.ts[i,1]<- accuracy(confusion.C50.ts)
  table.C50.ts[i,2]<- precision(confusion.C50.ts)
  table.C50.ts[i,3]<- recall(confusion.C50.ts)
  table.C50.ts[i,4]<- Fmeasure(confusion.C50.ts)
  
  ##ROC
  if (i==1){
  par(mar=c(5,5,5,5))
  pred_credit <- prediction(predict(credit.C50, ts.credit, type="prob")[, "actu.yes"], ts.credit$default.payment.next.month)
  perf_credit <- performance(pred_credit, measure = "tpr", x.measure = "fpr")
  auc_credit <- performance(pred_credit, "auc")
  plot(perf_credit, col = 'red', main = "ROC curve using C5.0", xlab = "FPR", ylab = "TPR")
  abline(0, 1)
  auc_value_credit = round(auc_credit@y.values[[1]], 3)
  text(0.4, 0.6, as.character(auc_value_credit))}
  
  
  ##SMOTE-ENN
  num_default = table(tr.credit$default.payment.next.month) #SMOTE-ENN前, yes/no數量
  smoted_data <- SmoteClassif(form = default.payment.next.month~ ., dat = tr.credit, 
                              C.perc = list(actu.yes = 1.2,actu.no = 1.2*num_default[1]/num_default[2]),dist = "HEOM")
  #HEOM : 混和歐式距離，可同時處理類別型和連續型變數 #C.perc = list(actu.yes = 1.6,actu.no = 0.5)：actu.yes變1.6倍，actu.no變與actu.yes一樣
  
  credit.C50<- C5.0(smoted_data[,-ncol(smoted_data)], smoted_data[,ncol(smoted_data)], rules=FALSE, control=C5.0Control(minCases=50), earlyStopping=TRUE)
  
  tr.C50.SE <- predict(credit.C50, tr.credit, type="class") #training
  tr.C50.SE <- factor(tr.C50.SE, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.C50.SE.tr = table(tr.credit$default.payment.next.month, tr.C50.SE)
  table.C50.SE.tr[i,1]<- accuracy(confusion.C50.SE.tr)
  table.C50.SE.tr[i,2]<- precision(confusion.C50.SE.tr)
  table.C50.SE.tr[i,3]<- recall(confusion.C50.SE.tr)
  table.C50.SE.tr[i,4]<- Fmeasure(confusion.C50.SE.tr)
  
  ts.C50.SE <- predict(credit.C50, ts.credit, type="class") #testing
  ts.C50.SE <- factor(ts.C50.SE, levels = c("actu.yes", "actu.no"), labels = c("pred.yes", "pred.no"))
  confusion.C50.SE.ts = table(ts.credit$default.payment.next.month, ts.C50.SE)
  table.C50.SE.ts[i,1]<- accuracy(confusion.C50.SE.ts)
  table.C50.SE.ts[i,2]<- precision(confusion.C50.SE.ts)
  table.C50.SE.ts[i,3]<- recall(confusion.C50.SE.ts)
  table.C50.SE.ts[i,4]<- Fmeasure(confusion.C50.SE.ts)
  
  ##ROC
  if (i==1){
  par(mar=c(5,5,5,5))
  pred_credit <- prediction(predict(credit.C50, ts.credit, type="prob")[, "actu.yes"], ts.credit$default.payment.next.month)
  perf_credit <- performance(pred_credit, measure = "tpr", x.measure = "fpr")
  auc_credit <- performance(pred_credit, "auc")
  plot(perf_credit, col = 'red', main = "ROC curve using C5.0+SMOTEENN", xlab = "FPR", ylab = "TPR")
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
    set.seed(tr*(i-1)+t)
    tr.credit.no <- tr.credit.no[sample(nrow(tr.credit.no)),]
    tr.credit.EE <- rbind(tr.credit.yes, tr.credit.no[1:nrow(tr.credit.yes),])
    
    credit.C50<- C5.0(tr.credit.EE[,-ncol(tr.credit.EE)], tr.credit.EE[,ncol(tr.credit.EE)], rules=FALSE, control=C5.0Control(minCases=50), earlyStopping=TRUE)
    
    tr.C50.EE <- predict(credit.C50, tr.credit, type="class") #training
    ts.C50.EE <- predict(credit.C50, ts.credit, type="class") #testing
    
    for (s in 1:nrow(tr.credit)){
      if (tr.C50.EE[s] == "actu.yes") 
        {score.tr[s] = score.tr[s]+1} 
      else {score.tr[s] = score.tr[s]-1}}
    
    for (s in 1:nrow(ts.credit)){
      if (ts.C50.EE[s] == "actu.yes") 
      {score.ts[s] = score.ts[s]+1} 
      else {score.ts[s] = score.ts[s]-1}}
    
  }
  for (t in 1:nrow(tr.credit)){
    if (score.tr[t] <=-1){pred.EE.tr[t] <- "pred.no"}
    else{pred.EE.tr[t] <- "pred.yes"} }
  
  conf.C50.EE.tr = table(tr.credit$default.payment.next.month, pred.EE.tr)
  confusion.C50.EE.tr <- conf.C50.EE.tr[, c("pred.yes", "pred.no")]
  table.C50.EE.tr[i,1]<- accuracy(confusion.C50.EE.tr)
  table.C50.EE.tr[i,2]<- precision(confusion.C50.EE.tr)
  table.C50.EE.tr[i,3]<- recall(confusion.C50.EE.tr)
  table.C50.EE.tr[i,4]<- Fmeasure(confusion.C50.EE.tr)
  
  for (t in 1:nrow(ts.credit)){
    if (score.ts[t] <=-1){pred.EE.ts[t] <- "pred.no"}
    else{pred.EE.ts[t] <- "pred.yes"} }

  
  conf.C50.EE.ts = table(ts.credit$default.payment.next.month, pred.EE.ts)
  confusion.C50.EE.ts <- conf.C50.EE.ts[, c("pred.yes", "pred.no")]
  table.C50.EE.ts[i,1]<- accuracy(confusion.C50.EE.ts)
  table.C50.EE.ts[i,2]<- precision(confusion.C50.EE.ts)
  table.C50.EE.ts[i,3]<- recall(confusion.C50.EE.ts)
  table.C50.EE.ts[i,4]<- Fmeasure(confusion.C50.EE.ts)
  
  
}

for (j in 1:4)
{
  #table1[cv+1,j]<-mean(table1[1:cv,j])
  table.C50.tr[cv+1,j]<-mean(table.C50.tr[1:cv,j])
  table.C50.ts[cv+1,j]<-mean(table.C50.ts[1:cv,j])
  table.C50.SE.tr[cv+1,j]<-mean(table.C50.SE.tr[1:cv,j])
  table.C50.SE.ts[cv+1,j]<-mean(table.C50.SE.ts[1:cv,j])
  table.C50.EE.tr[cv+1,j]<-mean(table.C50.EE.tr[1:cv,j])
  table.C50.EE.ts[cv+1,j]<-mean(table.C50.EE.ts[1:cv,j])
}

table.C50.tr
table.C50.ts
table.C50.SE.tr
table.C50.SE.ts
table.C50.EE.tr
table.C50.EE.ts


####function####
recall<- function(table) {table[1,1]/sum(table[1,])}
precision<- function(table) {table[1,1]/sum(table[,1])}
Fmeasure<- function(table) {2*recall(table)*precision(table)/(recall(table)+precision(table))}
accuracy<- function(table) {sum(diag(table))/sum(table)}


