#####Classification models based on paDEL descriptors using RF-VIM1 (mda)-ranked top 30 (corr. coeff. <0.75)
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM1_3")
library(data.table)
library(dplyr)
library(randomForest)
library(caret)

#RF-VIM
mtor.rfvim = readRDS("mtor.rfvim.RDS")
mtor.impvariable1 <- data.frame(importance(mtor.rfvim,type=1))
mtor.impvariable1$variable <- rownames (mtor.impvariable1)
mtor.impvariable1 <- mtor.impvariable1[order(mtor.impvariable1$MeanDecreaseAccuracy,decreasing = TRUE),]

summary(mtor.impvariable1)
write.csv(mtor.impvariable1,"mtor.impvariable1.csv", row.names = FALSE)
mtor.impvariable1<- read.csv("mtor.impvariable1.csv")
dim(mtor.impvariable1[mtor.impvariable1$MeanDecreaseAccuracy>3.156, ])

png("viplot.png") 
varImpPlot(mtor.rfvim, sort = TRUE, n.var = min(30,nrow(mtor.rfvim$importance)), type = NULL)
dev.off() 

mtor.vim1 <- (mtor.impvariable1[mtor.impvariable1$MeanDecreaseAccuracy>3.156, ])$variable

mtorfs <-read.csv("mtorfs.train2.csv")
colnames(mtorfs)
mtor.selecteddata <- mtorfs[  ,colnames(mtorfs) %in% mtor.vim1]
mtor.selecteddata <- mtorfs[  ,(colnames(mtorfs)) %in% c("Name", colnames(mtor.selecteddata), "label")]
dim(mtor.selecteddata)
write.csv(mtor.selecteddata,"mtor.selecteddata.csv", row.names = F)
mtor.selecteddata <- read.csv("mtor.selecteddata.csv")
dim(mtor.selecteddata)
##############correlation matrix 
# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values

flattenCorrMatrix2 <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  t=data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
    
  )
  return(t)
}

library(Hmisc)
mtor.selecteddata1 <- mtor.selecteddata[ ,!(colnames(mtor.selecteddata)) %in% c("Name","label")]
colnames(mtor.selecteddata1)
dim(mtor.selecteddata1)

xx <- colnames(mtor.selecteddata1)

res2<-rcorr(as.matrix(mtor.selecteddata1[,xx]))

tt=res2$P
flattenCorrMatrix2(res2$r, res2$P)

temp = flattenCorrMatrix2(res2$r, res2$P)
dim(temp)
write.csv(temp,'mtor_correlation.csv', row.names = FALSE)
head(temp)

tmp <- cor(mtor.selecteddata1)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
# #Above two commands can be replaced with 
#tmp[!lower.tri(tmp)] <- 0
data.new <- tmp[,!apply(tmp,2,function(x) any(abs(x) >= 0.75))]
dim(data.new)
colnames(data.new)
write.csv(data.new,'data.new.csv', row.names = FALSE)

library(corrplot)
#import java.util.List

png("corplot_mda30.png") 
corrplot(tmp, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 90)
dev.off() 

mtor.selecteddata <- read.csv("mtor.selecteddata.csv")
dim(mtor.selecteddata)
data.new <- read.csv('data.new.csv')
dim(data.new)
colnames(data.new)

mtor.selecteddata2 <- mtor.selecteddata[ ,(colnames(mtor.selecteddata)) %in% c("Name",colnames(data.new),"label")]
set.seed(123)

train <- mtor.selecteddata2[sample(nrow(mtor.selecteddata2)),]
dim(mtor.selecteddata2)
write.csv(train, "train.csv", row.names = FALSE)

##########################RF##SVM###DT#####NN#############################
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM1_3")
library(e1071)
library(randomForest)
library(kernlab)
library(pROC)
library(caret)

#################Trainingset################
train <- read.csv("train.csv")
dim(train)
train$y=ifelse(train$label==TRUE,"active","inactive")
train$y=as.factor(train$y)
train=train[,!(colnames(train)) %in% c("Name","label")]

#################Testset####################
mtorfs.test <-read.csv("mtorfs.test.csv")

mtorfs.test$y=ifelse(mtorfs.test$label==TRUE,"active","inactive")
mtorfs.test$y=as.factor(mtorfs.test$y)
testdata=mtorfs.test[,(colnames(train)) ]
colnames(testdata)

############################################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_rf30pp  <- train(y ~ ., 
                       data = train, ntree = 300, method = "rf", 
                       preProc = c("center", "scale"),
                       trControl = trctrl,
                       tunelength=10)
print(model_rf30pp)

saveRDS(model_rf30pp, file="model_rf30pp.RDS")
model_rf30pp = readRDS("model_rf30pp.RDS")

cm_rf30pp <-confusionMatrix(model_rf30pp$pred$pred, model_rf30pp$pred$obs, positive = "active")

tocsv_rf30pp <- data.frame(cbind(t(cm_rf30pp$overall),t(cm_rf30pp$byClass)))
write.csv(tocsv_rf30pp,file="file_rf30pp.csv")
write.table(cm_rf30pp$table,"cm_rf30pp.txt")


pred_rf30pptest <- predict(model_rf30pp, newdata = testdata, classProbs = TRUE)
print(table(testdata$y))
cm_rf30pptest <-confusionMatrix(pred_rf30pptest, testdata$y, positive = "active") 
tocsv_rf30pptest <- data.frame(cbind(t(cm_rf30pptest$overall),t(cm_rf30pptest$byClass)))
write.csv(tocsv_rf30pptest,file="tocsv_rf30pptest.csv")
write.table(cm_rf30pptest$table,"cm_rf30pptest.txt")

###############################SVM##############################################
### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = train, frac = 1)
## creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
#svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_svm30 <- train(y ~ .,
                     data = train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneGrid = svmTuneGrid,
                     trControl = trctrl,
                     tuneLength = 10)
print(model_svm30)



png("model_svm30.png") 
plot(model_svm30)
dev.off()

saveRDS(model_svm30, file="model_svm30.RDS")
model_svm30 = readRDS("model_svm30.RDS")

print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_svm30 <-confusionMatrix(model_svm30$pred$pred, model_svm30$pred$obs, positive = "active")
tocsv_svm30 <- data.frame(cbind(t(cm_svm30$overall),t(cm_svm30$byClass)))
write.csv(tocsv_svm30,file="file_svm30.csv")
write.table(cm_svm30$table,"cm_svm30.txt")

#########################################TEST SET

#mtorfs.testPsvm <-predict(model_svm40, ttdata, classProbs = TRUE)
svm_pred <- predict(model_svm30, newdata = testdata, classProbs = TRUE)
svm_cm30test <-confusionMatrix(svm_pred, testdata$y, positive = "active") 
svm_tocsvtest <- data.frame(cbind(t(svm_cm30test$overall),t(svm_cm30test$byClass)))
write.csv(svm_tocsvtest,file="svm_file30test.csv")
write.table(svm_cm30test$table,"svm_cm30test.txt")

#################################RP#########################################

library(rpart)
library(rattle)
library(rpart.plot)


trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
set.seed(123)
model_rp30 <- train(y ~., data = train, method = "rpart",
                    parms = list(split = "gini"),
                    preProc = c("center", "scale"),
                    trControl=trctrl,
                    tuneLength = 10)


plot(model_rp30)
print(model_rp30)

saveRDS(model_rp30, file="model_rp30.RDS")
model_rp30 = readRDS("model_rp30.RDS")


png("model_rp30.png") 
prp(model_rp30$finalModel, box.palette = "Blues", tweak = 1.2)
dev.off()

print(table(train$y))

cm_rp30 <-confusionMatrix(model_rp30$pred$pred, model_rp30$pred$obs, positive = "active")
tocsv_rp30 <- data.frame(cbind(t(cm_rp30$overall),t(cm_rp30$byClass)))
write.csv(tocsv_rp30,file="file_rp30.csv")
write.table(cm_rp30$table,"cm_rp30.txt")

#########################################TEST SET
rp30_pred <- predict(model_rp30, newdata = testdata, classProbs = TRUE)
rp30_cmtest <-confusionMatrix(rp30_pred, testdata$y, positive = "active") 
rp30_tocsvtest <- data.frame(cbind(t(rp30_cmtest$overall),t(rp30_cmtest$byClass)))
write.csv(rp30_tocsvtest,file="rp30_tocsvtest.csv")
write.table(rp30_cmtest$table,"rp30_cmtest.txt")

# Generate a textual view of the Decision Tree model.
print(model_rp30$finalModel)
printcp(model_rp30$finalModel)

# Decision Tree Plot...
prp(model_rp30$finalModel)
#dev.new()
png("fancymodel_rp30.png") 
fancyRpartPlot(model_rp30$finalModel, main="Decision Tree Graph")
dev.off()
##########################Neural NetWork#######################################################
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4,5,6,7,8))
model_nn30 <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trctrl, tuneLength = 10, maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    

print(model_nn30)

png("model_nn30.png") 
plot(model_nn30)
dev.off()

saveRDS(model_nn30, file="model_nn30.RDS")
model_nn30 = readRDS("model_nn30.RDS")

print(table(train$y))
cm_nn30 <-confusionMatrix(model_nn30$pred$pred, model_nn30$pred$obs, positive = "active")
tocsv_nn30 <- data.frame(cbind(t(cm_nn30$overall),t(cm_nn30$byClass)))
write.csv(tocsv_nn30,file="file_nn30.csv")
write.table(cm_nn30$table,"cm_nn30.txt")

#########################################TEST SET
nn30_pred <- predict(model_nn30, newdata = testdata, classProbs = TRUE)
nn30_cmtest <-confusionMatrix(nn30_pred, testdata$y, positive = "active") 
nn30_tocsvtest <- data.frame(cbind(t(nn30_cmtest$overall),t(nn30_cmtest$byClass)))
write.csv(nn30_tocsvtest,file="nn30_tocsvtest.csv")
write.table(nn30_cmtest$table,"nn30_cmtest.txt")

####################################EEEENNNNNNDDDDDDDD###############################################################
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM1_3")
library(e1071)
library(randomForest)
library(kernlab)
library(PRROC)
library(caret)

#################Trainingset################
train <- read.csv("train.csv")
dim(train)
train$y=ifelse(train$label==TRUE,"active","inactive")
train$y=as.factor(train$y)
train=train[,!(colnames(train)) %in% c("Name","label")]

#################Testset####################
mtorfs.test <-read.csv("mtorfs.test.csv")

mtorfs.test$y=ifelse(mtorfs.test$label==TRUE,"active","inactive")
mtorfs.test$y=as.factor(mtorfs.test$y)
testdata=mtorfs.test[,(colnames(train)) ]
colnames(testdata)

set.seed(123)
model_rf30pp = readRDS("model_rf30pp.RDS")
train <- read.csv("train.csv")
pr.rftr30 = predict(model_rf30pp, newdata = train[,!(colnames(train)) %in% c("Name","label")], type="prob")[,2]

train$label = as.numeric(train$label)
fgtr30 <- pr.rftr30[train$label == 1]
length(fgtr30)
set.seed(5)
fgtr30 <- rnorm(fgtr30)
bgtr30 <- pr.rftr30[train$label == 0]
length(bgtr30)
set.seed(5)
bgtr30 <- rnorm(bgtr30, -2)
roctr30 <- roc.curve(scores.class0 = fgtr30, scores.class1 = bgtr30, curve = T)
plot(roctr30)
png("roc_rftrain30.png") 
plot(roctr30)
dev.off()

prtr30 <- pr.curve(scores.class0 = fgtr30, scores.class1 = bgtr30, curve = T)
plot(prtr30)
png("pr_rftrain30.png") 
plot(prtr30)
dev.off()


pr.rftest = predict(model_rf30pp, newdata = testdata[,!(colnames(testdata)) %in% c("Name","label")], type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rf30 <- pr.rftest[mtorfs.test$label == 1]
length(fg_rf30)
set.seed(5)
fg_rf30 <- rnorm(fg_rf30)

bg_rf30 <- pr.rftest[mtorfs.test$label == 0]
length(bg_rf30)

set.seed(5)
bg_rf30 <- rnorm(bg_rf30, -2)

#ROC curve
roc_rf30 <- roc.curve(scores.class0 = fg_rf30, scores.class1 = bg_rf30, curve = T)
plot(roc_rf30)

png("roc_rftest30.png") 
plot(roc_rf30)
dev.off()

# PR Curve
pr_rf30 <- pr.curve(scores.class0 = fg_rf30, scores.class1 = bg_rf30, curve = T)
plot(pr_rf30)
png("pr_rftest30.png") 
plot(pr_rf30)
dev.off()


set.seed(123)
model_svm30 = readRDS("model_svm30.RDS")
pr.svmtr30 = predict(model_svm30, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrsvm30 <- pr.svmtr30[train$label == 1]
length(fgtrsvm30)
set.seed(54)
fgtrsvm30 <- rnorm(fgtrsvm30)
bgtrsvm30 <- pr.svmtr30[train$label == 0]
length(bgtrsvm30)
set.seed(54)
bgtrsvm30 <- rnorm(bgtrsvm30, -2)
roctrsvm30 <- roc.curve(scores.class0 = fgtrsvm30, scores.class1 = bgtrsvm30, curve = T)

plot(roctrsvm30)
png("roc_svmtrain30.png") 
plot(roctrsvm30)
dev.off()

prtrsvm30 <- pr.curve(scores.class0 = fgtrsvm30, scores.class1 = bgtrsvm30, curve = T)
plot(prtrsvm30)
png("pr_svmtrain30.png") 
plot(prtrsvm30)
dev.off()

pr.svmtest = predict(model_svm30, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_svm30 <- pr.svmtest[mtorfs.test$label == 1]
length(fg_svm30)
set.seed(34)
fg_svm30 <- rnorm(fg_svm30)

bg_svm30 <- pr.svmtest[mtorfs.test$label == 0]
length(bg_svm30)

set.seed(34)
bg_svm30 <- rnorm(bg_svm30, -2)

#ROC curve
roc_svm30 <- roc.curve(scores.class0 = fg_svm30, scores.class1 = bg_svm30, curve = T)
plot(roc_svm30)

png("roc_svmtest30.png") 
plot(roc_svm30)
dev.off()

# PR Curve
pr_svm30 <- pr.curve(scores.class0 = fg_svm30, scores.class1 = bg_svm30, curve = T)
plot(pr_svm30)
png("pr_svmtest30.png") 
plot(pr_svm30)
dev.off()

set.seed(123)
model_rp30 = readRDS("model_rp30.RDS")
pr.rptr30 = predict(model_rp30, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrrp30 <- pr.rptr30[train$label == 1]
length(fgtrrp30)
set.seed(42)
fgtrrp30 <- rnorm(fgtrrp30)
bgtrrp30 <- pr.rptr30[train$label == 0]
length(bgtrrp30)
set.seed(42)
bgtrrp30 <- rnorm(bgtrrp30, -2)
roctrrp30 <- roc.curve(scores.class0 = fgtrrp30, scores.class1 = bgtrrp30, curve = T)

plot(roctrrp30)
png("roc_rptrain30.png") 
plot(roctrrp30)
dev.off()

prtrrp30 <- pr.curve(scores.class0 = fgtrrp30, scores.class1 = bgtrrp30, curve = T)
plot(prtrrp30)
png("pr_rptrain30.png") 
plot(prtrrp30)
dev.off()


pr.rptest = predict(model_rp30, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rp30 <- pr.rptest[mtorfs.test$label == 1]
length(fg_rp30)
set.seed(24)
fg_rp30 <- rnorm(fg_rp30)

bg_rp30 <- pr.rptest[mtorfs.test$label == 0]
length(bg_rp30)

set.seed(24)
bg_rp30 <- rnorm(bg_rp30, -2)

#ROC curve
roc_rp30 <- roc.curve(scores.class0 = fg_rp30, scores.class1 = bg_rp30, curve = T)
plot(roc_rp30)

png("roc_rptest30.png") 
plot(roc_rp30)
dev.off()

# PR Curve
pr_rp30 <- pr.curve(scores.class0 = fg_rp30, scores.class1 = bg_rp30, curve = T)
plot(pr_rp30)
png("pr_rptest30.png") 
plot(pr_rp30)
dev.off()


set.seed(123)
model_nn30 = readRDS("model_nn30.RDS")
pr.nntr30 = predict(model_nn30, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnn30 <- pr.nntr30[train$label == 1]
length(fgtrnn30)
set.seed(51)
fgtrnn30 <- rnorm(fgtrnn30)
bgtrnn30 <- pr.nntr30[train$label == 0]
length(bgtrnn30)
set.seed(51)
bgtrnn30 <- rnorm(bgtrnn30, -2)
roctrnn30 <- roc.curve(scores.class0 = fgtrnn30, scores.class1 = bgtrnn30, curve = T)

plot(roctrnn30)
png("roc_nntrain30.png") 
plot(roctrnn30)
dev.off()

prtrnn30 <- pr.curve(scores.class0 = fgtrnn30, scores.class1 = bgtrnn30, curve = T)
plot(prtrnn30)
png("pr_nntrain30.png") 
plot(prtrnn30)
dev.off()

pr.nntest = predict(model_rp30, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nn30 <- pr.nntest[mtorfs.test$label == 1]
length(fg_nn30)
set.seed(15)
fg_nn30 <- rnorm(fg_nn30)

bg_nn30 <- pr.nntest[mtorfs.test$label == 0]
length(bg_nn30)

set.seed(15)
bg_nn30 <- rnorm(bg_nn30, -2)

#ROC curve
roc_nn30 <- roc.curve(scores.class0 = fg_nn30, scores.class1 = bg_nn30, curve = T)
plot(roc_nn30)

png("roc_nntest30.png") 
plot(roc_nn30)
dev.off()

# PR Curve
pr_nn30 <- pr.curve(scores.class0 = fg_nn30, scores.class1 = bg_nn30, curve = T)
plot(pr_nn30)
png("pr_nntest30.png") 
plot(pr_nn30)
dev.off()

#################################END##############################################


