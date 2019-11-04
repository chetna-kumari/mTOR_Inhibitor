######Classification models based on paDEL descriptors using RF-VIM2 (mdg)-ranked top 20 (corr. coeff. <0.75)
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM2_2")
library(data.table)
library(dplyr)
library(randomForest)
library(e1071)
library(caret)
library(PRROC) 
mtor.rfvim = readRDS("mtor.rfvim.RDS")
mtor.impvariable2 <- data.frame(importance(mtor.rfvim,type=2))
mtor.impvariable2$variable <- rownames (mtor.impvariable2)
mtor.impvariable2 <- mtor.impvariable2[order(mtor.impvariable2$MeanDecreaseGini,decreasing = TRUE),]
summary(mtor.impvariable2)
write.csv(mtor.impvariable2,"mtor.impvariable2.csv", row.names = FALSE)
mtor.impvariable2<- read.csv("mtor.impvariable2.csv")
dim(mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>2.1, ])
head(mtor.impvariable2, 20)

png("vi1plot.png") 
varImpPlot(mtor.rfvim, sort = TRUE, n.var = min(30,nrow(mtor.rfvim$importance)), type = 1)
dev.off() 

png("vi2plot.png") 
varImpPlot(mtor.rfvim, sort = TRUE, n.var = min(30,nrow(mtor.rfvim$importance)), type = 2)
dev.off() 

mtor.vim2 <- (mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>2.1, ])$variable
mtorfs <-read.csv("mtorfs.train2.csv")
#colnames(mtorfs)
mtor.selecteddata <- mtorfs[  ,colnames(mtorfs) %in% mtor.vim2]
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
write.csv(temp,'mtor_correlation.csv', row.names = F)
head(temp)

tmp <- cor(mtor.selecteddata1)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
# #Above two commands can be replaced with 
#tmp[!lower.tri(tmp)] <- 0
data.new <- tmp[,!apply(tmp,2,function(x) any(abs(x) >= 0.75))]
dim(data.new)
colnames(data.new)
write.csv(data.new,'data.new.csv', row.names = F)

library(corrplot)
#import java.util.List

png("corplot_mdg20.png") 
corrplot(tmp, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 90)
dev.off() 


mtor.selecteddata <- read.csv("mtor.selecteddata.csv")
data.new <- read.csv("data.new.csv")
dim(data.new)
colnames(data.new)

mtor.selecteddata2 <- mtor.selecteddata[ ,(colnames(mtor.selecteddata)) %in% c("Name",colnames(data.new),"label")]
set.seed(123)

train <- mtor.selecteddata2[sample(nrow(mtor.selecteddata2)),]
write.csv(train, "train.csv", row.names = FALSE)
colnames(train)

set.seed(123)
#################Trainingset################
train <- read.csv("train.csv")
train$y=ifelse(train$label==TRUE,"active","inactive")
train$y=as.factor(train$y)
train=train[,!(colnames(train)) %in% c("Name","label")]

#################Testset####################
mtorfs.test <-read.csv("mtorfs.test.csv")

mtorfs.test$y=ifelse(mtorfs.test$label==TRUE,"active","inactive")
mtorfs.test$y=as.factor(mtorfs.test$y)
testdata=mtorfs.test[,(colnames(train)) ]
colnames(testdata)

###################RF#########################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_rf20pp  <- train(y ~ ., 
                       data = train, ntree = 300, method = "rf", 
                       preProc = c("center", "scale"),
                       trControl = trctrl,
                       tunelength=10)
print(model_rf20pp)
png("mtor_rf20pp.png") 
plot(model_rf20pp)
dev.off()

saveRDS(model_rf20pp, file="model_rf20pp.RDS")
model_rf20pp = readRDS("model_rf20pp.RDS")

cm_rf20pp <-confusionMatrix(model_rf20pp$pred$pred, model_rf20pp$pred$obs, positive = "active")
tocsv_rf20pp <- data.frame(cbind(t(cm_rf20pp$overall),t(cm_rf20pp$byClass)))
write.csv(tocsv_rf20pp,file="file_rf20pp.csv")
write.table(cm_rf20pp$table,"cm_rf20pp.txt")

pred_rf20pptest <- predict(model_rf20pp, newdata = testdata, classProbs = TRUE)
print(table(testdata$y))
cm_rf20pptest <-confusionMatrix(pred_rf20pptest, testdata$y, positive = "active") 
tocsv_rf20pptest <- data.frame(cbind(t(cm_rf20pptest$overall),t(cm_rf20pptest$byClass)))
write.csv(tocsv_rf20pptest,file="tocsv_rf20pptest.csv")
write.table(cm_rf20pptest$table,"cm_rf20pptest.txt")
#####################################SVM####################################
### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = train, frac = 1)
## creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
#svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_svm20 <- train(y ~ .,
                     data = train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneGrid = svmTuneGrid,
                     trControl = trctrl,
                     tuneLength = 10)

print(model_svm20)

png("mtor_svm20.png") 
plot(model_svm20)
dev.off()

saveRDS(model_svm20, file="model_svm20.RDS")
model_svm20 = readRDS("model_svm20.RDS")

print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_svm20 <-confusionMatrix(model_svm20$pred$pred, model_svm20$pred$obs, positive = "active")
tocsv_svm20 <- data.frame(cbind(t(cm_svm20$overall),t(cm_svm20$byClass)))
write.csv(tocsv_svm20,file="file_svm20.csv")
write.table(cm_svm20$table,"cm_svm20.txt")

#########################################TEST SET
svm_pred <- predict(model_svm20, newdata = testdata, classProbs = TRUE)
svm_cm20test <-confusionMatrix(svm_pred, testdata$y, positive = "active") 
svm_tocsvtest <- data.frame(cbind(t(svm_cm20test$overall),t(svm_cm20test$byClass)))
write.csv(svm_tocsvtest,file="svm_file20test.csv")
write.table(svm_cm20test$table,"svm_cm20test.txt")

#################################RP#########################################
library(rpart)
library(rattle)
library(rpart.plot)

trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
set.seed(123)
model_rp20 <- train(y ~., data = train, method = "rpart",
                    parms = list(split = "gini"),
                    preProc = c("center", "scale"),
                    trControl=trctrl,
                    tuneLength = 10)


plot(model_rp20)
print(model_rp20)

saveRDS(model_rp20, file="model_rp20.RDS")
model_rfp0.RDS = readRDS("model_rp20.RDS")

png("model_rp20.png") 
prp(model_rp20$finalModel, box.palette = "Blues", tweak = 1.2)
dev.off()

print(table(train$y))
cm_rp20 <-confusionMatrix(model_rp20$pred$pred, model_rp20$pred$obs, positive = "active")
tocsv_rp20 <- data.frame(cbind(t(cm_rp20$overall),t(cm_rp20$byClass)))
write.csv(tocsv_rp20,file="file_rp20.csv")
write.table(cm_rp20$table,"cm_rp20.txt")

#########################################TEST SET
rp20_pred <- predict(model_rp20, newdata = testdata, classProbs = TRUE)
rp20_cmtest <-confusionMatrix(rp20_pred, testdata$y, positive = "active") 
rp20_tocsvtest <- data.frame(cbind(t(rp20_cmtest$overall),t(rp20_cmtest$byClass)))
write.csv(rp20_tocsvtest,file="rp20_tocsvtest.csv")
write.table(rp20_cmtest$table,"rp20_cmtest.txt")

# Generate a textual view of the Decision Tree model.
print(model_rp20$finalModel)
printcp(model_rp20$finalModel)

# Decision Tree Plot...
prp(model_rp20$finalModel)
#dev.new()
png("fancymodel_rp20.png") 
fancyRpartPlot(model_rp20$finalModel, main="Decision Tree Graph")
dev.off()

##########################Neural NetWork#######################################################
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4, 5, 6, 7, 8))
model_nn20 <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trainControl(method = "cv", number = 5, 
                                                                                                 classProbs =  TRUE, savePredictions = "final"), maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    
print(model_nn20)

saveRDS(model_nn20, file="model_nn20.RDS")
model_nn20.RDS = readRDS("model_nn20.RDS")

png("model_nn20.png") 
plot(model_nn20)
dev.off()

print(table(train$y))
cm_nn20 <-confusionMatrix(model_nn20$pred$pred, model_nn20$pred$obs, positive = "active")
tocsv_nn20 <- data.frame(cbind(t(cm_nn20$overall),t(cm_nn20$byClass)))
write.csv(tocsv_nn20,file="file_nn20.csv")
write.table(cm_nn20$table,"cm_nn20.txt")

#########################################TEST SET
nn20_pred <- predict(model_nn20, newdata = testdata, classProbs = TRUE)
nn20_cmtest <-confusionMatrix(nn20_pred, testdata$y, positive = "active") 
nn20_tocsvtest <- data.frame(cbind(t(nn20_cmtest$overall),t(nn20_cmtest$byClass)))
write.csv(nn20_tocsvtest,file="nn20_tocsvtest.csv")
write.table(nn20_cmtest$table,"nn20_cmtest.txt")

#################Trainingset################
train <- read.csv("train.csv")
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
model_rf20pp = readRDS("model_rf20pp.RDS")

train <- read.csv("train.csv")
pr.rftr20 = predict(model_rf20pp, newdata = train[,!(colnames(train)) %in% c("Name","label")], type="prob")[,2]
train$label = as.numeric(train$label)
fgtr20 <- pr.rftr20[train$label == 1]
length(fgtr20)
set.seed(12)
fgtr20 <- rnorm(fgtr20)
bgtr20 <- pr.rftr20[train$label == 0]
length(bgtr20)
set.seed(12)
bgtr20 <- rnorm(bgtr20, -2)
roctr20 <- roc.curve(scores.class0 = fgtr20, scores.class1 = bgtr20, curve = T)
plot(roctr20)
png("roc_rftrain20.png") 
plot(roctr20)
dev.off()

prtr20 <- pr.curve(scores.class0 = fgtr20, scores.class1 = bgtr20, curve = T)
plot(prtr20)
png("pr_rftrain20.png") 
plot(prtr20)
dev.off()


pr.rftest = predict(model_rf20pp, newdata = testdata[,!(colnames(testdata)) %in% c("Name","label")], type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rf20 <- pr.rftest[mtorfs.test$label == 1]
length(fg_rf20)
set.seed(14)
fg_rf20 <- rnorm(fg_rf20)

bg_rf20 <- pr.rftest[mtorfs.test$label == 0]
#bg_rf20 <-pred_rf20pptest[mtorfs.test$label == 0]
length(bg_rf20)

set.seed(12)
bg_rf20 <- rnorm(bg_rf20, -2)

#ROC curve
roc_rf20 <- roc.curve(scores.class0 = fg_rf20, scores.class1 = bg_rf20, curve = T)
plot(roc_rf20)

png("roc_rftest20.png") 
plot(roc_rf20)
dev.off()

# PR Curve
pr_rf20 <- pr.curve(scores.class0 = fg_rf20, scores.class1 = bg_rf20, curve = T)
plot(pr_rf20)
png("pr_rftest20.png") 
plot(pr_rf20)
dev.off()


set.seed(123)
model_svm20 = readRDS("model_svm20.RDS")
pr.svmtr20 = predict(model_svm20, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrsvm20 <- pr.svmtr20[train$label == 1]
length(fgtrsvm20)
set.seed(345)
fgtrsvm20 <- rnorm(fgtrsvm20)
bgtrsvm20 <- pr.svmtr20[train$label == 0]
length(bgtrsvm20)
set.seed(345)
bgtrsvm20 <- rnorm(bgtrsvm20, -2)
roctrsvm20 <- roc.curve(scores.class0 = fgtrsvm20, scores.class1 = bgtrsvm20, curve = T)

plot(roctrsvm20)
png("roc_svmtrain20.png") 
plot(roctrsvm20)
dev.off()

prtrsvm20 <- pr.curve(scores.class0 = fgtrsvm20, scores.class1 = bgtrsvm20, curve = T)
plot(prtrsvm20)
png("pr_svmtrain20.png") 
plot(prtrsvm20)
dev.off()

pr.svmtest = predict(model_svm20, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_svm20 <- pr.svmtest[mtorfs.test$label == 1]
length(fg_svm20)
set.seed(345)
fg_svm20 <- rnorm(fg_svm20)

bg_svm20 <- pr.svmtest[mtorfs.test$label == 0]
length(bg_svm20)

set.seed(345)
bg_svm20 <- rnorm(bg_svm20, -2)

#ROC curve
roc_svm20 <- roc.curve(scores.class0 = fg_svm20, scores.class1 = bg_svm20, curve = T)
plot(roc_svm20)

png("roc_svmtest20.png") 
plot(roc_svm20)
dev.off()

# PR Curve
pr_svm20 <- pr.curve(scores.class0 = fg_svm20, scores.class1 = bg_svm20, curve = T)
plot(pr_svm20)
png("pr_svmtest20.png") 
plot(pr_svm20)
dev.off()

set.seed(123)
model_rp20 = readRDS("model_rp20.RDS")
pr.rptr20 = predict(model_rp20, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrrp20 <- pr.rptr20[train$label == 1]
length(fgtrrp20)
set.seed(56)
fgtrrp20 <- rnorm(fgtrrp20)
bgtrrp20 <- pr.rptr20[train$label == 0]
length(bgtrrp20)
set.seed(56)
bgtrrp20 <- rnorm(bgtrrp20, -2)
roctrrp20 <- roc.curve(scores.class0 = fgtrrp20, scores.class1 = bgtrrp20, curve = T)

plot(roctrrp20)
png("roc_rptrain20.png") 
plot(roctrrp20)
dev.off()

prtrrp20 <- pr.curve(scores.class0 = fgtrrp20, scores.class1 = bgtrrp20, curve = T)
plot(prtrrp20)
png("pr_rptrain20.png") 
plot(prtrrp20)
dev.off()


pr.rptest = predict(model_rp20, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rp20 <- pr.rptest[mtorfs.test$label == 1]
length(fg_rp20)
set.seed(67)
fg_rp20 <- rnorm(fg_rp20)

bg_rp20 <- pr.rptest[mtorfs.test$label == 0]
length(bg_rp20)

set.seed(67)
bg_rp20 <- rnorm(bg_rp20, -2)

#ROC curve
roc_rp20 <- roc.curve(scores.class0 = fg_rp20, scores.class1 = bg_rp20, curve = T)
plot(roc_rp20)

png("roc_rptest20.png") 
plot(roc_rp20)
dev.off()

# PR Curve
pr_rp20 <- pr.curve(scores.class0 = fg_rp20, scores.class1 = bg_rp20, curve = T)
plot(pr_rp20)
png("pr_rptest20.png") 
plot(pr_rp20)
dev.off()


set.seed(123)
model_nn20 = readRDS("model_nn20.RDS")
pr.nntr20 = predict(model_nn20, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnn20 <- pr.nntr20[train$label == 1]
length(fgtrnn20)
set.seed(89)
fgtrnn20 <- rnorm(fgtrnn20)
bgtrnn20 <- pr.nntr20[train$label == 0]
length(bgtrnn20)
set.seed(89)
bgtrnn20 <- rnorm(bgtrnn20, -2)
roctrnn20 <- roc.curve(scores.class0 = fgtrnn20, scores.class1 = bgtrnn20, curve = T)

plot(roctrnn20)
png("roc_nntrain20.png") 
plot(roctrnn20)
dev.off()

prtrnn20 <- pr.curve(scores.class0 = fgtrnn20, scores.class1 = bgtrnn20, curve = T)
plot(prtrnn20)
png("pr_nntrain20.png") 
plot(prtrnn20)
dev.off()

pr.nntest = predict(model_rp20, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nn20 <- pr.nntest[mtorfs.test$label == 1]
length(fg_nn20)
set.seed(89)
fg_nn20 <- rnorm(fg_nn20)

bg_nn20 <- pr.nntest[mtorfs.test$label == 0]
length(bg_nn20)

set.seed(89)
bg_nn20 <- rnorm(bg_nn20, -2)

#ROC curve
roc_nn20 <- roc.curve(scores.class0 = fg_nn20, scores.class1 = bg_nn20, curve = T)
plot(roc_nn20)

png("roc_nntest20.png") 
plot(roc_nn20)
dev.off()

# PR Curve
pr_nn20 <- pr.curve(scores.class0 = fg_nn20, scores.class1 = bg_nn20, curve = T)
plot(pr_nn20)
png("pr_nntest20.png") 
plot(pr_nn20)
dev.off()
################################END###############################################