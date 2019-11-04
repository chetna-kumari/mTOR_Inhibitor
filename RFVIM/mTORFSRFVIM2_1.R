######Classification models based on paDEL descriptors using RF-VIM2 (mdg)-ranked top 10 (corr. coeff. <0.75)
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM2_1")
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
dim(mtor.impvariable2)
summary(mtor.impvariable2)
dim(mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>3.6, ])
head(mtor.impvariable2,)
png("viplot.png") 

varImpPlot(mtor.rfvim, sort = TRUE, n.var = min(30,nrow(mtor.rfvim$importance)), type = NULL)

dev.off() 

mtor.vim2 <- (mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>3.6, ])$variable

mtorfs <-read.csv("mtorfs.train2.csv")

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
write.csv(temp,'mtor_correlation.csv')
head(temp)

tmp <- cor(mtor.selecteddata1)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
# #Above two commands can be replaced with 
#tmp[!lower.tri(tmp)] <- 0
data.new <- tmp[,!apply(tmp,2,function(x) any(abs(x) >= 0.75))]
dim(data.new)
colnames(data.new)
write.csv(data.new,'data.new.csv')

library(corrplot)
#import java.util.List

png("corplot_mdg10.png") 
corrplot(tmp, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 90)
dev.off() 

mtor.selecteddata <- read.csv("mtor.selecteddata.csv")
data.new <- read.csv("data.new.csv")
mtor.selecteddata2 <- mtor.selecteddata[ ,(colnames(mtor.selecteddata)) %in% c("Name",colnames(data.new),"label")]
set.seed(123)

train <- mtor.selecteddata2[sample(nrow(mtor.selecteddata2)),]
write.csv(train, "train.csv", row.names = FALSE)
read.csv("train.csv")
colnames(train)

###########################
set.seed(123)


library(randomForest)
library(kernlab)
library(PRROC)
library(caret)

#################Trainingset################
train <- read.csv("train.csv")
dim(train)
colnames(train)
train$y=ifelse(train$label==TRUE,"active","inactive")
train$y=as.factor(train$y)

#################Testset####################
mtorfs.test <-read.csv("mtorfs.test.csv")

mtorfs.test$y=ifelse(mtorfs.test$label==TRUE,"active","inactive")
mtorfs.test$y=as.factor(mtorfs.test$y)
testdata=mtorfs.test[,(colnames(train)) ]
colnames(testdata)

############################################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_rf10pp  <- train(y ~ ., 
                       data = train, ntree = 300, method = "rf", 
                       preProc = c("center", "scale"),
                       trControl = trctrl,
                       tuneLength = 10)
print(model_rf10pp)

saveRDS(model_rf10pp, file="model_rf10pp.RDS")
model_rf10pp.RDS = readRDS("model_rf10pp.RDS")

cm_rf10pp <-confusionMatrix(model_rf10pp$pred$pred, model_rf10pp$pred$obs, positive = "active")
tocsv_rf10pp <- data.frame(cbind(t(cm_rf10pp$overall),t(cm_rf10pp$byClass)))
write.csv(tocsv_rf10pp,file="file_rf10pp.csv")
write.table(cm_rf10pp$table,"cm_rf10pp.txt")

########ROC AUC   #### PR AUC   #######################
model_rf10pp = readRDS("model_rf10pp.RDS")
pr.rftr10 = predict(model_rf10pp, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtr10 <- pr.rftr10[train$label == 1]
length(fgtr10)
set.seed(135)
fgtr10 <- rnorm(fgtr10)
bgtr10 <- pr.rftr10[train$label == 0]
length(bgtr10)
set.seed(135)
bgtr10 <- rnorm(bgtr10, -2)
roctr10 <- roc.curve(scores.class0 = fgtr10, scores.class1 = bgtr10, curve = T)

png("roc_rftrain10.png") 
plot(roctr10)
dev.off()

prtr10 <- pr.curve(scores.class0 = fgtr10, scores.class1 = bgtr10, curve = T)

png("pr_rftrain10.png") 
plot(prtr10)
dev.off()

pred_rf10pptest <- predict(model_rf10pp, newdata = testdata, classProbs = TRUE)
print(table(testdata$y))
cm_rf10pptest <-confusionMatrix(pred_rf10pptest, testdata$y, positive = "active") 
tocsv_rf10pptest <- data.frame(cbind(t(cm_rf10pptest$overall),t(cm_rf10pptest$byClass)))
write.csv(tocsv_rf10pptest,file="tocsv_rf10pptest.csv")
write.table(cm_rf10pptest$table,"cm_rf10pptest.txt")

pr.rftest = predict(model_rf10pp, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rf10 <- pr.rftest[mtorfs.test$label == 1]
mtorfs.test$label = as.numeric(mtorfs.test$label)
length(fg_rf10)
set.seed(135)
fg_rf10 <- rnorm(fg_rf10)

bg_rf10 <- pr.rftest[mtorfs.test$label == 0]
length(bg_rf10)

set.seed(135)
bg_rf10 <- rnorm(bg_rf10, -2)

#ROC curve
roc_rf10 <- roc.curve(scores.class0 = fg_rf10, scores.class1 = bg_rf10, curve = T)
plot(roc_rf10)

png("roc_rftest10.png") 
plot(roc_rf10)
dev.off()

# PR Curve
pr_rf10 <- pr.curve(scores.class0 = fg_rf10, scores.class1 = bg_rf10, curve = T)
plot(pr_rf10)
png("pr_rftest10.png") 
plot(pr_rf10)
dev.off()

################SVM
### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = train, frac = 1)
## creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
#svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_svm10 <- train(y ~ .,
                     data = train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneGrid = svmTuneGrid,
                     trControl = trctrl,
                     tuneLength = 10)
print(model_svm10)


png("mtor_svm10.png") 
plot(model_svm10)
dev.off()

saveRDS(model_svm10, file="model_svm10.RDS")
model_svm10 = readRDS("model_svm10.RDS")

print(table(train$y))
cm_svm10 <-confusionMatrix(model_svm10$pred$pred, model_svm10$pred$obs, positive = "active")
tocsv_svm10 <- data.frame(cbind(t(cm_svm10$overall),t(cm_svm10$byClass)))
write.csv(tocsv_svm10,file="file_svm10.csv")
write.table(cm_svm10$table,"cm_svm10.txt")

set.seed(123)
model_svm10 = readRDS("model_svm10.RDS")
pr.svmtr10 = predict(model_svm10, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrsvm10 <- pr.svmtr10[train$label == 1]
length(fgtrsvm10)
set.seed(357)
fgtrsvm10 <- rnorm(fgtrsvm10)
bgtrsvm10 <- pr.svmtr10[train$label == 0]
length(bgtrsvm10)
set.seed(357)
bgtrsvm10 <- rnorm(bgtrsvm10, -2)
roctrsvm10 <- roc.curve(scores.class0 = fgtrsvm10, scores.class1 = bgtrsvm10, curve = T)

plot(roctrsvm10)
png("roc_svmtrain10.png") 
plot(roctrsvm10)
dev.off()

prtrsvm10 <- pr.curve(scores.class0 = fgtrsvm10, scores.class1 = bgtrsvm10, curve = T)
plot(prtrsvm10)
png("pr_svmtrain10.png") 
plot(prtrsvm10)
dev.off()


svm_pred <- predict(model_svm10, newdata = testdata, classProbs = TRUE)
svm_cm10test <-confusionMatrix(svm_pred, testdata$y, positive = "active") 
svm_tocsvtest <- data.frame(cbind(t(svm_cm10test$overall),t(svm_cm10test$byClass)))
write.csv(svm_tocsvtest,file="svm_file10test.csv")
write.table(svm_cm10test$table,"svm_cm10test.txt")

pr.svmtest = predict(model_svm10, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

#cmSVM10test <-confusionMatrix(pr.svmtest, mtorfs.test$y, positive = "active") 

fg_svm10 <- pr.svmtest[mtorfs.test$label == 1]
length(fg_svm10)
set.seed(357)
fg_svm10 <- rnorm(fg_svm10)

bg_svm10 <- pr.svmtest[mtorfs.test$label == 0]
length(bg_svm10)

set.seed(357)
bg_svm10 <- rnorm(bg_svm10, -2)

#ROC curve
roc_svm10 <- roc.curve(scores.class0 = fg_svm10, scores.class1 = bg_svm10, curve = T)
plot(roc_svm10)

png("roc_svmtest10.png") 
plot(roc_svm10)
dev.off()

# PR Curve
pr_svm10 <- pr.curve(scores.class0 = fg_svm10, scores.class1 = bg_svm10, curve = T)
plot(pr_svm10)
png("pr_svmtest10.png") 
plot(pr_svm10)
dev.off()

#################################RP#########################################
library(rpart)
library(rattle)
library(rpart.plot)

trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
set.seed(123)
model_rp10 <- train(y ~., data = train, method = "rpart",
                    parms = list(split = "gini"),
                    preProc = c("center", "scale"),
                    trControl=trctrl,
                    tuneLength = 10)

saveRDS(model_rp10, file="model_rp10.RDS")
model_rfp10.RDS = readRDS("model_rp10.RDS")

plot(model_rp10)
print(model_rp10)


png("model_rp10.png") 
prp(model_rp10$finalModel, box.palette = "Blues", tweak = 1.2)
dev.off()



print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_rp10 <-confusionMatrix(model_rp10$pred$pred, model_rp10$pred$obs, positive = "active")
tocsv_rp10 <- data.frame(cbind(t(cm_rp10$overall),t(cm_rp10$byClass)))
write.csv(tocsv_rp10,file="file_rp10.csv")
write.table(cm_rp10$table,"cm_rp10.txt")

set.seed(123)
model_rp10 = readRDS("model_rp10.RDS")
pr.rptr10 = predict(model_rp10, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrrp10 <- pr.rptr10[train$label == 1]
length(fgtrrp10)
set.seed(579)
fgtrrp10 <- rnorm(fgtrrp10)
bgtrrp10 <- pr.rptr10[train$label == 0]
length(bgtrrp10)
set.seed(579)
bgtrrp10 <- rnorm(bgtrrp10, -2)
roctrrp10 <- roc.curve(scores.class0 = fgtrrp10, scores.class1 = bgtrrp10, curve = T)

plot(roctrrp10)
png("roc_rptrain10.png") 
plot(roctrrp10)
dev.off()

prtrrp10 <- pr.curve(scores.class0 = fgtrrp10, scores.class1 = bgtrrp10, curve = T)
plot(prtrrp10)
png("pr_rptrain10.png") 
plot(prtrrp10)
dev.off()

rp10_pred <- predict(model_rp10, newdata = testdata, classProbs = TRUE)
rp10_cmtest <-confusionMatrix(rp10_pred, testdata$y, positive = "active") 
rp10_tocsvtest <- data.frame(cbind(t(rp10_cmtest$overall),t(rp10_cmtest$byClass)))
write.csv(rp10_tocsvtest,file="rp10_tocsvtest.csv")
write.table(rp10_cmtest$table,"rp10_cmtest.txt")

pr.rptest = predict(model_rp10, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rp10 <- pr.rptest[mtorfs.test$label == 1]
length(fg_rp10)
set.seed(579)
fg_rp10 <- rnorm(fg_rp10)

bg_rp10 <- pr.rptest[mtorfs.test$label == 0]
length(bg_rp10)

set.seed(579)
bg_rp10 <- rnorm(bg_rp10, -2)

#ROC curve
roc_rp10 <- roc.curve(scores.class0 = fg_rp10, scores.class1 = bg_rp10, curve = T)
plot(roc_rp10)

png("roc_rptest10.png") 
plot(roc_rp10)
dev.off()

# PR Curve
pr_rp10 <- pr.curve(scores.class0 = fg_rp10, scores.class1 = bg_rp10, curve = T)
plot(pr_rp10)
png("pr_rptest10.png") 
plot(pr_rp10)
dev.off()


# Generate a textual view of the Decision Tree model.
print(model_rp10$finalModel)
printcp(model_rp10$finalModel)

# Decision Tree Plot...
prp(model_rp10$finalModel)
#dev.new()
png("fancymodel_rp10.png") 
fancyRpartPlot(model_rp10$finalModel, main="Decision Tree Graph")
dev.off()
##########################Neural NetWork#######################################################
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4,5,6,7,8))
set.seed(123)
model_nn10 <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trctrl, tuneLength = 10, maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    

print(model_nn10)

saveRDS(model_nn10, file="model_nn10.RDS")
model_nn10 = readRDS("model_nn10.RDS")

png("model_nn10.png") 
plot(model_nn10)
dev.off()

print(table(train$y))
cm_nn10 <-confusionMatrix(model_nn10$pred$pred, model_nn10$pred$obs, positive = "active")
tocsv_nn10 <- data.frame(cbind(t(cm_nn10$overall),t(cm_nn10$byClass)))
write.csv(tocsv_nn10,file="file_nn10.csv")
write.table(cm_nn10$table,"cm_nn10.txt")


set.seed(123)
model_nn10 = readRDS("model_nn10.RDS")
pr.nntr10 = predict(model_nn10, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnn10 <- pr.nntr10[train$label == 1]
length(fgtrnn10)
set.seed(753)
fgtrnn10 <- rnorm(fgtrnn10)
bgtrnn10 <- pr.nntr10[train$label == 0]
length(bgtrnn10)
set.seed(753)
bgtrnn10 <- rnorm(bgtrnn10, -2)
roctrnn10 <- roc.curve(scores.class0 = fgtrnn10, scores.class1 = bgtrnn10, curve = T)

plot(roctrnn10)
png("roc_nntrain10.png") 
plot(roctrnn10)
dev.off()

prtrnn10 <- pr.curve(scores.class0 = fgtrnn10, scores.class1 = bgtrnn10, curve = T)
plot(prtrnn10)
png("pr_nntrain10.png") 
plot(prtrnn10)
dev.off()

nn10_pred <- predict(model_nn10, newdata = testdata, classProbs = TRUE)
nn10_cmtest <-confusionMatrix(nn10_pred, testdata$y, positive = "active") 
nn10_tocsvtest <- data.frame(cbind(t(nn10_cmtest$overall),t(nn10_cmtest$byClass)))
write.csv(nn10_tocsvtest,file="nn10_tocsvtest.csv")
write.table(nn10_cmtest$table,"nn10_cmtest.txt")


pr.nntest = predict(model_nn10, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nn10 <- pr.nntest[mtorfs.test$label == 1]
length(fg_nn10)
set.seed(753)
fg_nn10 <- rnorm(fg_nn10)

bg_nn10 <- pr.nntest[mtorfs.test$label == 0]
length(bg_nn10)

set.seed(753)
bg_nn10 <- rnorm(bg_nn10, -2)

#ROC curve
roc_nn10 <- roc.curve(scores.class0 = fg_nn10, scores.class1 = bg_nn10, curve = T)
plot(roc_nn10)

png("roc_nntest10.png") 
plot(roc_nn10)
dev.off()

# PR Curve
pr_nn10 <- pr.curve(scores.class0 = fg_nn10, scores.class1 = bg_nn10, curve = T)
plot(pr_nn10)
png("pr_nntest10.png") 
plot(pr_nn10)
dev.off()
####################################END###############################################################