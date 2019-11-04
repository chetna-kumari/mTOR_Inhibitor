######Classification models based on paDEL descriptors using RF-VIM2 (mdg)-ranked top 40 (corr. coeff. <0.75)
setwd("~/Desktop/ML_mTOR/mTORFSRFVIM2_4")
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

summary(mtor.impvariable2)
dim(mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>1.2, ])
#head(mtor.impvariable2,)
png("viplot.png") 

varImpPlot(mtor.rfvim, sort = TRUE, n.var = min(30,nrow(mtor.rfvim$importance)), type = NULL)

dev.off() 
mtor.vim2 <- (mtor.impvariable2[mtor.impvariable2$MeanDecreaseGini>1.2, ])$variable
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
data.new <- tmp[,!apply(tmp,2,function(x) any(abs(x)>= 0.75))]
dim(data.new)
colnames(data.new)
write.csv(data.new,'data.new.csv')

library(corrplot)
#import java.util.List

png("corplot_mdg40.png") 
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
############################################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_rf40pp  <- train(y ~ ., 
                       data = train, ntree = 300, method = "rf", 
                       preProc = c("center", "scale"),
                       trControl = trctrl,
                       tunelength=10)
print(model_rf40pp)

png("model_rf40pp.png") 
plot(model_rf40pp)
dev.off()

saveRDS(model_rf40pp, file="model_rf40pp.RDS")
model_rf40pp.RDS = readRDS("model_rf40pp.RDS")

cm_rf40pp <-confusionMatrix(model_rf40pp$pred$pred, model_rf40pp$pred$obs, positive = "active")

tocsv_rf40pp <- data.frame(cbind(t(cm_rf40pp$overall),t(cm_rf40pp$byClass)))
write.csv(tocsv_rf40pp,file="file_rf40pp.csv")
write.table(cm_rf40pp$table,"cm_rf40pp.txt")

pred_rf40pptest <- predict(model_rf40pp, newdata = testdata)
print(table(testdata$y))
cm_rf40pptest <-confusionMatrix(pred_rf40pptest, testdata$y, positive = "active") 
tocsv_rf40pptest <- data.frame(cbind(t(cm_rf40pptest$overall),t(cm_rf40pptest$byClass)))
write.csv(tocsv_rf40pptest,file="tocsv_rf40pptest.csv")
write.table(cm_rf40pptest$table,"cm_rf40pptest.txt")

###############################SVM##############################################
### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = train, frac = 1)
## creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
#svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_svm40 <- train(y ~ .,
                     data = train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneGrid = svmTuneGrid,
                     trControl = trctrl,
                     tuneLength = 10)
print(model_svm40)

png("model_svm40.png") 
plot(model_svm40)
dev.off()

saveRDS(model_svm40, file="model_svm40.RDS")
model_svm40 = readRDS("model_svm40.RDS")

print(table(train$y))
cm_svm40 <-confusionMatrix(model_svm40$pred$pred, model_svm40$pred$obs, positive = "active")
tocsv_svm40 <- data.frame(cbind(t(cm_svm40$overall),t(cm_svm40$byClass)))
write.csv(tocsv_svm40,file="file_svm40.csv")
write.table(cm_svm40$table,"cm_svm40.txt")

svm_pred <- predict(model_svm40, newdata = testdata, classProb = TRUE)
svm_cm40test <-confusionMatrix(svm_pred, testdata$y, positive = "active") 
svm_tocsvtest <- data.frame(cbind(t(svm_cm40test$overall),t(svm_cm40test$byClass)))
write.csv(svm_tocsvtest,file="svm_file40test.csv")
write.table(svm_cm40test$table,"svm_cm40test.txt")

#################################RP#########################################

library(rpart)
library(rattle)
library(rpart.plot)

trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
set.seed(123)
model_rp40 <- train(y ~., data = train, method = "rpart",
                    parms = list(split = "gini"),
                    preProc = c("center", "scale"),
                    trControl=trctrl,
                    tuneLength = 10)
print(model_rp40)

# model_rp40i <- train(y ~., data = train, method = "rpart",
#                      parms = list(split = "information"),
#                      preProc = c("center", "scale"),
#                      trControl=trctrl,
#                      tuneLength = 10)
# 
# 
# print(model_rp40i)

saveRDS(model_rp40, file="model_rp40.RDS")
model_rp40.RDS = readRDS("model_rp40.RDS")


png("model_rp40.png")
plot(model_rp40)
dev.off()

png("tree_rp40.png") 
prp(model_rp40$finalModel, box.palette = "Blues", tweak = 1.2)
dev.off()

print(table(train$y))
cm_rp40 <-confusionMatrix(model_rp40$pred$pred, model_rp40$pred$obs, positive = "active")
tocsv_rp40 <- data.frame(cbind(t(cm_rp40$overall),t(cm_rp40$byClass)))
write.csv(tocsv_rp40,file="file_rp40.csv")
write.table(cm_rp40$table,"cm_rp40.txt")

rp40_pred <- predict(model_rp40, newdata = testdata, classProbs = TRUE)
rp40_cmtest <-confusionMatrix(rp40_pred, testdata$y, positive = "active") 
rp40_tocsvtest <- data.frame(cbind(t(rp40_cmtest$overall),t(rp40_cmtest$byClass)))
write.csv(rp40_tocsvtest,file="rp40_tocsvtest.csv")
write.table(rp40_cmtest$table,"rp40_cmtest.txt")

# Generate a textual view of the Decision Tree model.
print(model_rp40$finalModel)
printcp(model_rp40$finalModel)

# Decision Tree Plot...
prp(model_rp40$finalModel)

png("fancymodel_rp40.png") 
fancyRpartPlot(model_rp40$finalModel, main="Decision Tree Graph")
dev.off()
##########################Neural NetWork#######################################################
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4,5,6,7,8))
model_nn40 <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trctrl, tuneLength = 10, maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    

print(model_nn40)
plot(model_nn40)  

saveRDS(model_nn40, file="model_nn40.RDS")
model_nn40.RDS = readRDS("model_nn40.RDS")

png("model_nn40.png") 
plot(model_nn40)
dev.off()

print(table(train$y))
cm_nn40 <-confusionMatrix(model_nn40$pred$pred, model_nn40$pred$obs, positive = "active")
tocsv_nn40 <- data.frame(cbind(t(cm_nn40$overall),t(cm_nn40$byClass)))
write.csv(tocsv_nn40,file="file_nn40.csv")
write.table(cm_nn40$table,"cm_nn40.txt")

nn40_pred <- predict(model_nn40, newdata = testdata, classProbs = TRUE)
nn40_cmtest <-confusionMatrix(nn40_pred, testdata$y, positive = "active") 
nn40_tocsvtest <- data.frame(cbind(t(nn40_cmtest$overall),t(nn40_cmtest$byClass)))
write.csv(nn40_tocsvtest,file="nn40_tocsvtest.csv")
write.table(nn40_cmtest$table,"nn40_cmtest.txt")

##################################################
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
model_rf40pp = readRDS("model_rf40pp.RDS")
train <- read.csv("train.csv")
pr.rftr40 = predict(model_rf40pp, newdata = train[,!(colnames(train)) %in% c("Name","label")], type="prob")[,2]
train$label = as.numeric(train$label)
fgtr40 <- pr.rftr40[train$label == 1]
length(fgtr40)
set.seed(702)
fgtr40 <- rnorm(fgtr40)
bgtr40 <- pr.rftr40[train$label == 0]
length(bgtr40)
set.seed(702)
bgtr40 <- rnorm(bgtr40, -2)
roctr40 <- roc.curve(scores.class0 = fgtr40, scores.class1 = bgtr40, curve = T)
plot(roctr40)
png("roc_rftrain40.png") 
plot(roctr40)
dev.off()

prtr40 <- pr.curve(scores.class0 = fgtr40, scores.class1 = bgtr40, curve = T)
plot(prtr40)
png("pr_rftrain40.png") 
plot(prtr40)
dev.off()


pr.rftest = predict(model_rf40pp, newdata = testdata[,!(colnames(testdata)) %in% c("Name","label")], type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rf40 <- pr.rftest[mtorfs.test$label == 1]
length(fg_rf40)
set.seed(705)
fg_rf40 <- rnorm(fg_rf40)

bg_rf40 <- pr.rftest[mtorfs.test$label == 0]
#bg_rf40 <-pred_rf40pptest[mtorfs.test$label == 0]
length(bg_rf40)

set.seed(705)
bg_rf40 <- rnorm(bg_rf40, -2)

#ROC curve
roc_rf40 <- roc.curve(scores.class0 = fg_rf40, scores.class1 = bg_rf40, curve = T)
plot(roc_rf40)

png("roc_rftest40.png") 
plot(roc_rf40)
dev.off()

# PR Curve
pr_rf40 <- pr.curve(scores.class0 = fg_rf40, scores.class1 = bg_rf40, curve = T)
plot(pr_rf40)
png("pr_rftest40.png") 
plot(pr_rf40)
dev.off()


set.seed(123)
model_svm40 = readRDS("model_svm40.RDS")
pr.svmtr40 = predict(model_svm40, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrsvm40 <- pr.svmtr40[train$label == 1]
length(fgtrsvm40)
set.seed(605)
fgtrsvm40 <- rnorm(fgtrsvm40)
bgtrsvm40 <- pr.svmtr40[train$label == 0]
length(bgtrsvm40)
set.seed(605)
bgtrsvm40 <- rnorm(bgtrsvm40, -2)
roctrsvm40 <- roc.curve(scores.class0 = fgtrsvm40, scores.class1 = bgtrsvm40, curve = T)

plot(roctrsvm40)
png("roc_svmtrain40.png") 
plot(roctrsvm40)
dev.off()

prtrsvm40 <- pr.curve(scores.class0 = fgtrsvm40, scores.class1 = bgtrsvm40, curve = T)
plot(prtrsvm40)
png("pr_svmtrain40.png") 
plot(prtrsvm40)
dev.off()

pr.svmtest = predict(model_svm40, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_svm40 <- pr.svmtest[mtorfs.test$label == 1]
length(fg_svm40)
set.seed(569)
fg_svm40 <- rnorm(fg_svm40)

bg_svm40 <- pr.svmtest[mtorfs.test$label == 0]
length(bg_svm40)

set.seed(569)
bg_svm40 <- rnorm(bg_svm40, -2)

#ROC curve
roc_svm40 <- roc.curve(scores.class0 = fg_svm40, scores.class1 = bg_svm40, curve = T)
plot(roc_svm40)

png("roc_svmtest40.png") 
plot(roc_svm40)
dev.off()

# PR Curve
pr_svm40 <- pr.curve(scores.class0 = fg_svm40, scores.class1 = bg_svm40, curve = T)
plot(pr_svm40)
png("pr_svmtest40.png") 
plot(pr_svm40)
dev.off()

set.seed(123)
model_rp40 = readRDS("model_rp40.RDS")
pr.rptr40 = predict(model_rp40, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrrp40 <- pr.rptr40[train$label == 1]
length(fgtrrp40)
set.seed(445)
fgtrrp40 <- rnorm(fgtrrp40)
bgtrrp40 <- pr.rptr40[train$label == 0]
length(bgtrrp40)
set.seed(445)
bgtrrp40 <- rnorm(bgtrrp40, -2)
roctrrp40 <- roc.curve(scores.class0 = fgtrrp40, scores.class1 = bgtrrp40, curve = T)

plot(roctrrp40)
png("roc_rptrain40.png") 
plot(roctrrp40)
dev.off()

prtrrp40 <- pr.curve(scores.class0 = fgtrrp40, scores.class1 = bgtrrp40, curve = T)
plot(prtrrp40)
png("pr_rptrain40.png") 
plot(prtrrp40)
dev.off()


pr.rptest = predict(model_rp40, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rp40 <- pr.rptest[mtorfs.test$label == 1]
length(fg_rp40)
set.seed(19)
fg_rp40 <- rnorm(fg_rp40)

bg_rp40 <- pr.rptest[mtorfs.test$label == 0]
length(bg_rp40)

set.seed(19)
bg_rp40 <- rnorm(bg_rp40, -2)

#ROC curve
roc_rp40 <- roc.curve(scores.class0 = fg_rp40, scores.class1 = bg_rp40, curve = T)
plot(roc_rp40)

png("roc_rptest40.png") 
plot(roc_rp40)
dev.off()

# PR Curve
pr_rp40 <- pr.curve(scores.class0 = fg_rp40, scores.class1 = bg_rp40, curve = T)
plot(pr_rp40)
png("pr_rptest40.png") 
plot(pr_rp40)
dev.off()

set.seed(123)
model_nn40 = readRDS("model_nn40.RDS")
pr.nntr40 = predict(model_nn40, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnn40 <- pr.nntr40[train$label == 1]
length(fgtrnn40)
set.seed(21)
fgtrnn40 <- rnorm(fgtrnn40)
bgtrnn40 <- pr.nntr40[train$label == 0]
length(bgtrnn40)
set.seed(21)
bgtrnn40 <- rnorm(bgtrnn40, -2)
roctrnn40 <- roc.curve(scores.class0 = fgtrnn40, scores.class1 = bgtrnn40, curve = T)

plot(roctrnn40)
png("roc_nntrain40.png") 
plot(roctrnn40)
dev.off()

prtrnn40 <- pr.curve(scores.class0 = fgtrnn40, scores.class1 = bgtrnn40, curve = T)
plot(prtrnn40)
png("pr_nntrain40.png") 
plot(prtrnn40)
dev.off()

pr.nntest = predict(model_rp40, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nn40 <- pr.nntest[mtorfs.test$label == 1]
length(fg_nn40)
set.seed(91)
fg_nn40 <- rnorm(fg_nn40)

bg_nn40 <- pr.nntest[mtorfs.test$label == 0]
length(bg_nn40)

set.seed(91)
bg_nn40 <- rnorm(bg_nn40, -2)

#ROC curve
roc_nn40 <- roc.curve(scores.class0 = fg_nn40, scores.class1 = bg_nn40, curve = T)
plot(roc_nn40)

png("roc_nntest40.png") 
plot(roc_nn40)
dev.off()

# PR Curve
pr_nn40 <- pr.curve(scores.class0 = fg_nn40, scores.class1 = bg_nn40, curve = T)
plot(pr_nn40)
png("pr_nntest40.png") 
plot(pr_nn40)
dev.off()

#######################################END###############


