#MODEL BASED ON paDEL FP (AUTOENCODER) + ALL (RF,SVM,DT,NN) CLASSIFIER  
#setwd("~/Desktop/ML_mTOR/mTORFP")
library(data.table)
library(dplyr)
library(randomForest)
library(caret)
library(PRROC)
library (h2o)
library(foreach)
library(ForeCA)

localH2O = h2o.init()

mtordata1 <-read.csv("HM_IC50_SCORE_1808.csv", colClasses = c("character","NULL","NULL","numeric","NULL","NULL","NULL","NULL","NULL"))
mtordata <- unique( mtordata1 )

mtorfp <- read.csv('mtorpadel_MACCS.csv')
InputDatafp <- merge(mtorfp, mtordata, by.x=("Name"), by.y=("CMPD_CHEMBLID"))
InputDatafp$label <- FALSE
InputDatafp[InputDatafp$STANDARD_VALUE.IC50nM.<10000,]$label <- TRUE
table(InputDatafp$label)
write.csv(InputDatafp, "InputDatafp.csv", row.names = FALSE)
dim(InputDatafp)

table(InputDatafp$label)

inputData1=InputDatafp[,sapply(InputDatafp, function(v) var(v, na.rm=TRUE)!=0)]
dim(inputData1)
ncol(inputData1)

inputData = inputData1[,c(-1, -ncol(inputData1), -132)]
dim(inputData)
colnames(inputData)
table(sapply(inputData, class))

inputData = as.h2o(inputData)


h1=c(100,100,100)
deepNet.modelNew <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                     training_frame = inputData,
                                     l1=10^-5,
                                     activation="Tanh",
                                     autoencoder=T,
                                     hidden=h1,
                                     epochs=100,
                                     ignore_const_cols=F)    

save(deepNet.modelNew,file='deepNet.modelNew.RData')
deepNet.modelNew
encodedFeaturesNew<- h2o.deepfeatures(deepNet.modelNew,as.h2o(inputData), layer=2) # Take the features of 2nd layer
encodedFeaturesNew=as.data.frame(encodedFeaturesNew)
dim(encodedFeaturesNew)
reducedFeaturesWithIDandLabel=cbind(inputData1[,c(1,133)],encodedFeaturesNew)
write.csv(reducedFeaturesWithIDandLabel,'mtordeepfp.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel)
#########################################################
mtorfs <-read.csv("mtorfs.train2.csv")
dim(mtorfs)

mtordeepfp100<-read.csv("mtordeepfp100.csv")
dim(mtordeepfp100)
mtorfsdlfp <- merge(mtorfs[,c(1,2)], mtordeepfp100, by.x=("Name"), by.y=("Name"))
dim(mtorfsdlfp)

mtordlfp <- mtorfsdlfp[ ,!(colnames(mtorfsdlfp)) %in% c("nAcid")]
write.csv(mtordlfp,"mtordlfp.csv", row.names = FALSE)

set.seed(123)
mtordlfp <-read.csv("mtordlfp.csv")
dim(mtordlfp)
####################################
set.seed(123)

train <-mtordlfp[sample(nrow(mtordlfp)),]
#colnames(train)
write.csv(train,'train.csv', row.names = FALSE)
train <-read.csv(train.csv)
dim(train)

set.seed(123)

########################FP##RF##SVM###DT#####NN#############################
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
dim(mtorfs.test)
mtorfsdlfp.test <- merge(mtorfs.test[,c(1,2)], mtordeepfp100, by.x=("Name"), by.y=("Name"))
dim(mtorfsdlfp.test)

mtordlfp.test <- mtorfsdlfp.test[ ,!(colnames(mtorfsdlfp.test)) %in% c("nAcid")]
write.csv(mtordlfp.test,"mtordlfp.test.csv", row.names = FALSE)


mtordlfp.test <-read.csv("mtordlfp.test.csv")
dim(mtordlfp.test )
head(mtordlfp.test)
mtordlfp.test$y=ifelse(mtordlfp.test$label==TRUE,"active","inactive")
mtordlfp.test$y=as.factor(mtordlfp.test$y)
testdata=mtordlfp.test[,(colnames(train)) ]
dim(testdata)
colnames(testdata)

############################################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_rffp  <- train(y ~ ., 
                       data = train, ntree = 300, method = "rf", 
                       preProc = c("center", "scale"),
                       trControl = trctrl,
                       tunelength=10)
print(model_rffp)

png("model_rffp.png") 
plot(model_rffp)
dev.off()

plot(model_rffp)

saveRDS(model_rffp, file="model_rffp.RDS")
model_rffp.RDS = readRDS("model_rffp.RDS")

cm_rffp <-confusionMatrix(model_rffp$pred$pred, model_rffp$pred$obs, positive = "active")

tocsv_rffp <- data.frame(cbind(t(cm_rffp$overall),t(cm_rffp$byClass)))
write.csv(tocsv_rffp,file="file_rffp.csv")
write.table(cm_rffp$table,"cm_rffp.txt")

train <- read.csv("train.csv")
dim(train)
pr.rftrfp = predict(model_rffp, newdata = train[,!(colnames(train)) %in% c("Name","label")], type="prob")[,2]
train$label = as.numeric(train$label)
fgtrfp <- pr.rftrfp[train$label == 1]
length(fgtrfp)
set.seed(19)
fgtrfp <- rnorm(fgtrfp)
bgtrfp <- pr.rftrfp[train$label == 0]
length(bgtrfp)
set.seed(10)
bgtrfp <- rnorm(bgtrfp, -2)
roctrfp <- roc.curve(scores.class0 = fgtrfp, scores.class1 = bgtrfp, curve = T)
plot(roctrfp)
png("roc_rftrainfp.png") 
plot(roctrfp)
dev.off()

prtrfp <- pr.curve(scores.class0 = fgtrfp, scores.class1 = bgtrfp, curve = T)
plot(prtrfp)
png("pr_rftrainfp.png") 
plot(prtrfp)
dev.off()


pred_rffptest <- predict(model_rffp, newdata = testdata)
print(table(testdata$y))
cm_rffptest <-confusionMatrix(pred_rffptest, testdata$y, positive = "active") 
tocsv_rffptest <- data.frame(cbind(t(cm_rffptest$overall),t(cm_rffptest$byClass)))
write.csv(tocsv_rffptest,file="tocsv_rffptest.csv")
write.table(cm_rffptest$table,"cm_rffptest.txt")


pr.rftest = predict(model_rffp, newdata = testdata[,!(colnames(testdata)) %in% c("Name","label")], type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rffp <- pr.rftest[mtorfs.test$label == 1]
length(fg_rffp)
set.seed(20)
fg_rffp <- rnorm(fg_rffp)

bg_rffp <- pr.rftest[mtorfs.test$label == 0]
length(bg_rffp)

set.seed(19)
bg_rffp <- rnorm(bg_rffp, -2)

#ROC curve
roc_rffp <- roc.curve(scores.class0 = fg_rffp, scores.class1 = bg_rffp, curve = T)
plot(roc_rffp)

png("roc_rftestfp.png") 
plot(roc_rffp)
dev.off()

# PR Curve
pr_rffp <- pr.curve(scores.class0 = fg_rffp, scores.class1 = bg_rffp, curve = T)
plot(pr_rffp)
png("pr_rftestfp.png") 
plot(pr_rffp)
dev.off()

###############################SVM##############################################
### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = train, frac = 1)
## creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
#svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:2))
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
model_svmfp <- train(y ~ .,
                     data = train,
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     tuneGrid = svmTuneGrid,
                     trControl = trctrl,
                     tuneLength = 10)
print(model_svmfp)

png("model_svmfp.png") 
plot(model_svmfp)
dev.off()

saveRDS(model_svmfp, file="model_svmfp.RDS")
model_svmfp = readRDS("model_svmfp.RDS")

print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_svmfp <-confusionMatrix(model_svmfp$pred$pred, model_svmfp$pred$obs, positive = "active")
tocsv_svmfp <- data.frame(cbind(t(cm_svmfp$overall),t(cm_svmfp$byClass)))
write.csv(tocsv_svmfp,file="file_svmfp.csv")
write.table(cm_svmfp$table,"cm_svmfp.txt")

#########################################TEST SET
svm_pred <- predict(model_svmfp, newdata = testdata, classProb = TRUE)
svm_cmfptest <-confusionMatrix(svm_pred, testdata$y, positive = "active") 
svm_tocsvtest <- data.frame(cbind(t(svm_cmfptest$overall),t(svm_cmfptest$byClass)))
write.csv(svm_tocsvtest,file="file_svmfptest.csv")
write.table(svm_cmfptest$table,"svm_cmfptest.txt")

#################################RP#########################################
library(rpart)
library(rattle)
library(rpart.plot)


trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
set.seed(123)
model_rpfp <- train(y ~., data = train, method = "rpart",
                    parms = list(split = "gini"),
                    preProc = c("center", "scale"),
                    trControl=trctrl,
                    tuneLength = 10)
print(model_rpfp)

# model_rpfpi <- train(y ~., data = train, method = "rpart",
#                      parms = list(split = "information"),
#                      preProc = c("center", "scale"),
#                      trControl=trctrl,
#                      tuneLength = 10)


#print(model_rpfpi)

saveRDS(model_rpfp, file="model_rpfp.RDS")
model_rpfp.RDS = readRDS("model_rpfp.RDS")

png("model_rpfp.png")
plot(model_rpfp)
dev.off()

png("tree_rpfp.png") 
prp(model_rpfp$finalModel, box.palette = "Blues", tweak = 1.2)
dev.off()



print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_rpfp <-confusionMatrix(model_rpfp$pred$pred, model_rpfp$pred$obs, positive = "active")
tocsv_rpfp <- data.frame(cbind(t(cm_rpfp$overall),t(cm_rpfp$byClass)))
write.csv(tocsv_rpfp,file="file_rpfp.csv")
write.table(cm_rpfp$table,"cm_rpfp.txt")

#########################################TEST SET
rpfp_pred <- predict(model_rpfp, newdata = testdata, classProbs = TRUE)
rpfp_cmtest <-confusionMatrix(rpfp_pred, testdata$y, positive = "active") 
rpfp_tocsvtest <- data.frame(cbind(t(rpfp_cmtest$overall),t(rpfp_cmtest$byClass)))
write.csv(rpfp_tocsvtest,file="rpfp_tocsvtest.csv")
write.table(rpfp_cmtest$table,"rpfp_cmtest.txt")

# Generate a textual view of the Decision Tree model.
print(model_rpfp$finalModel)
printcp(model_rpfp$finalModel)

# Decision Tree Plot...
prp(model_rpfp$finalModel)

png("fancymodel_rpfp.png") 
fancyRpartPlot(model_rpfp$finalModel, main="Decision Tree Graph")
dev.off()
##########################Neural NetWork#######################################################
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4,5,6,7,8))
model_nnfp <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trctrl, tuneLength = 10, maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    

print(model_nnfp)
plot(model_nnfp)  

saveRDS(model_nnfp, file="model_nnfp.RDS")
model_nnfp.RDS = readRDS("model_nnfp.RDS")

png("model_nnfp.png") 
plot(model_nnfp)
dev.off()

print(table(train$y))
cm_nnfp <-confusionMatrix(model_nnfp$pred$pred, model_nnfp$pred$obs, positive = "active")
tocsv_nnfp <- data.frame(cbind(t(cm_nnfp$overall),t(cm_nnfp$byClass)))
write.csv(tocsv_nnfp,file="file_nnfp.csv")
write.table(cm_nnfp$table,"cm_nnfp.txt")
#########################################TEST SET
nnfp_pred <- predict(model_nnfp, newdata = testdata, classProbs = TRUE)
nnfp_cmtest <-confusionMatrix(nnfp_pred, testdata$y, positive = "active") 
nnfp_tocsvtest <- data.frame(cbind(t(nnfp_cmtest$overall),t(nnfp_cmtest$byClass)))
write.csv(nnfp_tocsvtest,file="nnfp_tocsvtest.csv")
write.table(nnfp_cmtest$table,"nnfp_cmtest.txt")

####################################END##################################
set.seed(123)
train<- read.csv("train.csv")
model_svmfp = readRDS("model_svmfp.RDS")
pr.svmtrfp = predict(model_svmfp, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrsvmfp <- pr.svmtrfp[train$label == 1]
length(fgtrsvmfp)
set.seed(60)
fgtrsvmfp <- rnorm(fgtrsvmfp)
bgtrsvmfp <- pr.svmtrfp[train$label == 0]
length(bgtrsvmfp)
set.seed(60)
bgtrsvmfp <- rnorm(bgtrsvmfp, -2)
roctrsvmfp <- roc.curve(scores.class0 = fgtrsvmfp, scores.class1 = bgtrsvmfp, curve = T)

plot(roctrsvmfp)
png("roc_svmtrainfp.png") 
plot(roctrsvmfp)
dev.off()

prtrsvmfp <- pr.curve(scores.class0 = fgtrsvmfp, scores.class1 = bgtrsvmfp, curve = T)
plot(prtrsvmfp)
png("pr_svmtrainfp.png") 
plot(prtrsvmfp)
dev.off()

pr.svmtest = predict(model_svmfp, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_svmfp <- pr.svmtest[mtorfs.test$label == 1]
length(fg_svmfp)
set.seed(50)
fg_svmfp <- rnorm(fg_svmfp)

bg_svmfp <- pr.svmtest[mtorfs.test$label == 0]
length(bg_svmfp)

set.seed(50)
bg_svmfp <- rnorm(bg_svmfp, -2)

#ROC curve
roc_svmfp <- roc.curve(scores.class0 = fg_svmfp, scores.class1 = bg_svmfp, curve = T)
plot(roc_svmfp)

png("roc_svmtestfp.png") 
plot(roc_svmfp)
dev.off()

# PR Curve
pr_svmfp <- pr.curve(scores.class0 = fg_svmfp, scores.class1 = bg_svmfp, curve = T)
plot(pr_svmfp)
png("pr_svmtestfp.png") 
plot(pr_svmfp)
dev.off()

set.seed(111)
model_rpfp = readRDS("model_rpfp.RDS")
pr.rptrfp = predict(model_rpfp, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrrpfp <- pr.rptrfp[train$label == 1]
length(fgtrrpfp)
set.seed(111)
fgtrrpfp <- rnorm(fgtrrpfp)
bgtrrpfp <- pr.rptrfp[train$label == 0]
length(bgtrrpfp)
set.seed(111)
bgtrrpfp <- rnorm(bgtrrpfp, -2)
roctrrpfp <- roc.curve(scores.class0 = fgtrrpfp, scores.class1 = bgtrrpfp, curve = T)

plot(roctrrpfp)
png("roc_rptrainfp.png") 
plot(roctrrpfp)
dev.off()

prtrrpfp <- pr.curve(scores.class0 = fgtrrpfp, scores.class1 = bgtrrpfp, curve = T)
plot(prtrrpfp)
png("pr_rptrainfp.png") 
plot(prtrrpfp)
dev.off()


pr.rptest = predict(model_rpfp, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_rpfp <- pr.rptest[mtorfs.test$label == 1]
length(fg_rpfp)
set.seed(801)
fg_rpfp <- rnorm(fg_rpfp)

bg_rpfp <- pr.rptest[mtorfs.test$label == 0]
length(bg_rpfp)

set.seed(821)
bg_rpfp <- rnorm(bg_rpfp, -2)

#ROC curve
roc_rpfp <- roc.curve(scores.class0 = fg_rpfp, scores.class1 = bg_rpfp, curve = T)
plot(roc_rpfp)

png("roc_rptestfp.png") 
plot(roc_rpfp)
dev.off()

# PR Curve
pr_rpfp <- pr.curve(scores.class0 = fg_rpfp, scores.class1 = bg_rpfp, curve = T)
plot(pr_rpfp)
png("pr_rptestfp.png") 
plot(pr_rpfp)
dev.off()

set.seed(123)
model_nnfp = readRDS("model_nnfp.RDS")
pr.nntrfp = predict(model_nnfp, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnnfp <- pr.nntrfp[train$label == 1]
length(fgtrnnfp)
set.seed(511)
fgtrnnfp <- rnorm(fgtrnnfp)
bgtrnnfp <- pr.nntrfp[train$label == 0]
length(bgtrnnfp)
set.seed(511)
bgtrnnfp <- rnorm(bgtrnnfp, -2)
roctrnnfp <- roc.curve(scores.class0 = fgtrnnfp, scores.class1 = bgtrnnfp, curve = T)

plot(roctrnnfp)
png("roc_nntrainfp.png") 
plot(roctrnnfp)
dev.off()

prtrnnfp <- pr.curve(scores.class0 = fgtrnnfp, scores.class1 = bgtrnnfp, curve = T)
plot(prtrnnfp)
png("pr_nntrainfp.png") 
plot(prtrnnfp)
dev.off()

pr.nntest = predict(model_rpfp, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nnfp <- pr.nntest[mtorfs.test$label == 1]
length(fg_nnfp)
set.seed(567)
fg_nnfp <- rnorm(fg_nnfp)

bg_nnfp <- pr.nntest[mtorfs.test$label == 0]
length(bg_nnfp)

set.seed(577)
bg_nnfp <- rnorm(bg_nnfp, -2)

#ROC curve
roc_nnfp <- roc.curve(scores.class0 = fg_nnfp, scores.class1 = bg_nnfp, curve = T)
plot(roc_nnfp)

png("roc_nntestfp.png") 
plot(roc_nnfp)
dev.off()

# PR Curve
pr_nnfp <- pr.curve(scores.class0 = fg_nnfp, scores.class1 = bg_nnfp, curve = T)
plot(pr_nnfp)
png("pr_nntestfp.png") 
plot(pr_nnfp)
dev.off()



########################END################################333



