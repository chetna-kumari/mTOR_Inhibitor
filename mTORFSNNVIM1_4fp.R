#####MODEL BASED ON paDEL descriptors Hybrid Feature set (RF VIM1-ranked top 40 + Autoencoder reduced fps)
setwd("~/Desktop/ML_mTOR/mTORFSNNVIM1_4fp")
library(data.table)
library(randomForest)
library(caret)
library(PRROC)
mtor.selecteddata<-read.csv("mtor.selecteddata.csv")
dim(mtor.selecteddata)
colnames(mtor.selecteddata)
data.new <-read.csv("data.new.csv")
colnames(data.new)

mtor.selecteddata2 <- mtor.selecteddata[ ,(colnames(mtor.selecteddata)) %in% c("Name",colnames(data.new))]
dim(mtor.selecteddata2)
mtordeepfp100 <- read.csv("mtordeepfp100.csv")
dim(mtordeepfp100)
InputDatatrain <- merge(mtordeepfp100, mtor.selecteddata2, by.x=("Name"), by.y=("Name"))
dim(InputDatatrain)
colnames(InputDatatrain)
set.seed(123)
train <- InputDatatrain[sample(nrow(InputDatatrain)),]
colnames(InputDatatrain)
write.csv(train, "train.csv", row.names=FALSE)
train <- read.csv("train.csv")

#train <- read.csv("train.csv")
colnames(train)
train$y=ifelse(train$label==TRUE,"active","inactive")
train$y=as.factor(train$y)
train=train[,!(colnames(train)) %in% c("Name","label")]

#####################TEST DATA
mtorfs.test2<-read.csv('mtorfs.test2.csv')
mtordeepfp100<-read.csv("mtordeepfp100.csv")
#colnames(mtordeepfp100)
mtordeepfp100<-mtordeepfp100[, -2]
dim(mtordeepfp100)
colnames(mtordeepfp100)
#dim(mtorfs)
mtorfs40fp.test <- merge(mtordeepfp100, mtorfs.test2, by.x=("Name"), by.y=("Name"))
dim(mtorfs40fp.test)
write.csv(mtorfs40fp.test,"mtorfs40fp.test.csv", row.names = FALSE)

mtorfs40fp.test <-read.csv("mtorfs40fp.test.csv")
mtorfs40fp.test$y=ifelse(mtorfs.test$label==TRUE,"active","inactive")
mtorfs40fp.test$y=as.factor(mtorfs40fp.test$y)
testdata=mtorfs40fp.test[,(colnames(train)) ]
colnames(testdata)

######################NN HYBRID MODEL
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5, classProbs =  TRUE, savePredictions = "final")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4,5,6,7,8))
model_nnfsfp <- train(y ~ ., data = train,
                    method = "nnet", preProcess = c('center', 'scale'), trControl = trctrl, tuneLength = 10, maxit = 100, tuneGrid =  my.grid, trace = F, linout = 0)    

print(model_nnfsfp)
plot(model_nnfsfp)  

saveRDS(model_nnfp, file="model_nnfsfp.RDS")
model_nnfsfp.RDS = readRDS("model_nnfsfp.RDS")

png("model_nnfsfp.png") 
plot(model_nnfsfp)
dev.off()

print(table(train$y))
#cm_svm40 <-confusionMatrix(model_svm40$pred[order(model_svm40$pred$rowIndex),3], train$y, positive = "OK")
cm_nnfsfp <-confusionMatrix(model_nnfsfp$pred$pred, model_nnfsfp$pred$obs, positive = "active")
tocsv_nnfsfp <- data.frame(cbind(t(cm_nnfsfp$overall),t(cm_nnfsfp$byClass)))
write.csv(tocsv_nnfsfp,file="file_nnfsfp.csv")
write.table(cm_nnfsfp$table,"cm_nnfsfp.txt")
#########################################TEST SET
nnfsfp_pred <- predict(model_nnfsfp, newdata = testdata, classProbs = TRUE)
nnfsfp_cmtest <-confusionMatrix(nnfsfp_pred, testdata$y, positive = "active") 
nnfsfp_tocsvtest <- data.frame(cbind(t(nnfsfp_cmtest$overall),t(nnfsfp_cmtest$byClass)))
write.csv(nnfsfp_tocsvtest,file="nnfsfp_tocsvtest.csv")
write.table(nnfsfp_cmtest$table,"nnfsfp_cmtest.txt")

######################ROC & PR AUC  #####################
set.seed(123)
train<- read.csv("train.csv")
model_nnfsfp = readRDS("model_nnfsfp.RDS")
pr.nntrfsfp = predict(model_nnfsfp, newdata = train, type="prob")[,2]
train$label = as.numeric(train$label)
fgtrnnfsfp <- pr.nntrfsfp[train$label == 1]
length(fgtrnnfsfp)
set.seed(511)
fgtrnnfsfp <- rnorm(fgtrnnfsfp)
bgtrnnfsfp <- pr.nntrfsfp[train$label == 0]
length(bgtrnnfsfp)
set.seed(511)
bgtrnnfsfp <- rnorm(bgtrnnfsfp, -2)
roctrnnfsfp <- roc.curve(scores.class0 = fgtrnnfsfp, scores.class1 = bgtrnnfsfp, curve = T)

plot(roctrnnfsfp)
png("roc_nntrainfsfp.png") 
plot(roctrnnfsfp)
dev.off()

prtrnnfsfp <- pr.curve(scores.class0 = fgtrnnfsfp, scores.class1 = bgtrnnfsfp, curve = T)
plot(prtrnnfsfp)
png("pr_nntrainfsfp.png") 
plot(prtrnnfsfp)
dev.off()

pr.nntest = predict(model_nnfsfp, newdata = testdata, type = "prob")[,2]
mtorfs.test$label = as.numeric(mtorfs.test$label)

fg_nnfsfp <- pr.nntest[mtorfs.test$label == 1]
length(fg_nnfsfp)
set.seed(567)
fg_nnfsfp <- rnorm(fg_nnfsfp)

bg_nnfsfp <- pr.nntest[mtorfs.test$label == 0]
length(bg_nnfsfp)

set.seed(577)
bg_nnfsfp <- rnorm(bg_nnfsfp, -2)

#ROC curve
roc_nnfsfp <- roc.curve(scores.class0 = fg_nnfsfp, scores.class1 = bg_nnfsfp, curve = T)
plot(roc_nnfsfp)

png("roc_nntestfsfp.png") 
plot(roc_nnfsfp)
dev.off()

# PR Curve
pr_nnfsfp <- pr.curve(scores.class0 = fg_nnfsfp, scores.class1 = bg_nnfsfp, curve = T)
plot(pr_nnfsfp)
png("pr_nntestfsfp.png") 
plot(pr_nnfsfp)
dev.off()

#######################END###############