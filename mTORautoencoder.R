#AUTOENCODER for dimension reduction of MACCS-FP 
library(data.table)
library(dplyr)
#library(xlsx)
setwd("~/Desktop/ML_mTOR/mTORautoencoder")
#****************************************************#*******************************************
#mtordata1 <-read.csv("HM_IC50_SCORE_1808.csv", colClasses = c("character","NULL","NULL","numeric","NULL","NULL","NULL","NULL","NULL"))
#mtordata <- unique( mtordata1 )

#mtorfp <- read.csv('mtorpadel_MACCS.csv')
#InputDatafp <- merge(mtorfp, mtordata, by.x=("Name"), by.y=("CMPD_CHEMBLID"))
#InputDatafp$label <- FALSE
#InputDatafp[InputDatafp$STANDARD_VALUE.IC50nM.<10000,]$label <- TRUE
#table(InputDatafp$label)

write.csv(InputDatafp, "InputDatafp.csv", row.names = FALSE)

# InputDatafp<- read.csv("InputDatafp.csv")
# dim(InputDatafp)
# 
# table(InputDatafp$label)
# 
# inputData1=InputDatafp[,sapply(InputDatafp, function(v) var(v, na.rm=TRUE)!=0)]
# dim(inputData1)
# ncol(inputData1)
# 
# write.csv(inputData1, "inputData1.csv", row.names = FALSE)

# inputData = inputData1[,c(-1, -ncol(inputData1), -132)]
# dim(inputData)
# colnames(inputData)
# table(sapply(inputData, class))
# write.csv(inputData, "inputData.csv", row.names = FALSE)
##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 10:10:10)##################################
library(data.table)
library(dplyr)
#library(xlsx)
setwd("~/Desktop/ML_mTOR/mTORautoencoder")

library (h2o)
library(foreach)
library(ForeCA)
# localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, Xmx = '12g')
localH2O = h2o.init()

inputData1<- read.csv("inputData1.csv")
dim(inputData1)

inputData<- read.csv("inputData.csv")
dim(inputData)
#colnames(inputData)

inputData = as.h2o(inputData)

h1=c(10,10,10)
set.seed(135)
deepNetmodelNew10 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                     training_frame = inputData,
                                     l1=10^-5,
                                     activation="Tanh",
                                     #activation="Rectifier",
                                     autoencoder=T,
                                     hidden=h1,
                                     epochs=100,
                                     ignore_const_cols=F)    
saveRDS(deepNetmodelNew10,file='deepNetmodelNew10.RDS')
deepNetmodelNew10<- readRDS('deepNetmodelNew10.RDS')
#save(deepNetmodelNew10,file='deepNetmodelNew10.RData')
#deepNetmodelNew10<- load("deepNetmodelNew10.RData")
encodedFeaturesNew10<- h2o.deepfeatures(deepNetmodelNew10,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew10=as.data.frame(encodedFeaturesNew10)
dim(encodedFeaturesNew10)
reducedFeaturesWithIDandLabel10=cbind(inputData1[,c(1,133)],encodedFeaturesNew10)
write.csv(reducedFeaturesWithIDandLabel10,'mtordeepfp10.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel10)
##############################################################
library(Metrics)
mod.out10 <- as.data.frame(h2o.predict(deepNetmodelNew10, inputData, type=response))
mod.out10 = as.h2o(mod.out10)

MeanAbsErr10 <-mae(as.matrix(inputData),as.matrix(mod.out10))
MeanAbsPercErr10 <-mape(as.matrix(inputData),as.matrix(mod.out10))
SymMeanPercErr10 <-smape(as.matrix(inputData),as.matrix(mod.out10))
Vartot10<-var(as.matrix(inputData)) # variance total
Varres10<-var(as.matrix(inputData-mod.out10)) # variance residual
Rsq10 <- 1- (sum(Varres10)/sum(Vartot10))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew10)
print(MeanAbsErr10)
print(MeanAbsPercErr10)
print(SymMeanPercErr10)
print(Rsq10)

#fpp.anon = h2o.anomaly(deepNetmodelNew10, inputData, per_feature=FALSE)
#err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
#plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
#plot(err$Reconstruction.MSE, main='Reconstruction Error')
##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 20:20:20)####
h1=c(20,20,20)
set.seed(135)
deepNetmodelNew20 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew20,file='deepNetmodelNew20.RDS')
deepNetmodelNew20<- readRDS('deepNetmodelNew20.RDS')
#save(deepNetmodelNew20,file='deepNetmodelNew20.RData')
#deepNetmodelNew20<- load("deepNetmodelNew20.RData")
encodedFeaturesNew20<- h2o.deepfeatures(deepNetmodelNew20,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew20=as.data.frame(encodedFeaturesNew20)
dim(encodedFeaturesNew20)
reducedFeaturesWithIDandLabel20=cbind(inputData1[,c(1,133)],encodedFeaturesNew20)
write.csv(reducedFeaturesWithIDandLabel20,'mtordeepfp20.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel20)

library(Metrics)
mod.out20 <- as.data.frame(h2o.predict(deepNetmodelNew20, inputData, type=response))
mod.out20 = as.h2o(mod.out20)

MeanAbsErr20 <-mae(as.matrix(inputData),as.matrix(mod.out20))
MeanAbsPercErr20 <-mape(as.matrix(inputData),as.matrix(mod.out20))
SymMeanPercErr20 <-smape(as.matrix(inputData),as.matrix(mod.out20))
Vartot20<-var(as.matrix(inputData)) # variance total
Varres20<-var(as.matrix(inputData-mod.out20)) # variance residual
Rsq20 <- 1- (sum(Varres20)/sum(Vartot20))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew20)
print(MeanAbsErr20)
print(MeanAbsPercErr20)
print(SymMeanPercErr20)
print(Rsq20)

fpp.anon = h2o.anomaly(deepNetmodelNew20, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 30:30:30)############
h1=c(30,30,30)
set.seed(135)
deepNetmodelNew30 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew30,file='deepNetmodelNew30.RDS')
deepNetmodelNew30<- readRDS('deepNetmodelNew30.RDS')
#save(deepNetmodelNew30,file='deepNetmodelNew30.RData')
#deepNetmodelNew30<- load("deepNetmodelNew30.RData")
encodedFeaturesNew30<- h2o.deepfeatures(deepNetmodelNew30,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew30=as.data.frame(encodedFeaturesNew30)
dim(encodedFeaturesNew30)
reducedFeaturesWithIDandLabel30=cbind(inputData1[,c(1,133)],encodedFeaturesNew30)
write.csv(reducedFeaturesWithIDandLabel30,'mtordeepfp30.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel30)

library(Metrics)
mod.out30 <- as.data.frame(h2o.predict(deepNetmodelNew30, inputData, type=response))
mod.out30 = as.h2o(mod.out30)

MeanAbsErr30 <-mae(as.matrix(inputData),as.matrix(mod.out30))
MeanAbsPercErr30 <-mape(as.matrix(inputData),as.matrix(mod.out30))
SymMeanPercErr30 <-smape(as.matrix(inputData),as.matrix(mod.out30))
Vartot30<-var(as.matrix(inputData)) # variance total
Varres30<-var(as.matrix(inputData-mod.out30)) # variance residual
Rsq30 <- 1- (sum(Varres30)/sum(Vartot30))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew30)
print(MeanAbsErr30)
print(MeanAbsPercErr30)
print(SymMeanPercErr30)
print(Rsq30)

fpp.anon = h2o.anomaly(deepNetmodelNew30, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 40:40:40)############
h1=c(40,40,40)
set.seed(135)
deepNetmodelNew40 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew40,file='deepNetmodelNew40.RDS')
deepNetmodelNew40<- readRDS('deepNetmodelNew40.RDS')
#save(deepNetmodelNew40,file='deepNetmodelNew40.RData')
#deepNetmodelNew40<- load("deepNetmodelNew40.RData")
encodedFeaturesNew40<- h2o.deepfeatures(deepNetmodelNew40,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew40=as.data.frame(encodedFeaturesNew40)
dim(encodedFeaturesNew40)
reducedFeaturesWithIDandLabel40=cbind(inputData1[,c(1,133)],encodedFeaturesNew40)
write.csv(reducedFeaturesWithIDandLabel40,'mtordeepfp40.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel40)

library(Metrics)
mod.out40 <- as.data.frame(h2o.predict(deepNetmodelNew40, inputData, type=response))
mod.out40 = as.h2o(mod.out40)

MeanAbsErr40 <-mae(as.matrix(inputData),as.matrix(mod.out40))
MeanAbsPercErr40 <-mape(as.matrix(inputData),as.matrix(mod.out40))
SymMeanPercErr40 <-smape(as.matrix(inputData),as.matrix(mod.out40))
Vartot40<-var(as.matrix(inputData)) # variance total
Varres40<-var(as.matrix(inputData-mod.out40)) # variance residual
Rsq40 <- 1- (sum(Varres40)/sum(Vartot40))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew40)
print(MeanAbsErr40)
print(MeanAbsPercErr40)
print(SymMeanPercErr40)
print(Rsq40)

fpp.anon = h2o.anomaly(deepNetmodelNew40, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 50:50:50)############
h1=c(50,50,50)
set.seed(135)
deepNetmodelNew50 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew50,file='deepNetmodelNew50.RDS')
deepNetmodelNew50<- readRDS('deepNetmodelNew50.RDS')
#save(deepNetmodelNew50,file='deepNetmodelNew50.RData')
#deepNetmodelNew50<- load("deepNetmodelNew50.RData")
encodedFeaturesNew50<- h2o.deepfeatures(deepNetmodelNew50,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew50=as.data.frame(encodedFeaturesNew50)
dim(encodedFeaturesNew50)
reducedFeaturesWithIDandLabel50=cbind(inputData1[,c(1,133)],encodedFeaturesNew50)
write.csv(reducedFeaturesWithIDandLabel50,'mtordeepfp50.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel50)

library(Metrics)
mod.out50 <- as.data.frame(h2o.predict(deepNetmodelNew50, inputData, type=response))
mod.out50 = as.h2o(mod.out50)

MeanAbsErr50 <-mae(as.matrix(inputData),as.matrix(mod.out50))
MeanAbsPercErr50 <-mape(as.matrix(inputData),as.matrix(mod.out50))
SymMeanPercErr50 <-smape(as.matrix(inputData),as.matrix(mod.out50))
Vartot50<-var(as.matrix(inputData)) # variance total
Varres50<-var(as.matrix(inputData-mod.out50)) # variance residual
Rsq50 <- 1- (sum(Varres50)/sum(Vartot50))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew50)
print(MeanAbsErr50)
print(MeanAbsPercErr50)
print(SymMeanPercErr50)
print(Rsq50)

fpp.anon = h2o.anomaly(deepNetmodelNew50, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 60:60:60)############
h1=c(60,60,60)
set.seed(135)
deepNetmodelNew60 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew60,file='deepNetmodelNew60.RDS')
deepNetmodelNew60<- readRDS('deepNetmodelNew60.RDS')
#save(deepNetmodelNew60,file='deepNetmodelNew60.RData')
#deepNetmodelNew60<- load("deepNetmodelNew60.RData")
encodedFeaturesNew60<- h2o.deepfeatures(deepNetmodelNew60,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew60=as.data.frame(encodedFeaturesNew60)
dim(encodedFeaturesNew60)
reducedFeaturesWithIDandLabel60=cbind(inputData1[,c(1,133)],encodedFeaturesNew60)
write.csv(reducedFeaturesWithIDandLabel60,'mtordeepfp60.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel60)

library(Metrics)
mod.out60 <- as.data.frame(h2o.predict(deepNetmodelNew60, inputData, type=response))
mod.out60 = as.h2o(mod.out60)

MeanAbsErr60 <-mae(as.matrix(inputData),as.matrix(mod.out60))
MeanAbsPercErr60 <-mape(as.matrix(inputData),as.matrix(mod.out60))
SymMeanPercErr60 <-smape(as.matrix(inputData),as.matrix(mod.out60))
Vartot60<-var(as.matrix(inputData)) # variance total
Varres60<-var(as.matrix(inputData-mod.out60)) # variance residual
Rsq60 <- 1- (sum(Varres60)/sum(Vartot60))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew60)
print(MeanAbsErr60)
print(MeanAbsPercErr60)
print(SymMeanPercErr60)
print(Rsq60)

fpp.anon = h2o.anomaly(deepNetmodelNew60, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')

##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 70:70:70)########
h1=c(70,70,70)
set.seed(135)
deepNetmodelNew70 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew70,file='deepNetmodelNew70.RDS')
deepNetmodelNew70<- readRDS('deepNetmodelNew70.RDS')
#save(deepNetmodelNew70,file='deepNetmodelNew70.RData')
#deepNetmodelNew70<- load("deepNetmodelNew70.RData")
encodedFeaturesNew70<- h2o.deepfeatures(deepNetmodelNew70,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew70=as.data.frame(encodedFeaturesNew70)
dim(encodedFeaturesNew70)
reducedFeaturesWithIDandLabel70=cbind(inputData1[,c(1,133)],encodedFeaturesNew70)
write.csv(reducedFeaturesWithIDandLabel70,'mtordeepfp70.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel70)

library(Metrics)
mod.out70 <- as.data.frame(h2o.predict(deepNetmodelNew70, inputData, type=response))
mod.out70 = as.h2o(mod.out70)

MeanAbsErr70 <-mae(as.matrix(inputData),as.matrix(mod.out70))
MeanAbsPercErr70 <-mape(as.matrix(inputData),as.matrix(mod.out70))
SymMeanPercErr70 <-smape(as.matrix(inputData),as.matrix(mod.out70))
Vartot70<-var(as.matrix(inputData)) # variance total
Varres70<-var(as.matrix(inputData-mod.out70)) # variance residual
Rsq70 <- 1- (sum(Varres70)/sum(Vartot70))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew70)
print(MeanAbsErr70)
print(MeanAbsPercErr70)
print(SymMeanPercErr70)
print(Rsq70)

fpp.anon = h2o.anomaly(deepNetmodelNew70, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')

##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 80:80:80)########
h1=c(80,80,80)
set.seed(135)
deepNetmodelNew80 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew80,file='deepNetmodelNew80.RDS')
deepNetmodelNew80<- readRDS('deepNetmodelNew80.RDS')
#save(deepNetmodelNew80,file='deepNetmodelNew80.RData')
#deepNetmodelNew80<- load("deepNetmodelNew80.RData")
encodedFeaturesNew80<- h2o.deepfeatures(deepNetmodelNew80,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew80=as.data.frame(encodedFeaturesNew80)
dim(encodedFeaturesNew80)
reducedFeaturesWithIDandLabel80=cbind(inputData1[,c(1,133)],encodedFeaturesNew80)
write.csv(reducedFeaturesWithIDandLabel80,'mtordeepfp80.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel80)

library(Metrics)
mod.out80 <- as.data.frame(h2o.predict(deepNetmodelNew80, inputData, type=response))
mod.out80 = as.h2o(mod.out80)

MeanAbsErr80 <-mae(as.matrix(inputData),as.matrix(mod.out80))
MeanAbsPercErr80 <-mape(as.matrix(inputData),as.matrix(mod.out80))
SymMeanPercErr80 <-smape(as.matrix(inputData),as.matrix(mod.out80))
Vartot80<-var(as.matrix(inputData)) # variance total
Varres80<-var(as.matrix(inputData-mod.out80)) # variance residual
Rsq80 <- 1- (sum(Varres80)/sum(Vartot80))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew80)
print(MeanAbsErr80)
print(MeanAbsPercErr80)
print(SymMeanPercErr80)
print(Rsq80)

fpp.anon = h2o.anomaly(deepNetmodelNew80, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')



##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 90:90:90)########
h1=c(90,90,90)
set.seed(135)
deepNetmodelNew90 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew90,file='deepNetmodelNew90.RDS')
deepNetmodelNew90<- readRDS('deepNetmodelNew90.RDS')
#save(deepNetmodelNew90,file='deepNetmodelNew90.RData')
#deepNetmodelNew90<- load("deepNetmodelNew90.RData")
encodedFeaturesNew90<- h2o.deepfeatures(deepNetmodelNew90,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew90=as.data.frame(encodedFeaturesNew90)
dim(encodedFeaturesNew90)
reducedFeaturesWithIDandLabel90=cbind(inputData1[,c(1,133)],encodedFeaturesNew90)
write.csv(reducedFeaturesWithIDandLabel90,'mtordeepfp90.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel90)

library(Metrics)
mod.out90 <- as.data.frame(h2o.predict(deepNetmodelNew90, inputData, type=response))
mod.out90 = as.h2o(mod.out90)

MeanAbsErr90 <-mae(as.matrix(inputData),as.matrix(mod.out90))
MeanAbsPercErr90 <-mape(as.matrix(inputData),as.matrix(mod.out90))
SymMeanPercErr90 <-smape(as.matrix(inputData),as.matrix(mod.out90))
Vartot90<-var(as.matrix(inputData)) # variance total
Varres90<-var(as.matrix(inputData-mod.out90)) # variance residual
Rsq90 <- 1- (sum(Varres90)/sum(Vartot90))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew90)
print(MeanAbsErr90)
print(MeanAbsPercErr90)
print(SymMeanPercErr90)
print(Rsq90)

fpp.anon = h2o.anomaly(deepNetmodelNew90, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 100:100:100)########
h1=c(100,100,100)
set.seed(135)
deepNetmodelNew100 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                      training_frame = inputData,
                                      l1=10^-5,
                                      activation="Tanh",
                                      #activation="Rectifier",
                                      autoencoder=T,
                                      hidden=h1,
                                      epochs=100,
                                      ignore_const_cols=F)    
saveRDS(deepNetmodelNew100,file='deepNetmodelNew100.RDS')
deepNetmodelNew100<- readRDS('deepNetmodelNew100.RDS')
#save(deepNetmodelNew100,file='deepNetmodelNew100.RData')
#deepNetmodelNew100<- load("deepNetmodelNew100.RData")
encodedFeaturesNew100<- h2o.deepfeatures(deepNetmodelNew100,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew100=as.data.frame(encodedFeaturesNew100)
dim(encodedFeaturesNew100)
reducedFeaturesWithIDandLabel100=cbind(inputData1[,c(1,133)],encodedFeaturesNew100)
write.csv(reducedFeaturesWithIDandLabel100,'mtordeepfp100.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel100)

library(Metrics)
mod.out100 <- as.data.frame(h2o.predict(deepNetmodelNew100, inputData, type=response))
mod.out100 = as.h2o(mod.out100)

MeanAbsErr100 <-mae(as.matrix(inputData),as.matrix(mod.out100))
MeanAbsPercErr100 <-mape(as.matrix(inputData),as.matrix(mod.out100))
SymMeanPercErr100 <-smape(as.matrix(inputData),as.matrix(mod.out100))
Vartot100<-var(as.matrix(inputData)) # variance total
Varres100<-var(as.matrix(inputData-mod.out100)) # variance residual
Rsq100 <- 1- (sum(Varres100)/sum(Vartot100))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew100)
print(MeanAbsErr100)
print(MeanAbsPercErr100)
print(SymMeanPercErr100)
print(Rsq100)

fpp.anon = h2o.anomaly(deepNetmodelNew100, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 110:110:110)########
h1=c(110,110,110)
set.seed(135)
deepNetmodelNew110 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                       training_frame = inputData,
                                       l1=10^-5,
                                       activation="Tanh",
                                       #activation="Rectifier",
                                       autoencoder=T,
                                       hidden=h1,
                                       epochs=100,
                                       ignore_const_cols=F)    
saveRDS(deepNetmodelNew110,file='deepNetmodelNew110.RDS')
deepNetmodelNew110<- readRDS('deepNetmodelNew110.RDS')
#save(deepNetmodelNew110,file='deepNetmodelNew110.RData')
#deepNetmodelNew110<- load("deepNetmodelNew110.RData")
encodedFeaturesNew110<- h2o.deepfeatures(deepNetmodelNew110,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew110=as.data.frame(encodedFeaturesNew110)
dim(encodedFeaturesNew110)
reducedFeaturesWithIDandLabel110=cbind(inputData1[,c(1,133)],encodedFeaturesNew110)
write.csv(reducedFeaturesWithIDandLabel110,'mtordeepfp110.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel110)

library(Metrics)
mod.out110 <- as.data.frame(h2o.predict(deepNetmodelNew110, inputData, type=response))
mod.out110 = as.h2o(mod.out110)

MeanAbsErr110 <-mae(as.matrix(inputData),as.matrix(mod.out110))
MeanAbsPercErr110 <-mape(as.matrix(inputData),as.matrix(mod.out110))
SymMeanPercErr110 <-smape(as.matrix(inputData),as.matrix(mod.out110))
Vartot110<-var(as.matrix(inputData)) # variance total
Varres110<-var(as.matrix(inputData-mod.out110)) # variance residual
Rsq110 <- 1- (sum(Varres110)/sum(Vartot110))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew110)
print(MeanAbsErr110)
print(MeanAbsPercErr110)
print(SymMeanPercErr110)
print(Rsq110)

fpp.anon = h2o.anomaly(deepNetmodelNew110, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 120:120:120)########
h1=c(120,120,120)
set.seed(135)
deepNetmodelNew120 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                       training_frame = inputData,
                                       l1=10^-5,
                                       activation="Tanh",
                                       #activation="Rectifier",
                                       autoencoder=T,
                                       hidden=h1,
                                       epochs=100,
                                       ignore_const_cols=F)    
saveRDS(deepNetmodelNew120,file='deepNetmodelNew120.RDS')
deepNetmodelNew120<- readRDS('deepNetmodelNew120.RDS')
#save(deepNetmodelNew120,file='deepNetmodelNew120.RData')
#deepNetmodelNew120<- load("deepNetmodelNew120.RData")
encodedFeaturesNew120<- h2o.deepfeatures(deepNetmodelNew120,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew120=as.data.frame(encodedFeaturesNew120)
dim(encodedFeaturesNew120)
reducedFeaturesWithIDandLabel120=cbind(inputData1[,c(1,133)],encodedFeaturesNew120)
write.csv(reducedFeaturesWithIDandLabel120,'mtordeepfp120.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel120)

library(Metrics)
mod.out120 <- as.data.frame(h2o.predict(deepNetmodelNew120, inputData, type=response))
mod.out120 = as.h2o(mod.out120)

MeanAbsErr120 <-mae(as.matrix(inputData),as.matrix(mod.out120))
MeanAbsPercErr120 <-mape(as.matrix(inputData),as.matrix(mod.out120))
SymMeanPercErr120 <-smape(as.matrix(inputData),as.matrix(mod.out120))
Vartot120<-var(as.matrix(inputData)) # variance total
Varres120<-var(as.matrix(inputData-mod.out120)) # variance residual
Rsq120 <- 1- (sum(Varres120)/sum(Vartot120))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew120)
print(MeanAbsErr120)
print(MeanAbsPercErr120)
print(SymMeanPercErr120)
print(Rsq120)

fpp.anon = h2o.anomaly(deepNetmodelNew120, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')


##########################AUTOENCODER FOR DIM REDUCTION (Nodes: 130:130:130)########
h1=c(130,130,130)
set.seed(135)
deepNetmodelNew130 <- h2o.deeplearning(x = c(1:ncol(inputData)), 
                                       training_frame = inputData,
                                       l1=10^-5,
                                       activation="Tanh",
                                       #activation="Rectifier",
                                       autoencoder=T,
                                       hidden=h1,
                                       epochs=100,
                                       ignore_const_cols=F)    
saveRDS(deepNetmodelNew130,file='deepNetmodelNew130.RDS')
deepNetmodelNew130<- readRDS('deepNetmodelNew130.RDS')
#save(deepNetmodelNew130,file='deepNetmodelNew130.RData')
#deepNetmodelNew130<- load("deepNetmodelNew130.RData")
encodedFeaturesNew130<- h2o.deepfeatures(deepNetmodelNew130,as.h2o(inputData),layer=2) # Take the features of 2nd layer
encodedFeaturesNew130=as.data.frame(encodedFeaturesNew130)
dim(encodedFeaturesNew130)
reducedFeaturesWithIDandLabel130=cbind(inputData1[,c(1,133)],encodedFeaturesNew130)
write.csv(reducedFeaturesWithIDandLabel130,'mtordeepfp130.csv', row.names = FALSE)
dim(reducedFeaturesWithIDandLabel130)

library(Metrics)
mod.out130 <- as.data.frame(h2o.predict(deepNetmodelNew130, inputData, type=response))
mod.out130 = as.h2o(mod.out130)

MeanAbsErr130 <-mae(as.matrix(inputData),as.matrix(mod.out130))
MeanAbsPercErr130 <-mape(as.matrix(inputData),as.matrix(mod.out130))
SymMeanPercErr130 <-smape(as.matrix(inputData),as.matrix(mod.out130))
Vartot130<-var(as.matrix(inputData)) # variance total
Varres130<-var(as.matrix(inputData-mod.out130)) # variance residual
Rsq130 <- 1- (sum(Varres130)/sum(Vartot130))
#V3<-V1-V2
#R2<-sum(V3)/sum(V1)
print(deepNetmodelNew130)
print(MeanAbsErr130)
print(MeanAbsPercErr130)
print(SymMeanPercErr130)
print(Rsq130)

fpp.anon = h2o.anomaly(deepNetmodelNew130, inputData, per_feature=FALSE)
err <- as.data.frame(fpp.anon)
#err$Reconstruction.MSE
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
plot(err$Reconstruction.MSE, main='Reconstruction Error')
####################################END####################################
# Evaluation matrices of AUTOENCODER for dimension reduction of MACCS-FP (03.10.2019)  
library(data.table)
library(dplyr)
#library(xlsx)
setwd("~/Desktop/ML_mTOR/mTORautoencoder")

Nodes <-c(10,20,30,40,50,60,70,80,90,100,110,120,130)

MSE <- c(0.034656,
         0.02074942,
         0.01347888,
         0.009386425,
         0.006476838,
         0.004730733,
         0.002905174,
         0.002485524,
         0.002093465,
         0.001736421,
         0.00163888,
         0.001598748,
         0.001553838)

# RMSE <-c(0.1861612,
# 0.1440466,
# 0.1160986,
# 0.09688356,
# 0.0804788,
# 0.06878033,
# 0.05389967,
# 0.04985503,
# 0.04575439,
# 0.04167039,
# 0.04048308,
# 0.03998435,
# 0.03941875)

MAE <-c(0.101661,
        0.08185758,
        0.06603592,
        0.05579327,
        0.04643261,
        0.0406253,
        0.03149744,
        0.03084077,
        0.02881583,
        0.02677224,
        0.02652563,
        0.02685393,
        0.02625826)

SMAPE <-c(1.374614,
1.356822,
1.345243,
1.340862,
1.335346,
1.33189,
1.327743,
1.326847,
1.325792,
1.324549,
1.324721,
1.324813,
1.324108)

RSquared <- c(0.9654197,
        0.9845935,
        0.9879986,
        0.9908037,
        0.9930609,
        0.9929657,
        0.9939025,
        0.9932527,
        0.9942349,
        0.9948821,
        0.9955234,
        0.9957589,
        0.9956609)
par(mfrow=c(2,2))

plot(Nodes, MSE, xlab="Number of Nodes", ylab="MSE", type="l", col="blue")
#plot(Nodes, RMSE, xlab="#Nodes",type="l", col="blue")
plot(Nodes, MAE, xlab="Number of Nodes", ylab="MAE", type="l", col="blue")

plot(Nodes, SMAPE, xlab="Number of Nodes", ylab="SMAPE, type="l", col="blue")
plot(Nodes, RSquared, xlab="Number of Nodes",ylab="RSquared",type="l", col="red")

png("autoencoder.png")
par(mfrow=c(2,2))
plot(Nodes, MSE, xlab="NumberofNodes", type="l",  col="blue")
#plot(Nodes, RMSE, xlab="#Nodes",type="l", col="blue")
plot(Nodes, MAE, xlab="NumberofNodes", type="l", col="blue")
plot(Nodes, SMAPE, xlab="Number of Nodes", type= "l", col="blue")
plot(Nodes, RSquared, xlab="Number of Nodes", type="l",col="red")
dev.off ()

png("MSE_autoencoder.png")
plot(Nodes, MSE, xlab="#Nodes",type="l", col="blue")
dev.off ()

png("RMSE_autoencoder.png")
plot(Nodes,RMSE, xlab="Number of Nodes",type="l", col="blue")
dev.off ()

png("MAE_autoencoder.png")
plot(Nodes, MAE, xlab="Number of Nodes",type="l", col="blue")
dev.off ()

png("SMAPE_autoencoder.png")
plot(Nodes, SMAPE, xlab="Number of Nodes",type="l", col="blue")
dev.off ()

png("R2_autoencoder.png")
plot(Nodes, RSquared, xlab="Number of Nodes",type="l", col="blue")
dev.off ()

