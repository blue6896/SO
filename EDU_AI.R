#https://archive.ics.uci.edu/dataset/445/absenteeism+at+work

set.seed(100)
library(ROSE)
library(randomForest)
library(caret)

work.df_with_binary <- read.table("Absenteeism_at_work.csv", sep=";", header=T)
dim(work.df)
work.df= work.df_with_binary [, -c(12,15,16)]

work.df[which(work.df[,2]!=26),2]=0
work.df[which(work.df[,2]==26),2]=1
table(work.df$Reason.for.absence)
work.df= work.df[,c(-1)]

# Disciplinary.failure and unjustified absenteeism 
M1 <- matrix(table(work.df_with_binary[,12], work.df[,1]), nrow = 2)
oddsratio(M1)

# Social drinker and unjustified absenteeism 
M2 <- matrix(table(work.df_with_binary[,15], work.df[,1]), nrow = 2)
oddsratio(M2)

# Social smoker and unjustified absenteeism 
M3 <- matrix(table(work.df_with_binary[,16], work.df[,1]), nrow = 2)
oddsratio(M3)


# Random Forest

origacc=c(); overacc=c(); underacc=c()

for (i in 1:100){ 
print(i)
index= sample(dim(work.df)[1], dim(work.df)[1]*0.8)
training= work.df[index,]
testing= work.df[-index,]

rftrain <- randomForest(Reason.for.absence~., data =training)
orig_rf_results=confusionMatrix(as.factor(round(predict(rftrain, testing))), as.factor(testing$Reason.for.absence), positive = '1')
print( (orig_rf_results$table[2,2]+orig_rf_results$table[1,1])   / sum(orig_rf_results$table )  )
origacc=c(origacc,(orig_rf_results$table[2,2]+orig_rf_results$table[1,1])   / sum(orig_rf_results$table ) )

over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
rftrain_over <- randomForest(Reason.for.absence~., data =over)
over_rf_results=confusionMatrix(as.factor(round(predict(rftrain_over, testing))), as.factor(testing$Reason.for.absence), positive = '1')
print( (over_rf_results$table[2,2]+over_rf_results$table[1,1])   / sum(over_rf_results$table )  )
overacc= c(overacc,  (over_rf_results$table[2,2]+over_rf_results$table[1,1])   / sum(over_rf_results$table ))

under <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
rftrain_under <- randomForest(Reason.for.absence~., data =under)
under_rf_results=confusionMatrix(as.factor(round(predict(rftrain_under, testing))), as.factor(testing$Reason.for.absence), positive = '1')
print( (under_rf_results$table[2,2]+under_rf_results$table[1,1])   / sum(under_rf_results$table )  )
underacc=c(underacc,(under_rf_results$table[2,2]+under_rf_results$table[1,1])   / sum(under_rf_results$table ) )
}

mean(origacc); mean(overacc); mean(underacc)

library(ggfortify)
library(ggplot2)

# PCA
work.df_pca=work.df
work.df_pca[which(work.df_pca[,1]==0),1] = 'Presenteeism or Justified Absenteeism'
work.df_pca[which(work.df_pca[,1]==1),1] = 'Unjustified Absenteeism'
forPCA_work= work.df_pca[,-1]
pca_res <- prcomp(forPCA_work, scale. = T)
autoplot(pca_res, data=work.df_pca, colour = 'Reason.for.absence')

# glasso and community detection
set.seed(1000)
library(glasso)
library(igraph)
tableX= as.data.frame(forPCA_work)
s<- var(forPCA_work)
ai<-glasso(s, rho=0.0001)
aa<-glasso(s,rho=0.0001, w.init=ai$w, wi.init=ai$wi)

g=graph_from_adjacency_matrix(aa$wi, mode='undirected', diag=F)

V(g)$name=colnames(tableX)
plot(simplify(g),vertex.label.cex = 0.7, vertex.size=10)

# greedy method (hiearchical, fast method)
c1 = cluster_fast_greedy(simplify(g))


# Networks For drinker 
forPCA_work_drinker= as.data.frame(forPCA_work[which(work.df_with_binary[,15]==1),])
s_drinker<- var(forPCA_work_drinker)
ai_d<-glasso(s_drinker, rho=0.0001)
aa_d<-glasso(s_drinker,rho=0.0001, w.init=ai_d$w, wi.init=ai_d$wi)

g_d=graph_from_adjacency_matrix(aa_d$wi, mode='undirected', diag=F)

V(g_d)$name=colnames(tableX)
plot(simplify(g_d),vertex.label.cex = 0.7, vertex.size=10)

# Networks for non-drinker 

forPCA_work_Ndrinker= as.data.frame(forPCA_work[which(work.df_with_binary[,15]==0),])
s_Ndrinker<- var(forPCA_work_Ndrinker)
ai_nd<-glasso(s_Ndrinker, rho=0.0001)
aa_nd<-glasso(s_Ndrinker,rho=0.0001, w.init=ai_nd$w, wi.init=ai_nd$wi)

g_nd=graph_from_adjacency_matrix(aa_nd$wi, mode='undirected', diag=F)

V(g_nd)$name=colnames(tableX)
plot(simplify(g_nd),vertex.label.cex = 0.7, vertex.size=10)


# Lasso, Elastic-Net, Ridge
library(glmnet)

X=tableX

## Lasso 
set.seed(100)
coefLA = rep(0,17)
errLA = 0 
fLA=0

for(i in 1:100){
#  over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
  setindex= sample(1:dim(X)[1], replace = F)
  training= work.df[setindex[1:(dim(X)[1]*0.8)],]
  testing = work.df[-setindex[1:(dim(X)[1]*0.8)],]
  overdata <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
  
  trainX= as.matrix( overdata[,2:dim(work.df)[2] ] )
  testX= as.matrix( testing[,2:dim(work.df)[2]]  )
  #trainX=as.matrix(tempX[c(1:(dim(X)[1]*0.8) ),])
  y= work.df[,1] 
  trainy= as.matrix( overdata[,1] )
  testy= as.matrix(testing[,1] )
#  trainy=as.matrix(y[1:(dim(X)[1]*0.8)])
  
  #p.fac <- rep(1, dim(X)[2])
  #p.fac[c(dim(X)[2]-1, dim(X)[2])] 
  pfit <- cv.glmnet(trainX, trainy, family='binomial',
                    type.measure = 'class', alpha=1)
  #plot(pfit, label = TRUE)
  
  pfit$lambda.min
  print(i)
  print(which(pfit$glmnet.fit$lambda==pfit$lambda.min))
  
  predy=predict(pfit, newx = as.matrix(testX), s = "lambda.min")
  predy=(predy>0)+0
  obsvy=(testy=='1')
  
  print(as.numeric(coef(pfit)))
  coefLA = cbind(coefLA,as.numeric(coef(pfit)) )
  errLA = c(errLA, sum(predy==obsvy)/(dim(X)[1]*0.2))
  
  tempF = predy+obsvy 
  fLA= c( fLA , 2*sum(tempF==2)/(2*sum(tempF==2)+ sum(tempF==1)))
  
}

colnames(tableX)[which(rowSums(coefLA) >0) -1 ]
colnames(tableX)[which(rowSums(coefLA) <0) -1]
coefLA_Work= rowSums(coefLA)[-1]
POS=cbind(colnames(tableX)[which(coefLA_Work>0) ], round(coefLA_Work[coefLA_Work>0],3) )
write.csv(POS, "POSWORK.csv")

NEG=cbind(colnames(tableX)[which(coefLA_Work<0) ], round(coefLA_Work[coefLA_Work<0],3) )
write.csv(NEG, "NEGWORK.csv")



## ENET75 
set.seed(100)
coef75 = rep(0,17)
err75 = 0 
f75=0

for(i in 1:100){
  #  over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
  setindex= sample(1:dim(X)[1], replace = F)
  training= work.df[setindex[1:(dim(X)[1]*0.8)],]
  testing = work.df[-setindex[1:(dim(X)[1]*0.8)],]
  overdata <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
  
  trainX= as.matrix( overdata[,2:dim(work.df)[2] ] )
  testX= as.matrix( testing[,2:dim(work.df)[2]]  )
  #trainX=as.matrix(tempX[c(1:(dim(X)[1]*0.8) ),])
  y= work.df[,1] 
  trainy= as.matrix( overdata[,1] )
  testy= as.matrix(testing[,1] )
  #  trainy=as.matrix(y[1:(dim(X)[1]*0.8)])
  
  #p.fac <- rep(1, dim(X)[2])
  #p.fac[c(dim(X)[2]-1, dim(X)[2])] 
  pfit <- cv.glmnet(trainX, trainy, family='binomial',
                    type.measure = 'class', alpha=0.75)
  #plot(pfit, label = TRUE)
  
  pfit$lambda.min
  print(i)
  print(which(pfit$glmnet.fit$lambda==pfit$lambda.min))
  
  predy=predict(pfit, newx = as.matrix(testX), s = "lambda.min")
  predy=(predy>0)+0
  obsvy=(testy=='1')
  
  print(as.numeric(coef(pfit)))
  coef75 = cbind(coef75,as.numeric(coef(pfit)) )
  err75 = c(err75, sum(predy==obsvy)/(dim(X)[1]*0.2))
  
  tempF = predy+obsvy 
  f75= c( f75 , 2*sum(tempF==2)/(2*sum(tempF==2)+ sum(tempF==1)))
  
}


## ENET 50  
set.seed(100)
coef50 = rep(0,17)
err50 = 0 
f50=0

for(i in 1:100){
  #  over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
  setindex= sample(1:dim(X)[1], replace = F)
  training= work.df[setindex[1:(dim(X)[1]*0.8)],]
  testing = work.df[-setindex[1:(dim(X)[1]*0.8)],]
  overdata <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
  
  trainX= as.matrix( overdata[,2:dim(work.df)[2] ] )
  testX= as.matrix( testing[,2:dim(work.df)[2]]  )
  #trainX=as.matrix(tempX[c(1:(dim(X)[1]*0.8) ),])
  y= work.df[,1] 
  trainy= as.matrix( overdata[,1] )
  testy= as.matrix(testing[,1] )
  #  trainy=as.matrix(y[1:(dim(X)[1]*0.8)])
  
  #p.fac <- rep(1, dim(X)[2])
  #p.fac[c(dim(X)[2]-1, dim(X)[2])] 
  pfit <- cv.glmnet(trainX, trainy, family='binomial',
                    type.measure = 'class', alpha=0.50)
  #plot(pfit, label = TRUE)
  
  pfit$lambda.min
  print(i)
  print(which(pfit$glmnet.fit$lambda==pfit$lambda.min))
  
  predy=predict(pfit, newx = as.matrix(testX), s = "lambda.min")
  predy=(predy>0)+0
  obsvy=(testy=='1')
  
  print(as.numeric(coef(pfit)))
  coef50 = cbind(coef50,as.numeric(coef(pfit)) )
  err50 = c(err50, sum(predy==obsvy)/(dim(X)[1]*0.2))
  
  tempF = predy+obsvy 
  f50= c( f50 , 2*sum(tempF==2)/(2*sum(tempF==2)+ sum(tempF==1)))
  
}



## ENET 25  
set.seed(100)
coef25 = rep(0,17)
err25 = 0 
f25=0

for(i in 1:100){
  #  over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
  setindex= sample(1:dim(X)[1], replace = F)
  training= work.df[setindex[1:(dim(X)[1]*0.8)],]
  testing = work.df[-setindex[1:(dim(X)[1]*0.8)],]
  overdata <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
  
  trainX= as.matrix( overdata[,2:dim(work.df)[2] ] )
  testX= as.matrix( testing[,2:dim(work.df)[2]]  )
  #trainX=as.matrix(tempX[c(1:(dim(X)[1]*0.8) ),])
  y= work.df[,1] 
  trainy= as.matrix( overdata[,1] )
  testy= as.matrix(testing[,1] )
  #  trainy=as.matrix(y[1:(dim(X)[1]*0.8)])
  
  #p.fac <- rep(1, dim(X)[2])
  #p.fac[c(dim(X)[2]-1, dim(X)[2])] 
  pfit <- cv.glmnet(trainX, trainy, family='binomial',
                    type.measure = 'class', alpha=0.25)
  #plot(pfit, label = TRUE)
  
  pfit$lambda.min
  print(i)
  print(which(pfit$glmnet.fit$lambda==pfit$lambda.min))
  
  predy=predict(pfit, newx = as.matrix(testX), s = "lambda.min")
  predy=(predy>0)+0
  obsvy=(testy=='1')
  
  print(as.numeric(coef(pfit)))
  coef25 = cbind(coef25,as.numeric(coef(pfit)) )
  err25 = c(err25, sum(predy==obsvy)/(dim(X)[1]*0.2))
  
  tempF = predy+obsvy 
  f25= c( f25 , 2*sum(tempF==2)/(2*sum(tempF==2)+ sum(tempF==1)))
  
}


## Ridge 
set.seed(100)
coefRI = rep(0,17)
errRI = 0 
fRI=0

for(i in 1:100){
  #  over <- ovun.sample(Reason.for.absence~., data = training, method = "over", N = 1000)$data
  setindex= sample(1:dim(X)[1], replace = F)
  training= work.df[setindex[1:(dim(X)[1]*0.8)],]
  testing = work.df[-setindex[1:(dim(X)[1]*0.8)],]
  overdata <- ovun.sample(Reason.for.absence~., data = training, method = "under", N = 100)$data
  
  trainX= as.matrix( overdata[,2:dim(work.df)[2] ] )
  testX= as.matrix( testing[,2:dim(work.df)[2]]  )
  #trainX=as.matrix(tempX[c(1:(dim(X)[1]*0.8) ),])
  y= work.df[,1] 
  trainy= as.matrix( overdata[,1] )
  testy= as.matrix(testing[,1] )
  #  trainy=as.matrix(y[1:(dim(X)[1]*0.8)])
  
  #p.fac <- rep(1, dim(X)[2])
  #p.fac[c(dim(X)[2]-1, dim(X)[2])] 
  pfit <- cv.glmnet(trainX, trainy, family='binomial',
                    type.measure = 'class', alpha=0)
  #plot(pfit, label = TRUE)
  
  pfit$lambda.min
  print(i)
  print(which(pfit$glmnet.fit$lambda==pfit$lambda.min))
  
  predy=predict(pfit, newx = as.matrix(testX), s = "lambda.min")
  predy=(predy>0)+0
  obsvy=(testy=='1')
  
  print(as.numeric(coef(pfit)))
  coefRI = cbind(coefRI,as.numeric(coef(pfit)) )
  errRI = c(errRI, sum(predy==obsvy)/(dim(X)[1]*0.2))
  
  tempF = predy+obsvy 
  fRI = c( fRI , 2*sum(tempF==2)/(2*sum(tempF==2)+ sum(tempF==1)))
  
}

colnames(tableX)[which(rowSums(coefRI) >0) -1 ]
colnames(tableX)[which(rowSums(coefRI) <0) -1]
coefRI_Work= rowSums(coefRI)[-1]
POS=cbind(colnames(tableX)[which(coefRI_Work>0) ], round(coefRI_Work[coefRI_Work>0],3) )
write.csv(POS, "POSWORK_RI.csv")

NEG=cbind(colnames(tableX)[which(coefRI_Work<0) ], round(coefRI_Work[coefRI_Work<0],3) )
write.csv(NEG, "NEGWORK_RI.csv")


