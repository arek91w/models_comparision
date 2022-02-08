library(caret)
library(e1071)
library(rstudioapi)
library(fastDummies)
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("funkcje.R")

set.seed(666)
seed <- 666

parTune_knn = expand.grid(k = seq(3, 15, by = 3))
parTune_tree = expand.grid(type=c("Gini", "Entropy"), depth=3:6, minobs=3)
parTune_tree_rm <- expand.grid(type=c("gini", "information"), depth=3:6, minobs=3)
parTune_tree_reg <- expand.grid(type=c("SS"), depth=3:7, minobs=3)
parTune_nn<- expand.grid(iter=c(20, 50, 100), lr=c(0.1, 0.09, 0.08, 0.07))

# regresja


d_reg <- read.csv("regresja/forestfires.csv", sep=",", dec = ".")
head(d_reg)
d_reg <- d_reg[sample(nrow(d_reg), 50), ]
head(d_reg)
head(d_reg[,2])

d_reg <- cbind("Y"=as.numeric(d_reg[,2]), d_reg[,c(1,5:13)])
head(d_reg)
data_reg_norm <- d_reg
data_reg_norm[, 2:11]  <- scale(data_reg_norm[, 2:11])
head(data_reg_norm)
cv_res_reg_knn <- CrossValidation(d_reg, 5, parTune_knn, seed, model='KNN')
bestKNN_reg <- SelectBestModelCV(cv_res_reg_knn, "KNN")

cv_res_reg_tree <- CrossValidation(d_reg, 5, parTune_tree_reg, seed, model='TREE')
bestTree_reg <- SelectBestModelCV(cv_res_reg_tree, "TREE")

cv_res_reg_nn <- CrossValidation(data_reg_norm, 5, parTune_nn, seed, model='NN')
bestNN_reg <- SelectBestModelCV(cv_res_reg_nn, "NN")

models_reg_im <- rbind(rbind(bestKNN_reg, bestTree_reg), bestNN_reg)

###########################regresja gotowe algorytmy############################
cv_res_reg_knn_rm <- CrossValidationReadyModels(d_reg, 5, parTune_knn, seed, model='KNN')
bestKNN_reg_rm  <- SelectBestModelCV(cv_res_reg_knn_rm, "KNN_rm")

cv_res_reg_tree_rm <- CrossValidationReadyModels(d_reg, 5, parTune_tree_reg, seed, model='TREE')
bestTree_reg_rm  <- SelectBestModelCV(cv_res_reg_tree_rm, "TREE_rm")

cv_res_reg_nn_rm <- CrossValidationReadyModels(data_reg_norm, 5, parTune_nn, seed, model='NN')
bestNN_reg_rm  <- SelectBestModelCV(cv_res_reg_nn_rm, "NN_rm")

models_reg_rm <- rbind(rbind(bestKNN_reg_rm, bestTree_reg_rm), bestNN_reg_rm)

models_reg <- rbind(models_reg_im, models_reg_rm)
SelectBestModel(models_reg)


#####################wykresy implementowane algorytmy###########################
ggplot(cv_res_reg_knn, aes(k)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'k' na MAPE w zaimplementowanym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_reg_tree, aes(depth)) + 
  geom_point(aes(y = MAPEw, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = MAPEt, shape = "Trening", colour = type)) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'depth' oraz 'type' na MAPE w zaimplementowanym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_reg_nn, aes(lr)) + 
  geom_point(aes(y = MAPEw, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = iter, shape = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'lr' oraz 'iter' na MAPE w zaimplementowanym modelu NN") +
  labs(colour="Iteracje", shape="Dane")

#########################wykresy gotowe algorytmy###############################
ggplot(cv_res_reg_knn_rm, aes(k)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'k' na MAPE w gotowym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_reg_tree_rm, aes(depth)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Train" )) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'depth' na MAPE w gotowym modelu TREE") +
  labs(colour="Dane")

ggplot(cv_res_reg_nn_rm, aes(lr)) + 
  geom_point(aes(y = MAPEw, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = MAPEt,colour = iter, shape = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp造w 'lr' oraz 'iter' na MAPE w gotowym modelu NN") +
  labs(colour="Iteracje", shape="Dane")




# klasyfikacja binarna

d_bin <- read.csv("binarna/caesarian.csv", sep=",", dec = ".")
head(d_bin)

d_bin$Delivery.No <- as.numeric(as.factor(d_bin$Delivery.No))
d_bin$Blood.of.Pressure <- as.numeric(as.factor(d_bin$Blood.of.Pressure))
d_bin$Heart.Problem <- as.numeric(as.factor(d_bin$Heart.Problem ))

d_bin$Caesarian <- as.numeric(as.factor(d_bin$Caesarian ))

d_bin <- cbind("Caesarian"=d_bin[,6], d_bin[,1:5])
head(d_bin)
data_bin_norm <- d_bin
head(data_bin_norm[, 2:6])
data_bin_norm[, 2:6]  <- scale(data_bin_norm[, 2:6])

#####################binarna zaimplementowane algorytmy#########################
cv_res_bin_knn <- CrossValidation(d_bin, 5, parTune_knn, seed, model='KNN')
bestKNN  <- SelectBestModelCV(cv_res_bin_knn, "KNN")

cv_res_bin_tree <- CrossValidation(d_bin, 5, parTune_tree, seed, model='TREE')
bestTree <- SelectBestModelCV(cv_res_bin_tree, "TREE")

cv_res_bin_nn <- CrossValidation(data_bin_norm, 5, parTune_nn, seed, model='NN')
bestNN <- SelectBestModelCV(cv_res_bin_nn, "NN")

models_bin_im <- rbind(rbind(bestKNN, bestTree), bestNN)

##########################binarna gotowe algorytmy##############################
cv_res_bin_knn_rm <- CrossValidationReadyModels(d_bin, 5, parTune_knn, seed, model='KNN')
bestKNN_rm  <- SelectBestModelCV(cv_res_bin_knn_rm, "KNN_rm")

cv_res_bin_tree_rm <- CrossValidationReadyModels(d_bin, 5, parTune_tree_rm, seed, model='TREE')
bestTree_rm  <- SelectBestModelCV(cv_res_bin_tree_rm, "TREE_rm")

cv_res_bin_nn_rm <- CrossValidationReadyModels(data_bin_norm, 5, parTune_nn, seed, model='NN')
bestNN_rm  <- SelectBestModelCV(cv_res_bin_nn_rm, "NN_rm")

models_bin_rm <- rbind(rbind(bestKNN_rm, bestTree_rm), bestNN_rm)

models_bin <- rbind(models_bin_im, models_bin_rm)
SelectBestModel(models_bin)

#####################wykresy implementowane algorytmy###########################
ggplot(cv_res_bin_knn, aes(k)) + 
  geom_point(aes(y = AUCW, colour = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'k' na AUC w zaimplementowanym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_bin_tree, aes(depth)) + 
  geom_point(aes(y = AUCW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = AUCT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'depth' oraz 'type' na AUC w zaimplementowanym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_bin_nn, aes(lr)) + 
  geom_point(aes(y = AUCW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = iter, shape = "Treing")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'lr' oraz 'iter' na AUC w zaimplementowanym modelu NN") +
  labs(colour="Iteracje", shape="Dane")

#########################wykresy gotowe algorytmy###############################
ggplot(cv_res_bin_knn_rm, aes(k)) + 
  geom_point(aes(y = AUCW, colour = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'k' na AUC w gotowym modelu KNN") +
  labs(colour="Legenda")

ggplot(cv_res_bin_tree_rm, aes(depth)) + 
  geom_point(aes(y = AUCW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = AUCT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'depth' oraz 'type' na AUC w gotowym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_bin_nn_rm, aes(lr)) + 
  geom_point(aes(y = AUCW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = iter, shape = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp造w 'lr' oraz 'iter' na AUC w gotowym modelu NN") +
  labs(colour="Iteracje", shape="Dane")