# Machine learning course project

Avishek Sarkar
05 December 2017

## Executive Summary

From the dataset, it is first observed from basic exploratory analysis that only 1/3 of the data is informative. After screening out the uninformative data, 4 different machines learning models: random forest, boosting, linear discriminant, and classification trees was tried on subsets of the training data. After a few trials, the random forest model was chosen to generate the answers for the quiz, which achieved 19 correct answers out of 20 questions.

## Data Loading

The first step is to load the training and testing data to R. There are 160 variables on 19622 observations


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```



```r
setwd("D:/R_Data")
training = read.csv("D:/R_Data/pml-training.csv")
testing = read.csv("D:/R_Data/pml-testing.csv")
```


## Cleaning the data

```r
training1 = training[, -c(1:5, 12:36, 50:59, 69:83, 87:101, 103:112, 125:139, 
    141:150)]
testing1 = testing[, -c(1:5, 12:36, 50:59, 69:83, 87:101, 103:112, 125:139, 
    141:150)]
```


On closer inspection, the original dataset has 160 variables (including the response variable). However, upon scanning through the summary of the dataset, the first 5 variables are unlikely to be explanatory since they are data identifiers for individual and time, and an additional 100 variables has 98% of their observations in NAs so they are highly unlikely to contain much value as well so they are also dropped.


## Model selection

Due to the size of the dataset, it is diffcult to employ algorithms like random forest on all of the data. So the training dataset is further divided into 20 random subsets. The first subset is first used to train 4 types of models: random forest, boosting, classification trees, and linear discriminant.

The accuracy performance of classification trees (54%) and linear discriminant (67%) is far below that of random forest (88.7%) and boosting (87%). As such, only the latter two are tried on a second subset of the training dataset.


```r
set.seed(2233)
folds <- createFolds(y = training1$classe, k = 20, list = TRUE, returnTrain = FALSE)
training1_1 <- training1[folds$Fold01, ]
modFit1_1_rf <- train(classe ~ ., data = training1_1, method = "rf", prox = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```



```r
modFit1_1_gbm <- train(classe ~ ., data = training1_1, method = "gbm", verbose = FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object(s) are masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```



```r
modFit1_1_rpart <- train(classe ~ ., data = training1_1, method = "rpart")
```

```
## Loading required package: rpart
```

```r
modFit1_1_lda <- train(classe ~ ., data = training1_1, method = "lda")
```

```
## Loading required package: MASS
```

```r

training1_2 <- training1[folds$Fold02, ]
modFit1_2_rf <- train(classe ~ ., data = training1_2, method = "rf", prox = TRUE)
modFit1_2_gbm <- train(classe ~ ., data = training1_2, method = "gbm", verbose = FALSE)
```


On the second subset, accuracy of random forest is 88% and that of boosting is 87.2 %. Both still have very high accuracy. We will then deploy the model to test out-of-sample error with a third subset of data


```r
training1_3 <- training1[folds$Fold03, ]
confusionMatrix(training1_3$classe, predict(modFit1_1_rf, training1_3))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 268   5   2   4   0
##          B   8 169  10   1   1
##          C   0   4 165   2   0
##          D   0   2  16 139   3
##          E   0   1   4   2 173
## 
## Overall Statistics
##                                         
##                Accuracy : 0.934         
##                  95% CI : (0.916, 0.948)
##     No Information Rate : 0.282         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.916         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.971    0.934    0.838    0.939    0.977
## Specificity             0.984    0.975    0.992    0.975    0.991
## Pos Pred Value          0.961    0.894    0.965    0.869    0.961
## Neg Pred Value          0.989    0.985    0.960    0.989    0.995
## Prevalence              0.282    0.185    0.201    0.151    0.181
## Detection Rate          0.274    0.173    0.169    0.142    0.177
## Detection Prevalence    0.285    0.193    0.175    0.163    0.184
## Balanced Accuracy       0.978    0.954    0.915    0.957    0.984
```



```r
confusionMatrix(training1_3$classe, predict(modFit1_1_gbm, training1_3))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 270   4   4   1   0
##          B  10 163  12   3   1
##          C   0   7 161   3   0
##          D   0   2  16 139   3
##          E   0   4   3   3 170
## 
## Overall Statistics
##                                         
##                Accuracy : 0.922         
##                  95% CI : (0.904, 0.938)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.902         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.964    0.906    0.821    0.933    0.977
## Specificity             0.987    0.967    0.987    0.975    0.988
## Pos Pred Value          0.968    0.862    0.942    0.869    0.944
## Neg Pred Value          0.986    0.978    0.957    0.988    0.995
## Prevalence              0.286    0.184    0.200    0.152    0.178
## Detection Rate          0.276    0.166    0.164    0.142    0.174
## Detection Prevalence    0.285    0.193    0.175    0.163    0.184
## Balanced Accuracy       0.976    0.937    0.904    0.954    0.982
```



```r
training1_4 <- training1[folds$Fold04, ]
confusionMatrix(training1_4$classe, predict(modFit1_1_rf, training1_4))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 275   0   2   1   1
##          B   6 163  13   3   5
##          C   0   9 162   0   0
##          D   1   0  15 143   2
##          E   1   3   7   4 165
## 
## Overall Statistics
##                                         
##                Accuracy : 0.926         
##                  95% CI : (0.907, 0.941)
##     No Information Rate : 0.288         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.906         
##  Mcnemar's Test P-Value : 0.00013       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.972    0.931    0.814    0.947    0.954
## Specificity             0.994    0.967    0.988    0.978    0.981
## Pos Pred Value          0.986    0.858    0.947    0.888    0.917
## Neg Pred Value          0.989    0.985    0.954    0.990    0.990
## Prevalence              0.288    0.178    0.203    0.154    0.176
## Detection Rate          0.280    0.166    0.165    0.146    0.168
## Detection Prevalence    0.284    0.194    0.174    0.164    0.183
## Balanced Accuracy       0.983    0.949    0.901    0.963    0.968
```



```r
confusionMatrix(training1_4$classe, predict(modFit1_1_gbm, training1_4))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 270   4   2   1   2
##          B   5 168  12   1   4
##          C   0   8 161   2   0
##          D   1   4  14 139   3
##          E   0   8   9   5 158
## 
## Overall Statistics
##                                        
##                Accuracy : 0.913        
##                  95% CI : (0.894, 0.93)
##     No Information Rate : 0.281        
##     P-Value [Acc > NIR] : < 2e-16      
##                                        
##                   Kappa : 0.89         
##  Mcnemar's Test P-Value : 0.00307      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.978    0.875    0.813    0.939    0.946
## Specificity             0.987    0.972    0.987    0.974    0.973
## Pos Pred Value          0.968    0.884    0.942    0.863    0.878
## Neg Pred Value          0.991    0.970    0.954    0.989    0.989
## Prevalence              0.281    0.196    0.202    0.151    0.170
## Detection Rate          0.275    0.171    0.164    0.142    0.161
## Detection Prevalence    0.284    0.194    0.174    0.164    0.183
## Balanced Accuracy       0.983    0.924    0.900    0.956    0.960
```


From the out-of-sample testing for subset 3 and 4, random forest model still outperforms boosting, with an accuracy rate of over 94%. As such, the random forest model 1 is chosen.


```r
predict(modFit1_1_rf, testing1)
```

```
##  [1] B A A A A E D B A A C C B A E E A B B B
## Levels: A B C D E
```




