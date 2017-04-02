**Study Background**
====================

#### Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

**Executive Summary**
=====================

#### We clean the data set and fit three models. The first model was built by Decision Tree, but the accuracy was only 52.37%. So we built the second model by Random Forest with highest accuracy 99.66%, which implied that the out of sample error rate was 0.34%. The third model was built by Gradient Boosting Machine Model, which accuracy was 98.73%, which implied the out of sample error rate was 1.27%. We then select compare the last two models for the testing data sets, and get identical predictions.

**Data Processing**
===================

``` r
##Load the packages we need.
library(data.table)
library(caret)
library(ggplot2)

##Download the data if they don't exist in working directory.
if(!file.exists("./training.csv")){
url.training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url.training, destfile = "./training.csv")
}
if(!file.exists("./testing.csv")){
url.testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url.testing, destfile = "./testing.csv")
}

##Read the data into our workspace.
training <- fread("./training.csv", na.strings=c("NA","", "#DIV/0!"))
testing <- fread("./testing.csv", na.strings=c("NA","", "#DIV/0!"))
```

-   **We replace "" and "\#DIV/0!" as NA when we notice they are missing values, too.**
-   **There are 19,622 observations and 160 variables in the training data set**
-   **There are 20 observations and 160 variables in the testing data set.**

**Data Cleaning**
=================

``` r
training <- data.table(training)
testing <- data.table(testing)

##We drop the columns that only contains zero.
nacols <- colnames(testing)[colSums(!is.na(testing)) == 0]
training[,(nacols):= NULL]
testing[,(nacols):= NULL]

##We drop the index ,username and timestamp columns that we won't use in our analysis.
unwantedcols <- c("V1", "user_name", "raw_timestamp_part_1",
                  "raw_timestamp_part_2","cvtd_timestamp", "problem_id")
training[,(unwantedcols):= NULL]
testing[,(unwantedcols):= NULL]

##We set the variables into correct classes.
training$new_window <- as.factor(training$new_window)
testing$new_window <- as.factor(testing$new_window)
training$classe <- as.factor(training$classe)

#Set the seed for reproducible computation.
set.seed(12345)

##We split training data set into smaller training data set and validation data set.
intrain <- createDataPartition(training$classe, p = 0.7)
trainingdp <- training[c(unlist(intrain))]
validationdp <- training[-c(unlist(intrain))]
```

**Model Building**
==================

### **Decision Tree**

``` r
##We set 10 folds for cross validation
train_control <- trainControl(method="cv", number=10)
##We fit our first model with decision tree.
modelFit1 <- train(classe ~., method="rpart", data=trainingdp,
trControl = train_control)
```

    ## Loading required package: rpart

``` r
result1 <- confusionMatrix(validationdp$classe, predict(modelFit1, newdata=validationdp))
result1
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1494   21  128    0   31
    ##          B  470  380  289    0    0
    ##          C  467   29  530    0    0
    ##          D  416  184  324    0   40
    ##          E   95   90  219    0  678
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5237          
    ##                  95% CI : (0.5109, 0.5365)
    ##     No Information Rate : 0.4999          
    ##     P-Value [Acc > NIR] : 0.0001376       
    ##                                           
    ##                   Kappa : 0.3791          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.5078  0.53977  0.35570       NA   0.9052
    ## Specificity            0.9388  0.85350  0.88714   0.8362   0.9213
    ## Pos Pred Value         0.8925  0.33363  0.51657       NA   0.6266
    ## Neg Pred Value         0.6561  0.93173  0.80243       NA   0.9852
    ## Prevalence             0.4999  0.11963  0.25319   0.0000   0.1273
    ## Detection Rate         0.2539  0.06457  0.09006   0.0000   0.1152
    ## Detection Prevalence   0.2845  0.19354  0.17434   0.1638   0.1839
    ## Balanced Accuracy      0.7233  0.69664  0.62142       NA   0.9133

##### **Somehow the accuracy is only 52.37%, which is not acceptable. So we just stop here and try other models.**

### **Random Forest**

``` r
modelFit2 <- train(classe ~., method="rf", data=trainingdp)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
m2final <- modelFit2$finalModel
m2final
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 28
    ## 
    ##         OOB estimate of  error rate: 0.21%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    5 2651    1    1    0 0.0026335591
    ## C    0    5 2390    1    0 0.0025041736
    ## D    0    0    8 2244    0 0.0035523979
    ## E    0    0    0    6 2519 0.0023762376

``` r
result2 <- confusionMatrix(validationdp$classe, predict(modelFit2, newdata=validationdp))
result2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    5 1133    1    0    0
    ##          C    0    3 1023    0    0
    ##          D    0    0    7  957    0
    ##          E    0    0    0    4 1078
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9966          
    ##                  95% CI : (0.9948, 0.9979)
    ##     No Information Rate : 0.2853          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9957          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9970   0.9974   0.9922   0.9958   1.0000
    ## Specificity            1.0000   0.9987   0.9994   0.9986   0.9992
    ## Pos Pred Value         1.0000   0.9947   0.9971   0.9927   0.9963
    ## Neg Pred Value         0.9988   0.9994   0.9984   0.9992   1.0000
    ## Prevalence             0.2853   0.1930   0.1752   0.1633   0.1832
    ## Detection Rate         0.2845   0.1925   0.1738   0.1626   0.1832
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9985   0.9980   0.9958   0.9972   0.9996

### **Gradient Boosting Machine**

``` r
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

modelFit3 <- train(classe ~ ., data=trainingdp, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)

result3 <- confusionMatrix(validationdp$classe, predict(modelFit3, newdata=validationdp))
result3
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668    4    0    1    1
    ##          B   11 1119    7    2    0
    ##          C    0    9 1015    2    0
    ##          D    0    6   13  944    1
    ##          E    0    1    3   14 1064
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9873        
    ##                  95% CI : (0.9841, 0.99)
    ##     No Information Rate : 0.2853        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9839        
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9934   0.9824   0.9778   0.9803   0.9981
    ## Specificity            0.9986   0.9958   0.9977   0.9959   0.9963
    ## Pos Pred Value         0.9964   0.9824   0.9893   0.9793   0.9834
    ## Neg Pred Value         0.9974   0.9958   0.9953   0.9961   0.9996
    ## Prevalence             0.2853   0.1935   0.1764   0.1636   0.1811
    ## Detection Rate         0.2834   0.1901   0.1725   0.1604   0.1808
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9960   0.9891   0.9878   0.9881   0.9972

### **Predict the "classe" for testing data.**

``` r
##Prediction of Random Forest
testing2 <- predict(modelFit2, newdata = testing)
testing2
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

``` r
##Prediction of Generalized Boosted Model
testing3 <- predict(modelFit3, newdata = testing)
testing3
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

``` r
identical(testing2, testing3)
```

    ## [1] TRUE

``` r
table(testing2, testing3)
```

    ##         testing3
    ## testing2 A B C D E
    ##        A 7 0 0 0 0
    ##        B 0 8 0 0 0
    ##        C 0 0 1 0 0
    ##        D 0 0 0 1 0
    ##        E 0 0 0 0 3

``` r
##We can see the predictions are identical for both models.
```
