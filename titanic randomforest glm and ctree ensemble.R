#setwd("C:/Matthew/Data/titanic")

##############
# Data

traindata <- read.table("train.csv", header=T, sep=",")
testdata <- read.table("test.csv", header=T, sep=",")

# Median impute all missing values

# impute_NA is a little function that fills in NAs with either means or medians
impute.NA <- function(x, fill="mean"){
  if (fill=="mean")
  {
    x_complete <- ifelse(is.na(x), mean(x, na.rm=TRUE), x)
  }

  if (fill=="median")
  {
    x_complete <- ifelse(is.na(x), median(x, na.rm=TRUE), x)
  }

  return(x_complete)
}

traindata$age = impute.NA(traindata$age,fill="median")
testdata$age = impute.NA(testdata$age,fill="median")

traindata$fare = impute.NA(traindata$fare,fill="median")
testdata$fare = impute.NA(testdata$fare,fill="median")

# as factor
traindata$survived<-as.factor(traindata$survived)


###################
# Random Forest
  
    library(randomForest)

    # Fit the model by supplying gbm with the formula from above_ 
    # Including the train.fraction and cv.folds argument will perform 
    # cross-validation 

    rf_titanic <- randomForest(survived ~ pclass +sex + age + sibsp + parch + fare
        , n_trees=5000, data=traindata,importance=TRUE)

    rf_titanic
    importance(rf_titanic)


    rf_titanic_pvalues_train = predict(rf_titanic,type="prob")[,2]

    # Use the trained model to predict survival from the test data. 
    # probability of surviving values:
    rf_titanic_pvalues_test <- predict(rf_titanic, newdata=testdata,type="prob")[,2]

    # Package it in a dataframe and write it to a .csv file for uploading.
    rf_titanic_prediction_test = data.frame("survived"=as.numeric(rf_titanic_pvalues_test>0.5))
    write.table(rf_titanic_prediction_test, "rf_titanic_prediction_test", sep=",", row.names=FALSE)


###################
# Logit

  #Train
    logit_titanic = glm(survived ~ 
      pclass +sex + age + sibsp + parch + fare + embarked + pclass:sex
      # + fare:sex + sibsp:sex + pclass:fare  // some useless cross-variates for step example
        ,data=traindata,family=binomial(logit))

    summary(logit_titanic)
    
    # Stepwise model selection. k = log(nvalues) corresponds to BIC
    logit_titanic_reduced = step(logit_titanic,direction="both",k=log(dim(traindata)[2]))
    summary(logit_titanic_reduced)  # thrown out some variables

    rm(logit_titanic)  # use reduced model only


    # Train predictions
    logit_titanic_pvalues_train = predict(logit_titanic_reduced,type="response")

   # Test predictions
    logit_titanic_pvalues_test  = predict(logit_titanic_reduced,newdata=testdata,type="response")


####################
# Conditional tree

  library(party)

  ctree_titanic <- ctree(survived ~ pclass +sex + age + sibsp + parch + fare, data=traindata)

  plot(ctree_titanic) 
  ctree_titanic_pvalues_train = do.call(rbind,predict(ctree_titanic,type="prob"))[,2]
  ctree_titanic_pvalues_test = do.call(rbind,predict(ctree_titanic,newdata = testdata, type="prob"))[,2]




##################
# Model Comparison

par(mfrow=c(2,1))
par(mar=c(3,3,2,2))

# in-sample predictions
    plot(logit_titanic_pvalues_train,rf_titanic_pvalues_train,
      title("Train Data:  Logit vs RandomForest p-values"),xlab="logit",ylab="randomForest")

    plot(logit_titanic_pvalues_train,ctree_titanic_pvalues_train,
      title("Train Data:  Logit vs Ctree p-values"),xlab="logit",ylab="ctree")

# out-sample predictions
    plot(logit_titanic_pvalues_test,rf_titanic_pvalues_test,
            title("Test Data:  Logit vs RandomForest p-values"),xlab="logit",ylab="randomForest")

    plot(logit_titanic_pvalues_test,ctree_titanic_pvalues_test,
      title("Test Data:  Logit vs Ctree p-values"),xlab="logit",ylab="ctree")



############
# Blending

rf_weight = 0.5
logit_weight = 0.4
ctree_weight = 0.3

ensemble_titanic_pvalues_test = (rf_titanic_pvalues_test*rf_weight+
                          logit_titanic_pvalues_test*logit_weight+
                          ctree_titanic_pvalues_test*ctree_weight)/(sum(rf_weight,logit_weight,ctree_weight))

ensemble_titanic_prediction = ifelse(ensemble_titanic_pvalues_test>0.5,1,0)

write.csv(ensemble_titanic_prediction,"ensemble_titanic_prediction")

