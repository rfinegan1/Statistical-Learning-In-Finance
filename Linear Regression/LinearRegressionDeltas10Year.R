library(ggplot2)
library(caret)
library(MASS)
library(glmnet)
library(class)
library(ISLR)
library(tidyverse)

setwd("/Users/ryanfinegan/Documents")                # my working directory
df<-read.csv("10yrforecasting_r.csv")                # file for 10 year prediction
dates <- as.POSIXct(df$Dates, format = "%m/%d/%Y")   # converting to get just the year
df$Dates<-format(dates, format="%Y")                 # getting the year in dates

glimpse(df)  # look at the data

### Simple Linear Regression with move derivative lag 1 weeks
ten_derivative <- lm(ten ~ tenderivativelag1, data = df)
summary(ten_derivative)        # p-value at 0.09 which is close to 0.05 significance
summary(ten_derivative)$sigma  # RSE of the Model
summary(ten_derivative)$sigma / mean(df$ten)  # divide it by y to get percentage error
summary(ten_derivative)$r.squared             # model can explain .348 percent of variance in weekly ten year yield
coefficients(ten_derivative)[2]               # barely a negative relationship (higher move sends 10 year yield lower)

### this function predicts intervals for the ten year yield given 95% confidence interval with certain value of the derivative of the ten year lagged 1 week
predict(ten_derivative, data.frame(tenderivativelag1 = 0.123), interval = "confidence", level = 0.95)
predict(ten_derivative, data.frame(tenderivativelag1 = 0.123), interval = "prediction", level = 0.95) # same thing but for prediction interval

### Plotting response against predictor
theme_set(theme_light())
ggplot(df, aes(x = tenderivativelag1, y = ten)) + 
  geom_point() + 
  geom_abline(intercept = coef(ten_derivative)[1], slope = coef(ten_derivative)[2], 
              col = "deepskyblue3", 
              size = 1) + 
  geom_smooth(se = F)    # typical looking plot for deltas

### Diagnostic Plots
par(mfrow=c(2,2))
plot(ten_derivative)

### Multiple Linear Regression 
df.viz = subset(df, select = -c(Dates,direction,tenderivative,thirtyderivative,thirty,movederivative,
                                  move,dxyderivative,dxy) ) # getting rid of variables that aren't lagged
df.viz=subset(df.viz,select=c(ten,tenlag1,tenlag5,movelag1,movelag5,tenderivativelag1,tenderivativelag5,
                              movederivativelag1,movederivativelag5,thirtyderivativelag1,thirtyderivativelag5))
pairs(df.viz)                # data relationships
cor(df.viz[-1], df.viz$ten)  # correlation with ten year yield with rest of predictors
ten.year.lm <- lm(ten ~ ., data = df.viz)  # mlr model 
summary(ten.year.lm)         # tenlag1,movelag1,movelag5,tenderivativelag1,movederivativelag1

### Diagnostic Plots
par(mfrow=c(2,2))
plot(ten.year.lm)

### Interaction Effects (Show all Interactions)
summary(lm(formula = ten ~ . * ., data = df.viz))   # move and thirty year yield derivatives significance is interesting

### transformations of the variables, such as log(X), X^2
### Took awesome function from lmorgan95
best_predictor <- function(dataframe, response) {
  if (sum(sapply(dataframe, function(x) {is.numeric(x) | is.factor(x)})) < ncol(dataframe)) {
    stop("Make sure that all variables are of class numeric/factor!")
  }
  # pre-allocate vectors
  varname <- c()
  vartype <- c()
  R2 <- c()
  R2_log <- c()
  R2_quad <- c()
  AIC <- c()
  AIC_log <- c()
  AIC_quad <- c()
  y <- dataframe[ ,response]
  # # # # # NUMERIC RESPONSE # # # # #
  if (is.numeric(y)) {
    for (i in 1:ncol(dataframe)) {
      x <- dataframe[ ,i]
      varname[i] <- names(dataframe)[i]
      if (class(x) %in% c("numeric", "integer")) {
        vartype[i] <- "numeric"
      } else {
        vartype[i] <- "categorical"
      }
      if (!identical(y, x)) {
        # linear: y ~ x
        R2[i] <- summary(lm(y ~ x))$r.squared 
        # log-transform: y ~ log(x)
        if (is.numeric(x)) { 
          if (min(x) <= 0) { # if y ~ log(x) for min(x) <= 0, do y ~ log(x + abs(min(x)) + 1)
            R2_log[i] <- summary(lm(y ~ log(x + abs(min(x)) + 1)))$r.squared
          } else {
            R2_log[i] <- summary(lm(y ~ log(x)))$r.squared
          }
        } else {
          R2_log[i] <- NA
        }
        # quadratic: y ~ x + x^2
        if (is.numeric(x)) { 
          R2_quad[i] <- summary(lm(y ~ x + I(x^2)))$r.squared
        } else {
          R2_quad[i] <- NA
        }
      } else {
        R2[i] <- NA
        R2_log[i] <- NA
        R2_quad[i] <- NA
      }
    }
    print(paste("Response variable:", response))
    data.frame(varname, 
               vartype, 
               R2 = round(R2, 3), 
               R2_log = round(R2_log, 3), 
               R2_quad = round(R2_quad, 3)) %>%
      mutate(max_R2 = pmax(R2, R2_log, R2_quad, na.rm = T)) %>%
      arrange(desc(max_R2))
    # # # # # CATEGORICAL RESPONSE # # # # #
  } else {
    for (i in 1:ncol(dataframe)) {
      x <- dataframe[ ,i]
      varname[i] <- names(dataframe)[i]
      if (class(x) %in% c("numeric", "integer")) {
        vartype[i] <- "numeric"
      } else {
        vartype[i] <- "categorical"
      }
      if (!identical(y, x)) {
        # linear: y ~ x
        AIC[i] <- summary(glm(y ~ x, family = "binomial"))$aic 
        # log-transform: y ~ log(x)
        if (is.numeric(x)) { 
          if (min(x) <= 0) { # if y ~ log(x) for min(x) <= 0, do y ~ log(x + abs(min(x)) + 1)
            AIC_log[i] <- summary(glm(y ~ log(x + abs(min(x)) + 1), family = "binomial"))$aic
          } else {
            AIC_log[i] <- summary(glm(y ~ log(x), family = "binomial"))$aic
          }
        } else {
          AIC_log[i] <- NA
        }
        # quadratic: y ~ x + x^2
        if (is.numeric(x)) { 
          AIC_quad[i] <- summary(glm(y ~ x + I(x^2), family = "binomial"))$aic
        } else {
          AIC_quad[i] <- NA
        }
      } else {
        AIC[i] <- NA
        AIC_log[i] <- NA
        AIC_quad[i] <- NA
      }
    }
    print(paste("Response variable:", response))
    data.frame(varname, 
               vartype, 
               AIC = round(AIC, 3), 
               AIC_log = round(AIC_log, 3), 
               AIC_quad = round(AIC_quad, 3)) %>%
      mutate(min_AIC = pmin(AIC, AIC_log, AIC_quad, na.rm = T)) %>%
      arrange(min_AIC)
  } 
}
glimpse(df)
df.transform.check=subset(df,select = -c(Dates,direction,move,thirty,dxy,tenderivative,movederivative,
                                thirtyderivative,dxyderivative))   # this was to check another alternative (didn't show better results than df.viz)
### Took awesome function from lmorgan95

best_predictor(df.viz, "ten")    # function is saying the quad transformation needs testing
### telling me to mainly check quad for tenlag1,movederivativelag5,movelag1,tenderivativelag1,tenlag5,movederivativelag1,thirtyderivativelag5
transform.lm <- lm(ten ~ + I(tenlag1^2) + I(movederivativelag5^2) + I(movelag1^2) + I(tenderivativelag1^2) + I(tenlag5^2) + I(movederivativelag1^2) + I(thirtyderivativelag5^2), data = df.viz)
summary(transform.lm)
### significance increased for "I(tenlag1^2)", "I(movederivativelag5^2)", and "I(tenlag5^2)"

new.transform.lm <- lm(ten ~ + I(tenlag1^2) + I(movederivativelag5^2) + I(tenlag5^2), data = df.viz)
summary(new.transform.lm)
### Diagnostic Plots
par(mfrow=c(2,2))
plot(new.transform.lm)
### Try log transform of Target Variable [The Ten Year Yield] => cannot because deltas