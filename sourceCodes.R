library(ggplot2)
library(caret)
library(gridExtra)
library(randomForest)

# loading the dataset
data <- read.csv("dataset.CSV", header = T)
str(data); prop.table(table(data$Gender)) # dataset summary

plot1 <- ggplot(data, aes(x=Gender, y=Height, fill=Gender)) + 
  ylab("Height (inch)") +
  geom_boxplot(alpha=0.3) + 
  theme(legend.position="none") + 
  scale_fill_brewer(palette="Dark2")

plot2 <- ggplot(data, aes(x=Gender, y=Weight, fill=Gender)) + 
  ylab("Weight (pound)") +
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") + 
  scale_fill_brewer(palette="Dark2")

grid.arrange(plot1, plot2, ncol=2)

# splitting train and test sets
intrain <- createDataPartition(y=data$Gender,p=0.5,list=FALSE)
train <- data[intrain,]
test <- data[-intrain,]

prop.table(table(train$Gender)); prop.table(table(test$Gender)) # distribution of genders are same for both sets

plot3 <- ggplot(train, aes(x=Gender, y=Height, fill=Gender)) + 
  ggtitle("Train Set") +
  ylab("Height (inch)") +
  ylim(50,80) +
  geom_boxplot(alpha=0.3) + 
  theme(legend.position="none", plot.title = element_text(hjust = 0.5)) + 
  scale_fill_brewer(palette="Dark2")

plot4 <- ggplot(test, aes(x=Gender, y=Height, fill=Gender)) +
  ggtitle("Test Set") +
  ylab("Height (inch)") +
  ylim(50,80) +
  geom_boxplot(alpha=0.3) + 
  theme(legend.position="none", plot.title = element_text(hjust = 0.5)) + 
  scale_fill_brewer(palette="Dark2")

grid.arrange(plot3, plot4, ncol=2)

plot5 <- ggplot(train, aes(x=Gender, y=Weight, fill=Gender)) + 
  ylab("Weight (pound)") +
  ylim(50,300) +
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") + 
  scale_fill_brewer(palette="Dark2")

plot6 <- ggplot(test, aes(x=Gender, y=Weight, fill=Gender)) + 
  ylab("Weight (pound)") +
  ylim(50,300) +
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") + 
  scale_fill_brewer(palette="Dark2")

grid.arrange(plot5, plot6, ncol=2)

mat <- matrix(nrow = 5000, ncol = 0)
for (p in (1:9)/10) {
  mat <- cbind(mat,sample(c("Male", "Female"), nrow(test), replace = TRUE, prob=c(p, 1-p)))
}

df <- as.data.frame(mat)

# positive class: male
accuracy <- matrix(nrow = 0, ncol = 2)
for (i in 1:9) {
  tp <- table(test$Gender,df[,i])[2,2]
  tn <- table(test$Gender,df[,i])[1,1]
  fp <- table(test$Gender,df[,i])[1,2]
  fn <- table(test$Gender,df[,i])[2,1]
  accuracy <- rbind(accuracy, c(i/10,((tp+tn)/(tp+tn+fp+fn))))
}

f1_score <- matrix(nrow = 0, ncol = 1)
for (i in 1:9) {
  tp <- table(test$Gender,df[,i])[2,2]
  tn <- table(test$Gender,df[,i])[1,1]
  fp <- table(test$Gender,df[,i])[1,2]
  fn <- table(test$Gender,df[,i])[2,1]
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  f1_score <- rbind(f1_score, 2*(precision*recall)/(precision+recall))
}

perf_df <- as.data.frame(cbind(accuracy,f1_score))
perf_df <- setNames(perf_df, c("male_prob","accuracy","f1_score"))

plot7 <- ggplot(perf_df, aes(x = male_prob, y = accuracy)) + 
  geom_line(color = "red", size = 1) + 
  ylim(0,1) + 
  theme_light()

plot8 <- ggplot(perf_df, aes(x = male_prob, y = f1_score)) + 
  geom_line(color = "green", size = 1) + 
  ylim(0,1) + 
  theme_light()

grid.arrange(plot7, plot8, ncol=2)

# logistic regression, positive class: male
model <- glm(Gender ~., family=binomial(link='logit'), data=train)
model
pred_train <- predict.glm(model, newdata = train, type = "response")
summary(pred_train)
head(pred_train)

pred_train <- as.data.frame(ifelse(pred_train >= 0.5, "Male", "Female"))
pred_train$Gender <- pred_train[,1]
pred_train[,1] <- NULL

# confusion matrix and F1 score
cm <- table(train$Gender,pred_train$Gender)
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[1,2]
fn <- cm[2,1]
precision <- tp/(tp+fp)
recall <- tp/(tp+fn)
f1_score_log <- 2*(precision*recall)/(precision+recall)

# random forest, positive class: male
rf <- randomForest(Gender ~ ., ntree = 100, data = train)
plot(rf) # error significantly decreases until 20 trees
rf <- randomForest(Gender ~ ., ntree = 20, data = train)
pred_rf <- predict(rf, train, type = "prob")
summary(pred_rf)
head(pred_rf)

pred_rf <- pred_rf[,2]
pred_rf <- as.data.frame(ifelse(pred_rf >= 0.5, "Male", "Female"))
pred_rf$Gender <- pred_rf[,1]
pred_rf[,1] <- NULL

# confusion matrix and F1 score
cm <- table(train$Gender,pred_rf$Gender)
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[1,2]
fn <- cm[2,1]
precision <- tp/(tp+fp)
recall <- tp/(tp+fn)
f1_score_rf <- 2*(precision*recall)/(precision+recall)

ifelse(f1_score_rf >= f1_score_log, 
       print("Go for Random Forest!"), 
       print("Go for Logistic Regression!"))

# performance on the test set
pred_rf_test <- predict(rf, test, type = "prob")
summary(pred_rf_test)
head(pred_rf_test)

pred_rf_test <- pred_rf_test[,2]
pred_rf_test <- as.data.frame(ifelse(pred_rf_test >= 0.5, "Male", "Female"))
pred_rf_test$Gender <- pred_rf_test[,1]
pred_rf_test[,1] <- NULL

# confusion matrix and F1 score
cm <- table(test$Gender,pred_rf_test$Gender)
print(cm)
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[1,2]
fn <- cm[2,1]
precision <- tp/(tp+fp)
recall <- tp/(tp+fn)
accuracy <- (tp+tn)/(tp+tn+fp+fn)
f1_score_test <- 2*(precision*recall)/(precision+recall)
print(f1_score_test)
