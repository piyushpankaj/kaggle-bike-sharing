# This script creates a sample submission using Random Forests
# and also plots the feature importance from the trained model.
#
# To submit the sample, download 1_random_forest_submission.csv
# from the Output tab and submit it as normal to the competition
# (through https://www.kaggle.com/c/bike-sharing-demand/submissions/attach)
#
# Click "fork" to run this script yourself and make tweaks

library(ggplot2)
library(lubridate)
library(randomForest)

set.seed(1)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

library(randomForest)


rf <- randomForest(extractFeatures(train), train$count, ntree=100, importance=TRUE)
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
     geom_bar(stat="identity", fill="#53cfff") +
     coord_flip() + 
     theme_light(base_size=20) +
     xlab("Importance") +
     ylab("") + 
     ggtitle("Random Forest Feature Importance\n") +
     theme(plot.title=element_text(size=18))

ggsave("2_feature_importance.png", p)