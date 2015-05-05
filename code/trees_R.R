CEVM <- function(data.trainset){

	mean.frame = aggregate(cbind(temp,atemp,humidity,windspeed,casual,registered,count) ~ hour,data.trainset,mean)
	mean.frame$count = as.integer(mean.frame$count)
	mean.frame$registered = as.integer(mean.frame$registered)
	mean.frame$casual = as.integer(mean.frame$casual)

	return (mean.frame)
}

factorization <- function(data.trainset){

	data.trainset$season = factor(data.trainset$season)
	data.trainset$holiday = factor(data.trainset$holiday)
	data.trainset$workingday = factor(data.trainset$workingday)

	data.trainset$weather = as.integer(data.trainset$weather)
	#data.trainset[data.trainset$weather==4] <- 3
	
	data.trainset$weather = factor(data.trainset$weather)

	data.trainset$date = substring(data.trainset$datetime,1,10)
	data.trainset$hour = as.integer(substring(data.trainset$datetime,12,13))
	data.trainset$year = as.integer(substring(data.trainset$date,0,4))
	data.trainset$month = as.numeric(format(as.Date(data.trainset$date), "%m"))
	data.trainset$weekday = as.integer(as.POSIXlt(data.trainset$date)$wday)	
	#write.csv(data.trainset,"trainFeaturised.csv",row.names=FALSE)
	data.trainset$month = factor(data.trainset$month)
	data.trainset$weekday = factor(data.trainset$weekday)	
	data.trainset$hour = factor(data.trainset$hour)
	#data.trainset[data.trainset$year == 2011] <- 0
	#data.trainset[data.trainset$year == 2012] <- 1
	
	data.trainset$year = factor(data.trainset$year)
	
	return (data.trainset)

}

rf <- function(maxTree,data.trainset,data.testset){

	library(randomForest)
	set.seed(415)
	data.trainset$casual = as.integer(data.trainset$casual)
	data.trainset$registered = as.integer(data.trainset$registered)
	data.trainset$count = as.integer(data.trainset$count)
	

	fit_casual <- randomForest(casual ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset,ntree=maxTree,mtry=5,importance=TRUE)
	fit_reg <- randomForest(registered ~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset,ntree=maxTree,mtry=5,importance=TRUE)
	#fit_count <- randomForest(count ~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset,ntree=maxTree,mtry=5,importance=TRUE)
	

	casual.pred <- predict(fit_casual, data.testset)
	reg.pred <- predict(fit_reg, data.testset)
	#print (casual.pred)
	count.pred = round(casual.pred + reg.pred)

	print (head(count.pred))
	submission = data.frame(datetime = data.testset$datetime, count=count.pred)
	write.csv(submission, file="craniumCrushers_v3.csv",row.names=FALSE)
	#fit_count
	#print(paste("OOB Error is ",fit_count$err.rate[maxTree]))
}

ctrees <- function(data.trainset,data.testset){
	library(party)
	data.trainset$casual = as.integer(data.trainset$casual)
	data.trainset$registered = as.integer(data.trainset$registered)
	data.trainset$count = as.integer(data.trainset$count)

	fit_count <- ctree(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset)
	count.pred <- predict(fit_count, data.testset)
	#reg.pred <- predict(fit_reg, data.testset)
	count.pred = as.integer(count.pred)
	print (head(count.pred))
	submission = data.frame(datetime = data.testset$datetime, count=count.pred)
	write.csv(submission, file="craniumCrushers_ctree_v1.csv",row.names=FALSE)

}

gradientBoosting <- function(data.trainset,data.testset){
	library(gbm)

	data.trainset$casual = as.integer(data.trainset$casual)
	data.trainset$registered = as.integer(data.trainset$registered)
	data.trainset$count = as.integer(data.trainset$count)

	gbm.fit <- gbm(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,
							data=data.trainset,
							interaction.depth = 2,
							distribution="gaussian",
							n.trees = 4000,
							shrinkage=0.005,
							n.minobsinnode = 5,
							verbose=TRUE
							)

	best.iter <- gbm.perf(gbm.fit,method="OOB")
	print(best.iter)

	count.pred <- predict(gbm.fit, data.testset,best.iter,type="response",n.trees = 1000)
	count.pred[count.pred <0] <- 0
	count.pred = as.integer(count.pred)
	print (head(count.pred))
	submission = data.frame(datetime = data.testset$datetime, count=count.pred)
	write.csv(submission, file="craniumCrushers_gbm_v1.csv",row.names=FALSE)
}

data.trainset <- read.csv("train.csv", header=TRUE)
names(data.trainset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed","casual","registered","count")
data.trainset = factorization(data.trainset)

data.testset <- read.csv("test.csv", header=TRUE)
names(data.testset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed")
data.testset = factorization(data.testset)

rf(500,data.trainset,data.testset)
#ctrees(data.trainset,data.testset)
#gradientBoosting(data.trainset,data.testset)