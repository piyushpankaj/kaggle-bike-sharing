# -*- coding: utf-8 -*-
import math
import sys
import numpy as np
import datetime
from sklearn import ensemble
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
import scipy as sp
import matplotlib.pyplot as plt

count_season = [0]*4
count_weather = [0]*3
count_season[0] = 116
count_season[1] = 215
count_season[2] = 234
count_season[3] = 198
count_weather[0] = 205
count_weather[1] = 178
count_weather[0] = 118

weather_c = [0]*3

cols_to_use = (0,3,4,8,9,10,11,12,13,14,15,16)

def featureEngineeringHelper(row, isTrainData):
	global count_season
	row = row.split(',')
	(date, time) = row[0].split(' ')
	year = int(date.split('-')[0])
	month = int(date.split('-')[1])
	day = int(date.split('-')[2])
	weekday = datetime.datetime(year, month, day).weekday()
	year = year - 2011
	hour = int(time.split(':')[0])
	#hour = math.cos(((2*math.pi)/24)*hour)
	season = int(row[1])

	holiday = int(row[2])
	workingDay = int(row[3])
	weather = int(row[4])
	if(weather == 4):
		weather = 3

	sunday = 0
	clear =0 
	mist=0
	if(weather == 1):
		clear=1
	elif(weather == 2):
		mist = 1
	else:
		pass

	temp = float(row[5])
	atemp = float(row[6])
	humidity = float(row[7])
	windspeed = float(row[8])

	if(workingDay == 1 and ((hour >= 7 and hour <= 9) or (hour >= 17 or hour <= 19))):
		peakhour = 1
	else:
		peakhour = 0

	if(workingDay == 0 and ((hour >= 10 and hour <= 18))):
		peakhour = 1
	else:
		peakhour = 0

	if isTrainData:
		casual = int(row[9])
		registered = int(row[10])
		count = int(row[11])	
		#count_weather[weather-1] = count_weather[weather-1]+ count
		#weather_c[weather-1] = weather_c[weather-1]+1
		#                    0     1      2     3       4      5       6      7      8       	9        10       11     12      13        14       15		        16
		transformed_row = [year, month, day, weekday, hour, clear, mist, sunday ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, count_season[season-1], casual, registered]
		# transformed_row = [year, month, day, weekday, hour, count_season[season-1] ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, casual, registered]
		#month used here

		return transformed_row
	else:
		# transformed_row = [year, month, day, weekday, hour, spring, summer, fall, holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour]	
		transformed_row = [year, month, day, weekday, hour, clear, mist, sunday ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, count_season[season-1]]
		return transformed_row

def featureEngineering(dataSet, isTrainData):
	dataSet.readline() #Skipping the headers
	dataSetModified = []
	while True:
		row = dataSet.readline()
		if not row:
			break
		transformed_row = featureEngineeringHelper(row, isTrainData) #y1 is a tuple of the form (casual, registered) 												 #x1 is a list of new features
		dataSetModified.append(transformed_row)
	#for x in xrange(0,3):
	#	print count_weather[x]/weather_c[x]
	#sys.exit(0)
	return dataSetModified     

def plotFeatureImportance(forest):

	x_bar = ["year","weekday","hour","holiday","workingDay","weather","temp","atemp","humidity","windspeed","peakhour","count_season"]
	new_XBar = []
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]
	#indices = np.array(importances)[::-1]
	# Print the feature ranking
	print("Feature ranking:")

	for f in range(12):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	print indices
	for x in xrange(0,12):
		new_XBar.append(x_bar[indices[x]])
	print new_XBar
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(12), importances[indices],
	       color="g", yerr=std[indices], align="center")
	plt.xticks(range(12),new_XBar)
	plt.xlim([-1, 12])
	plt.show()

def getRMLSE(Y_predicted):
	Y_actual = np.load("../Dataset/answers.npy")
	return score_func(Y_actual, Y_predicted)

def score_func(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    res = math.sqrt( np.sum( np.square(np.log(y_pred+1) - np.log(y+1)) ) / len(y) )
    return res

def llfun(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll

def cross_validation(train, target):
	#read in  data, parse into training and target sets
	#dataset = np.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    

	#In this case we'll use a random forest, but this could be any classifier
	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':1 }
	clf = ensemble.RandomForestRegressor(**params)

	params = {'n_estimators': 130, 'max_depth': 6, 'random_state':0}
	gbm = ensemble.GradientBoostingRegressor(**params)

	#Simple K-Fold cross validation. 5 folds.
	cv = cross_validation.KFold(len(train), k=5, indices=False)

	#iterate through the training and test cross validation segments and
	#run the classifier on each one, aggregating the results into a list
	results = []
	for traincv, testcv in cv:
		rf1_Y_CV = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
		gbm_Y_CV = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
		results.append(llfun(target[testcv], [x[1] for x in probas]) )

	#print out the mean of the cross-validated results
	print "Results: " + str( np.array(results).mean() )

def randomForestModel():

	# rf = ensemble.RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=12,
 #           max_features='auto', max_leaf_nodes=None, min_samples_leaf=2,
 #           min_samples_split=2, min_weight_fraction_leaf=0.0,
 #           n_estimators=1000, n_jobs=1, oob_score=False, random_state=None,
 #           verbose=0, warm_start=False)

	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':-1 }
	rf = ensemble.RandomForestRegressor(**params)
	return rf

def gradientDescentModel():
	params = {'n_estimators': 100, 'max_depth': 6, 'random_state':0}
	gbm = ensemble.GradientBoostingRegressor(**params)
	return gbm

def extraTreesRegressor(X, Y_casual, Y_registered, testSet_final):
	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':1 }
	eTreeReg1 = ensemble.ExtraTreesRegressor()
	eTreeReg2 = ensemble.ExtraTreesRegressor()

	eTreeReg1.fit(X, Y_casual)
	eTreeReg2.fit(X, Y_registered)

	eTreeReg1_Y = np.exp(eTreeReg1.predict(testSet_final))-1
	eTreeReg2_Y = np.exp(eTreeReg2.predict(testSet_final))-1

	final_prediction = np.intp(np.around(eTreeReg1_Y + eTreeReg2_Y))
	return final_prediction

def elasticnet(X,Y_casual, Y_registered, testSet_final):
	alpha=0.001
	l1_ratio=0.1
	glmnet1 = linear_model.ElasticNetCV()
	glmnet2 = linear_model.ElasticNetCV()

	glmnet1.fit(X, Y_casual)
	glmnet2.fit(X, Y_registered)

	glmnet1_Y = np.exp(glmnet1.predict(testSet_final))-1
	glmnet2_Y = np.exp(glmnet2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(glmnet1_Y + glmnet2_Y))
	return final_prediction

def lasso(X, Y_casual, Y_registered, testSet_final):
	alpha = 0.5
	lasso1 =linear_model.Lasso(alpha=alpha)
	lasso2 =linear_model.Lasso(alpha=alpha)

	lasso1.fit(X, Y_casual)
	lasso2.fit(X, Y_registered)

	lasso1_Y = np.exp(lasso1.predict(testSet_final))-1
	lasso2_Y = np.exp(lasso2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(lasso1_Y + lasso2_Y))
	return final_prediction

# def stackingPredictor(rf,gbm):
# 	stacking = []
# 	stacking.append(rf)
# 	stacking.append(gbm)
# 	stacking = np.array(stacking)
# 	stacking = np.transpose(stacking)
# 	return stacking

# def stacking():
# 	print "--------Ensemble Stacking ---------------"
# 	print 

# 	rf1_Y_train =  rf1.predict(X)
# 	rf2_Y_train = rf2.predict(X)
# 	gbm1_Y_train = gbm1.predict(X)
# 	gbm2_Y_train = gbm2.predict(X)
	
# 	rf1_Y = rf1.predict(testSet_final)	
# 	rf2_Y = rf2.predict(testSet_final)
# 	gbm1_Y = gbm1.predict(testSet_final)
# 	gbm2_Y = gbm2.predict(testSet_final)

# 	stacking_cas_train = stackingPredictor(rf1_Y_train , gbm1_Y_train )
# 	stacking_cas_test = stackingPredictor(rf1_Y , gbm1_Y )
	
# 	stacking_reg_train = stackingPredictor(rf2_Y_train , gbm2_Y_train )
# 	stacking_reg_test = stackingPredictor(rf2_Y, gbm2_Y)
	
# 	#clf = svm.SVR()

# 	stacker_cas= linear_model.LinearRegression()
# 	stacker_reg= linear_model.LinearRegression()
	
# 	stacker_cas.fit(stacking_cas_train,Y_casual)	
# 	stacker_reg.fit(stacking_reg_train,Y_registered)	

# 	final_prediction_cas = stacker_cas.predict(stacking_cas_test)
# 	final_prediction_reg = stacker_reg.predict(stacking_reg_test)
# 	final_prediction = (np.exp(final_prediction_cas)+1) + (np.exp(final_prediction_reg)+1)
# 	final_prediction = np.intp(np.around(final_prediction))

# 	return final_prediction

def supportVectorRegression(X, Y_casual, Y_registered, testSet_final):
	svr1 = SVR(kernel='rbf', gamma=0.1)
	svr2 = SVR(kernel='rbf', gamma=0.1)
	svr1.fit(X, Y_casual)
	svr2.fit(X, Y_registered)
	svr1_Y = np.exp(svr1.predict(testSet_final))-1
	svr2_Y = np.exp(svr2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(svr1_Y + svr2_Y))
	return final_prediction

def rf(X, Y_casual, Y_registered, testSet_final):
	rf1 = randomForestModel()  #train for casual
	rf2 = randomForestModel()  #train for registered
	rf1.fit(X, Y_casual)
	rf2.fit(X, Y_registered)  
	rf1_Y = np.exp(rf1.predict(testSet_final))-1
	rf2_Y = np.exp(rf2.predict(testSet_final))-1
	final_prediction = (rf1_Y + rf2_Y )
	final_prediction = np.intp(np.around(final_prediction))  #round and convert to integer
	return final_prediction

def gbm(X, Y_casual, Y_registered, testSet_final):

	gbm1 = gradientDescentModel()   #train for casual
	gbm2 = gradientDescentModel()   #train for registered
	gbm1.fit(X, Y_casual)
	gbm2.fit(X, Y_registered)
	gbm1_Y = np.exp(gbm1.predict(testSet_final))-1
	gbm2_Y = np.exp(gbm2.predict(testSet_final))-1
	final_prediction = (gbm1_Y + gbm2_Y)
	final_prediction = np.intp(np.around(final_prediction))  #round and convert to integer
	return final_prediction


def rfGbmCombined(X, Y_casual, Y_registered, testSet_final):
	#creating models
	rf1 = randomForestModel()  #train for casual
	rf2 = randomForestModel()  #train for registered
	gbm1 = gradientDescentModel()   #train for casual
	gbm2 = gradientDescentModel()   #train for registered
	#fitting models
	# rf1.fit(train_X, train_Y[:, 0])  #train_Y[:, 0] - use 0th column of train_Y
	rf1.fit(X, Y_casual)

	plotFeatureImportance(rf1)
	
	rf2.fit(X, Y_registered)  
	gbm1.fit(X, Y_casual)
	gbm2.fit(X, Y_registered)

	#prediction
	rf1_Y = np.exp(rf1.predict(testSet_final))-1
	rf2_Y = np.exp(rf2.predict(testSet_final))-1
	gbm1_Y = np.exp(gbm1.predict(testSet_final))-1
	gbm2_Y = np.exp(gbm2.predict(testSet_final))-1

	#Average the prediction from classifiers
	final_prediction = (rf1_Y + rf2_Y + gbm1_Y + gbm2_Y)/2
	final_prediction = np.intp(np.around(final_prediction))  #round and convert to integer
	return final_prediction

def write_result_to_file(final_prediction, dateTimeColumn):
	f = open("../Dataset/submit.csv", "w")
	f.write("datetime,count\n")
	numRows = final_prediction.size
	for i in xrange(0,numRows):
		string_to_write = dateTimeColumn[i] + "," + str(final_prediction[i]) + "\n"
		f.write(string_to_write)
	f.close()

def getDateTimeColumn(testSetOriginal):
	testSetOriginal.seek(0)
	dateTimeColumn = []
	testSetOriginal.readline()
	while(True):
		row = testSetOriginal.readline()
		if not row:
			break
		row = row.split(',')
		dateTimeColumn.append(row[0])
	return dateTimeColumn

def grid_search(X, Y):
	
	from sklearn.grid_search import GridSearchCV
	
	tuned_parameters = [{'max_features': ['sqrt', 'log2', 'auto'], 'max_depth': [5, 8, 12], 'min_samples_leaf': [2, 5, 10]}]
	rf =  GridSearchCV(ensemble.RandomForestRegressor(n_jobs=1, n_estimators=1000), tuned_parameters, cv=3, verbose=2).fit(X, Y)
	print 'Best Parameters:'
	print rf.best_estimator_
	sys.exit(0)


if __name__ == '__main__':
	trainSetOriginal = open('../Dataset/train.csv', "r")
	testSetOriginal = open('../Dataset/test.csv', "r")

	#Feature Engineering
	trainSetModified = featureEngineering(trainSetOriginal, True)  #numpy array format

	testSetModified = featureEngineering(testSetOriginal, False)  #numpy array format

	#splitting trainset into X and Y components. Y consists of [casual, registered]
	trainSetMod_X = []
	trainSetMod_Y = []
	for row in trainSetModified:
		trainSetMod_X.append(row[:-2])
		trainSetMod_Y.append(row[-2:])	#[casual, registered]

	#converting sets into format acceptable by the learning models
	train_X = np.array(trainSetMod_X) 
	train_Y = np.array(trainSetMod_Y) 
	testSet = np.array(testSetModified)

	Y_casual = np.log(train_Y[:, 0]+1)
	Y_registered = np.log(train_Y[:, 1]+1)

	X = train_X[:, cols_to_use]              #final train set
	testSet_final = testSet[:, cols_to_use]  #final test set
	

	# Uncomment the model you want to train
	
	#grid_search(X, Y_casual)
	#RandomForest and GradientBoosting Combined
	#final_prediction = rfGbmCombined(X, Y_casual, Y_registered, testSet_final)
	#np.save("rf_Gbm.npy",final_prediction)

	final_prediction = gbm(X, Y_casual, Y_registered, testSet_final)
	np.save("../Dataset/gbm.npy",final_prediction)

	#final_prediction = rf(X, Y_casual, Y_registered, testSet_final)
	#np.save("rf.npy",final_prediction)

	#Elastic net model
	# final_prediction = elasticnet(X, Y_casual, Y_registered, testSet_final)

	#lasso regression model
	# final_prediction  = lasso(X, Y_casual, Y_registered, testSet_final)

	#Support Vector Regression
	# final_prediction = supportVectorRegression(X, Y_casual, Y_registered, testSet_final)

	#Extra Trees Regressor
	# final_prediction = extraTreesRegressor(X, Y_casual, Y_registered, testSet_final)

	error = getRMLSE(final_prediction)
	print error
	#get datetime column
	dateTimeColumn = getDateTimeColumn(testSetOriginal)

	#writing final_prediction to file
	write_result_to_file(final_prediction, dateTimeColumn)
