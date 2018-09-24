from math import sqrt




# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)


#Evalauet Model
def evaluate_algorithm(dataset, algorithm):
	test_set=list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

		
#Calculate mean value of list of numbers
def mean(values):
	return sum(values)/ float (len(values))


#Calculate variance  of list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])



#Calculate the co-variance
def covariance(x,mean_x,y,mean_y):
	covar=0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar



#Calculate Coefficients
def coefficients(dataset):
	x=[row[0] for row in dataset]
	y=[row[1] for row in dataset]
	mean_x, mean_y= mean(x) , mean(y)
	b1=covariance(x, mean_x, y, mean_y)/ variance(x, mean_x)
	b0=mean_y-b1*mean_x
	return [b0, b1]





#Model
def simple_linear_regression(train, test):
	predictions=list()
	b0, b1= coefficients(train)
	for row in test:
		yhat= b0 +b1 * row[0]
		predictions.append(yhat)
	return predictions

















#Calculate mean and variance
dataset= [[1,1], [2,3], [4,3], [3,2],[5,5]]

x=[row[0] for row in dataset]
y=[row[1] for row in dataset]

mean_x, mean_y= mean(x), mean(y)
var_x, var_y= variance(x, mean_x) , variance(y, mean_y)
covar= covariance(x, mean_x, y, mean_y)
b0, b1= coefficients(dataset)
rmse= evaluate_algorithm(dataset, simple_linear_regression)

print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
print ('covariance : %.3f' % (covar))
print ('coefficients: b0 is %.3f and b1 is %.3f' %(b0,b1))
print ('RMSE %.3f' % (rmse))