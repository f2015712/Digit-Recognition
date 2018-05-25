from sklearn import datasets #datasets imported from sklearn
from sklearn.svm import SVC #SVC is for classifier
from scipy import misc


digits = datasets.load_digits() #digit datasets loaded for training
features = digits.data #features stores matrix corresponding to different digits
labels = digits.target #labels stores the digits

clf = SVC(gamma = 0.001)
clf.fit(features, labels) #clf is trained on features and labels


img = misc.imread("2.jpg") #loading image
img = misc.imresize(img, (8,8)) #changing image to 64 pixels.We take 64 becuase the training data is also 64 pixels.
img = img.astype(digits.images.dtype) #making img data type same as the digits.image datatype. img is unsigned type and digits.images is float type. So, the two should be made same.
img = misc.bytescale(img, high=16, low=0) 


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0) #x_test is the 1D array to store pixels of the input image i.e same format as the features which contains 64 elements



print(clf.predict([x_test])) #predict the output
