# USAGE
# python Ensemble_classifier.py --dataset kaggle_dogs_vs_cats

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-R", "--estimators", type=int, default=50,
	help="Random forest")   
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))


rawImages = []

labels = []


for (i, imagePath) in enumerate(imagePaths):

	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]


	pixels = image_to_feature_vector(image)

	rawImages.append(pixels)

	labels.append(label)


	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))


rawImages = np.array(rawImages)

labels = np.array(labels)



(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)



print("[INFO] evaluating raw pixel accuracy...")
knn = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
knn.fit(trainRI, trainRL)
knn_acc = knn.score(testRI, testRL)
print("[INFO] knn: {:.2f}%".format(knn_acc * 100))

Log = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')                               
Log.fit(trainRI, trainRL)
Log_acc = Log.score(testRI, testRL)
print("[INFO] Log: {:.2f}%".format(Log_acc * 100))


Ran = RandomForestClassifier(n_estimators=args["estimators"],
	n_jobs=args["jobs"])                               
Ran.fit(trainRI, trainRL)
Ran_acc = Ran.score(testRI, testRL)
print("[INFO] Ran: {:.2f}%".format(Ran_acc * 100))

estimators=[('knn', knn), ('rf', Ran), ('log_reg', Log)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(trainRI, trainRL)
Ens_acc = ensemble.score(testRI, testRL)
print("[INFO] Ens: {:.2f}%".format(Ens_acc * 100))
