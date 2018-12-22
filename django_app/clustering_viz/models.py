from django.shortcuts import render
from django.db import models
from django.http import HttpResponse
from clustering_viz.clustering.core import KMeansAlgorithm

from sklearn import datasets 
from clustering_viz.clustering.core.KMeansAlgorithm import *
#from clustering_viz.clustering.core.BirchAlgorithm import *
#from clustering_viz.clustering.core.SomAlgorithm import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler


# Create your models here.


def model_clustering(request):
	username = request.GET.get('username', None)
	print("name..........................................")
	print(username)


	iris = datasets.load_iris()
	test = KMeansAlgorithm(iris.data, {'kmeans':int(username)} )
	test.run()
	#test.m_resultLabels



	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(iris.data)
	feature = sklearn_pca.fit_transform(std)


	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()



	print(tolist_knn)


	return render(request,'index.html', {"model_list":tolist_knn})
	#return HttpResponse("hello......")
	#return ""



