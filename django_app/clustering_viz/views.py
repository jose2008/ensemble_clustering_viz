from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.shortcuts import render
from clustering_viz.clustering.core import KMeansAlgorithm

from sklearn import datasets 
from clustering_viz.clustering.core.KMeansAlgorithm import *
from clustering_viz.clustering.core.BirchAlgorithm import *
#from clustering_viz.clustering.core.SomAlgorithm import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from django.template import RequestContext


from django.template.response import TemplateResponse 

from django.shortcuts import render_to_response

import json
import decimal
def trun_n_d(n,d):
    s=repr(n).split('.')
    if (len(s)==1):
        return int(s[0])
    return float(s[0]+'.'+s[1][:d])

# Create your views here.
#from clustering.core import KMeansAlgorithm, SomAlgorithm
def index(request):
	username = request.GET.get('username', None)

	iris = datasets.load_iris()

	if not username:
		test = KMeansAlgorithm(iris.data, {'kmeans':2} )
	else:
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
	return render_to_response('index.html',  {'model_list': "holaaaa"})



def kmean_model(request):
	username = request.GET.get('username', None)

	iris = datasets.load_iris()
	if not username:
		test = KMeansAlgorithm(iris.data, {'kmeans':2} )
	else:
		test = KMeansAlgorithm(iris.data, {'kmeans':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(iris.data)
	feature = sklearn_pca.fit_transform(std)


	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()

	list_set = []
	for i in range(int(username)):
		s = set()
		list_set.append(s)

	n = len(test.m_resultLabels)
	for i in range(n):
		for j in range(len(list_set)):
			list_set[test.m_resultLabels[i]].add(i)

	matrix = np.zeros((n, n))
	l = 0
	for k in range(len(list_set)):
		for i in range(len(list_set[k])):
			for j in range(len(list_set[k])):
				matrix[i + l][j+l] = 1
		l = l + len(list_set[k])


	list_matriz_kmean = matrix.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']

	ex = test.m_resultLabels.tolist()

	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))



def birch_model(request):
	username = request.GET.get('username', None)

	iris = datasets.load_iris()
	if not username:
		test = BirchAlgorithm(iris.data, {'birch':2} )
	else:
		test = BirchAlgorithm(iris.data, {'birch':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(iris.data)
	feature = sklearn_pca.fit_transform(std)

	matrix_feature_birch = np.matrix( feature )
	matrix_label_birch = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_birch, matrix_label_birch), axis=1)
	tolist_knn = matrix_general.tolist()

	list_set = []
	for i in range(int(username)):
		s = set()
		list_set.append(s)

	n = len(test.m_resultLabels)
	for i in range(n):
		for j in range(len(list_set)):
			list_set[test.m_resultLabels[i]].add(i)

	matrix = np.zeros((n, n))
	l = 0
	for k in range(len(list_set)):
		for i in range(len(list_set[k])):
			for j in range(len(list_set[k])):
				matrix[i + l][j+l] = 1
		l = l + len(list_set[k])

	list_matriz_kmean = matrix.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3)}
	return HttpResponse(json.dumps(data))


def som_model(request):
	username = request.GET.get('username', None)
	iris = datasets.load_iris()
	print(username)
	if not username:
		test = SomAlgorithm( iris.data, {'som_a':1, 'som_b':3} )
	else:
		test = SomAlgorithm( iris.data, {'som_a':1, 'som_b':3} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(iris.data)
	feature = sklearn_pca.fit_transform(std)

	matrix_feature_som = np.matrix(feature)
	matrix_label_som   = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_som, matrix_label_som), axis=1)
	totlist_som = matrix_general.tolist()

	list_set = []
	for i in range(int(username)):
		s = set()
		list_set.append(s)

	n = len(test.m_resultLabels)
	for i in range(n):
		for j in range(len(list_set)):
			list_set[test.m_resultLabels[i]].add(i)

	matrix = np.zeros((n, n))
	l = 0
	for k in range(len(list_set)):
		for i in range(len(list_set[k])):
			for j in range(len(list_set[k])):
				matrix[i + l][j+l] = 1
		l = l + len(list_set[k])

	list_matriz_som = matrix.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	data = {'val1': tolist_knn, 'val2': list_matriz_som, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3),'list_labels':test.m_resultLabels }
	return HttpResponse(json.dumps(data))



def ensamble_model(request):
	username = request.GET.get('username', None)

	iris = datasets.load_iris()
	if not username:
		test = KMeansAlgorithm(iris.data, {'kmeans':2} )
	else:
		test = KMeansAlgorithm(iris.data, {'kmeans':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(iris.data)
	feature = sklearn_pca.fit_transform(std)


	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()

	list_set = []
	for i in range(int(username)):
		s = set()
		list_set.append(s)

	n = len(test.m_resultLabels)
	for i in range(n):
		for j in range(len(list_set)):
			list_set[test.m_resultLabels[i]].add(i)

	matrix = np.zeros((n, n))
	l = 0
	for k in range(len(list_set)):
		for i in range(len(list_set[k])):
			for j in range(len(list_set[k])):
				matrix[i + l][j+l] = 1
		l = l + len(list_set[k])


	list_matriz_kmean = matrix.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']

	ex = test.m_resultLabels.tolist()

	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex}
	return HttpResponse(json.dumps(data))
