from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.shortcuts import render
from clustering_viz.clustering.core import KMeansAlgorithm

from sklearn import datasets 
from clustering_viz.clustering.core.KMeansAlgorithm import *
from clustering_viz.clustering.core.KMeansAlgorithm2 import *
from clustering_viz.clustering.core.KMedoidsAlgorithm import *
from clustering_viz.clustering.core.DBScanAlgorithm import *
from clustering_viz.clustering.core.BirchAlgorithm import *
from clustering_viz.clustering.core.SomAlgorithm import *
from clustering_viz.clustering.core.EnsembleAlgorithm import *
from clustering_viz.clustering.core.EnsembleAgglomerativeAlgorithm import *
from clustering_viz.clustering.core.AgglomerativeAlgorithm import *
from clustering_viz.clustering.core.Majority_voteAlgorithm import *
from clustering_viz.clustering.core.MiniBatchKMeansAlgorithm import *
from clustering_viz.clustering.core.library import *
from django.views.decorators.csrf import csrf_exempt

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from django.template import RequestContext

from django.template.response import TemplateResponse 
from django.shortcuts import render_to_response
import metis


from sklearn.datasets.samples_generator import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

#DataSet = StandardScaler().fit_transform(X)



import json
import decimal
def trun_n_d(n,d):
    s=repr(n).split('.')
    if (len(s)==1):
        return int(s[0])
    return float(s[0]+'.'+s[1][:d])

def load_dataset(name):
    return np.loadtxt(name)

#DataSet = load_dataset('flame.txt')[:,[0,1]]		#N =240 k=2----test(de hecho)(1)
#DataSet = load_dataset('jain.txt')[:,[0,1]]			#N=373, k=2, D=2----test (4)
#DataSet = load_dataset('agregation.txt')[:,[0,1]]	#N=788 k=7 d=2
#DataSet = load_dataset('spiral.txt')[:,[0,1]]
#DataSet = load_dataset('unbalance.txt')				#N=6500
#DataSet = load_dataset('thyroid.txt')
#DataSet = load_dataset('a1.txt')					#N=3000, k=20 2d
#DataSet = load_dataset('wine.txt')					#N=178, k=3, D=13-----test((de hecho))(2)
#DataSet = load_dataset('breast.txt')				#N=699, k=2, D=9  (triste resultados)
#DataSet = load_dataset('yeast.txt')					#N=1484, k=10, D=8 
#DataSet = load_dataset('dim32.txt')			#N=1024 and k=16 (triste resultados)
#DataSet = load_dataset('glass.txt')			#N=214, k=7, D=9	(muy triste)
#DataSet = load_dataset('wdbc.txt')			#N=569, k=2, D=32 
#DataSet = load_dataset('bridge.txt')			#N=4096, D=16		
#DataSet = load_dataset('mnist.txt')
#DataSet = load_dataset('s1.txt')				# 2-d data with N=5000 vectors and k=15 
#DataSet = datasets.load_breast_cancer().data
#DataSet = DataSet[:,0:2]
DataSet = datasets.load_iris().data                 #N = 150 puede ser (3)
X, y = datasets.make_moons(n_samples=150, shuffle=True, noise=0.02, random_state=None)
#DataSet = StandardScaler().fit_transform(X)

n_cspa = 3

def index(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()

	if not username:
		test = KMeansAlgorithm(DataSet, {'kmeans':2} )
	else:
		test = KMeansAlgorithm(DataSet, {'kmeans':int(username)} )

	test.run()
	#test.m_resultLabels
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet.data)
	feature = sklearn_pca.fit_transform(std)


	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()
	return render_to_response('index.html',  {'model_list': "holaaaa"})



def kmean_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	if not username:
		test = KMeansAlgorithm(DataSet, {'kmeans':2} )
	else:
		test = KMeansAlgorithm(DataSet, {'kmeans':int(username)} )
	test.run()
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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

	list_matriz_kmean2 = matrix.tolist()

	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-kmeans")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex.tolist()}
	return HttpResponse(json.dumps(data))





def kmedoids_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	if not username:
		test = KMedoidsAlgorithm(DataSet, {'kmeans':2} )
	else:
		test = KMedoidsAlgorithm(DataSet, {'kmeans':int(username)} )
	test.run()
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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

	list_matriz_kmean2 = matrix.tolist()

	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()

	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-kmeans")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))




def kmean_model_clasic(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()
	if not username:
		test = KMeansAlgorithm2(DataSet, {'kmeans':2} )
	else:
		test = KMeansAlgorithm2(DataSet, {'kmeans':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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


	list_matriz_kmean2 = matrix.tolist()




	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()



	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-clasic")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))







def dbscan_model(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()
	if not username:
		test = DBScanAlgorithm(DataSet, {'kmeans':2} )
	else:
		test = DBScanAlgorithm(DataSet, {'kmeans':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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


	list_matriz_kmean2 = matrix.tolist()




	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()



	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-dbscan")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))



def agglomerative_model(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()
	if not username:
		test = AgglomerativeAlgorithm(DataSet, {'AgglomerativeClustering':2} )
	else:
		test = AgglomerativeAlgorithm(DataSet, {'AgglomerativeClustering':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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


	list_matriz_kmean2 = matrix.tolist()




	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()



	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-agglomerative")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))



def MiniBatchKMeans_model(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()
	if not username:
		test = MiniBatchKMeansAlgorithm(DataSet, {'MiniBatchKMeansClustering':2} )
	else:
		test = MiniBatchKMeansAlgorithm(DataSet, {'MiniBatchKMeansClustering':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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


	list_matriz_kmean2 = matrix.tolist()




	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_kmean = h.tolist()



	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-minibacht")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))






def birch_model(request):
	username = request.GET.get('username', None)

	#iris = datasets.load_iris()
	if not username:
		test = BirchAlgorithm(DataSet, {'birch':2} )
	else:
		test = BirchAlgorithm(DataSet, {'birch':int(username)} )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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

	list_matriz_birch2 = matrix.tolist()


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_birch = h.tolist()


	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-birch")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_birch, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3),'list_labels':ex}
	return HttpResponse(json.dumps(data))


def som_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	print(username)
	if not username:
		test = SomAlgorithm( DataSet, {'som_a':1, 'som_b':3} )
	else:
		test = SomAlgorithm( DataSet, {'som_a':1, 'som_b':int(username) } )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
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

	list_matriz_som2 = matrix.tolist()


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0

	list_matriz_som = h.tolist()


	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-som")
	print(val3)
	print(val4)
	print(val5)
	ex = test.m_resultLabels.tolist()
	data = {'val1': totlist_som, 'val2': list_matriz_som, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3),'list_labels':ex }
	return HttpResponse(json.dumps(data))


@csrf_exempt
def ensamble_model(request):
	print("entro ensemble view...")

	username   = request.POST.getlist('username[]')
	points     = request.POST.getlist('points[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)
	##print( int(number_models))

	listPoint = []
	k=0
	for i in range( int(len(points)/2)) :
		tmp = []
		#print(k)
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)


	listWeight_test = Wachspress(listPoint)
	print("true weight:----")
	#print(listWeight_test)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)
	print(tmp2)





	n = int(len(username)/number_models)

	listModel = []

	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(username[j]) )
		listModel.append(tmp)

	#create matrix co-ocurrence for each model
	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				else:
					h[i][j] = 0
		h_models.append(h)


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
				h[i][j] = h[i][j] + h_models[k][i][j]*tmp2[k]

	print("test---------------------------------------------------")
	print(h)
	list_matriz_ensemble = h.tolist()
	#print(list_matriz_ensemble)

	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				continue
			else:
				print(int(round(h[i][j]*10)))
				t = (j, int(round(h[i][j]*10)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)


	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)


	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-ensamble-varios")
	print(val3)
	print(val4)
	print(val5)
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex , 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))


@csrf_exempt
def ensamble_model_initial(request):
	print("entro ensemble view...")

	username   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	print(number_models)
	number_models = int(number_models)
	N = int(number_models)
	##print( int(number_models))





	n = int(len(username)/number_models)

	listModel = []

	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(username[j]) )
		listModel.append(tmp)



	#create matrix co-ocurrence for each model
	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				else:
					h[i][j] = 0
		h_models.append(h)


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
				h[i][j] = h[i][j] + h_models[k][i][j]
			h[i][j] = (h[i][j])/number_models
	print("test---------------------------------------------------------")
	print(h)

	list_matriz_ensemble = h.tolist()
	#print(list_matriz_ensemble.shape())

	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				#print("xxxxxxxxxxxxxxxxxxxxxxxxx")
				continue
			else:
				print(int(round(h[i][j]*10)))
				t = (j, int(round(h[i][j]*10)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)





	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)


	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-initial")
	print(val3)
	print(val4)
	print(val5)
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex}
	return HttpResponse(json.dumps(data))



@csrf_exempt
def ensamble_Agglomerative_initial(request):
	print("entro ensemble Agglomerative")
	username   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	print(number_models)
	number_models = int(number_models)
	N = int(number_models)
	##print( int(number_models))	


	n = int(len(username)/number_models)
	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(username[j]) )
		listModel.append(tmp)


	#create matrix co-ocurrence for each model
	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
					#print(1)
				else:
					h[i][j] = 0
		h_models.append(h)


	h = np.zeros((n, n))
	p = 0
	q = 0
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
					h[i][j] = h[i][j] + h_models[k][i][j]
					print(h[i][j])
			if (h[i][j]/number_models)>0.5:
				h[i][j] = 1
				p = p+1
			else:
				h[i][j] = 0
				q = q+1
				#print(h[i][j])
			#print(2/3)
	
	#print("p y q")
	#print(p)
	#print(q)
	list_matriz_ensemble = h.tolist()


	test1 = EnsembleAgglomerativeAlgorithm(DataSet, {'AgglomerativeClustering':2, 'connectivity':h} )
	test1.run()


	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				continue
			else:
				t = (j, int(round(h[i][j]*1000)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	#cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)


	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)


	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(test1.m_resultLabels).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	#test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	#test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-initial")
	print(val3)
	print(val4)
	print(val5)
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex}
	return HttpResponse(json.dumps(data))





@csrf_exempt
def ensamble_Agglomerative(request):
	print("Agglomerative......................")

	list_model   = request.POST.getlist('username[]')
	points     = request.POST.getlist('points[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)
	##print( int(number_models))

	listPoint = []
	k=0
	for i in range( int(len(points)/2)) :
		tmp = []
		#print(k)
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)


	listWeight_test = Wachspress(listPoint)
	print("true weight:----")
	print(listWeight_test)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)
	print(tmp2)





	n = int(len(list_model)/number_models)

	listModel = []

	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(list_model[j]) )
		listModel.append(tmp)

	#create matrix co-ocurrence for each model
	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				else:
					h[i][j] = 0
		h_models.append(h)


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
				h[i][j] = h[i][j] + h_models[k][i][j]*tmp2[k]

	print("test---------------------------------------------------")
	print(h)
	list_matriz_ensemble = h.tolist()
	print(list_matriz_ensemble)

	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				continue
			else:
				t = (j, int(round(h[i][j]*100)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)


	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)


	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	print("list of metrics-ensamble-varios")
	print(val3)
	print(val4)
	print(val5)
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex , 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))






@csrf_exempt
def majority_vote(request):
	print("voting polygon")
	list_models_input   = request.POST.getlist('username[]')
	points     = request.POST.getlist('points[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	listMetric = request.POST.getlist('metrics[]')
	listPoint = []

	n = int(len(list_models_input)/number_models)

	listPoint = []
	k=0
	for i in range( int(len(points)/2)) :
		tmp = []
		#print(k)
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)

	listWeight_test = Wachspress(listPoint)
	print("true weight:----")
	print(points)
	print(listWeight_test)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)
	print(tmp2)









	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(list_models_input[j]) )
		listModel.append(tmp)


	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				else:
					h[i][j] = 0
		h_models.append(h)


	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
				h[i][j] = h[i][j] + h_models[k][i][j]*tmp2[k]

	list_matriz_kmean = h.tolist()
	print(h)

	labels = np.zeros((n), dtype=int)
	threshold = 0.51
	currCluster = 1
	for i in range(n):
		for j in range(i+1, n):
			if h[i, j] >= threshold:
				if labels[i] and labels[j]:
					cluster_num = min(labels[i] , labels[j])
					cluster_toChange = max(labels[i] , labels[j])

					indices = [k for k, x in enumerate(labels) if x == cluster_toChange]
					labels[indices] = cluster_num

				elif not labels[i] and not labels[j]: #a new cluster
					labels[i] = currCluster
					labels[j] = currCluster
					currCluster += 1

				else: #one of them is in a cluster and one is not, one will be assigned the same thing, but saves an if
					cluster_num = max(labels[i] , labels[j])
					labels[i] = cluster_num
					labels[j] = cluster_num

			else: #else don't join them and give them an assignment the first time they are traversed
				if not labels[i]:
					labels[i] = currCluster
					currCluster += 1
				if not labels[j]:
					labels[j] = currCluster
					currCluster += 1

	clusters = np.sort(np.unique(labels))
	for ind in range(0,len(clusters)-1): #operating on ind+1
		if clusters[ind+1] != clusters[ind]+1:
			cluster_num = clusters[ind] + 1
			#print("updating cluster num %d to %d")%(clusters[ind+1], cluster_num)
			indices = [k for k, x in enumerate(labels) if x == clusters[ind+1]]
			labels[indices] = cluster_num
			clusters[ind+1] = cluster_num 

	final_label = []
	s = set(labels)
	min_value = min(labels)
	for i in range(n):
		final_label.append( labels[i] -min_value)
	print(final_label)

	tolist_cspa = []
	ex   = 0

	test1 = Majority_voteAlgorithm(DataSet, {'Majority_voteAlgorithm':final_label} )
	print("length.....")
	print(len(s))
	if len(s)>1:
		test1.run()
		val3 = test1.m_resultMetrics['silhouette_score']
		val4 = test1.m_resultMetrics['Sum_Squared_Within']
		val5 = test1.m_resultMetrics['Sum_Squared_Between']
	else:
		val3 = 1
		val4 = 1
		val5 = 1

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)

	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(final_label).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()

	
	print("list of metrics")
	print(val3)
	print(val4)
	print(val5)
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex, 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))



@csrf_exempt
def majority_vote_initial(request):
	list_model_input   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	n = int(len(list_model_input)/number_models)

	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(list_model_input[j]) )
		listModel.append(tmp)

	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				else:
					h[i][j] = 0
		h_models.append(h)

	h = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
				h[i][j] = h[i][j] + h_models[k][i][j]
			h[i][j] = (h[i][j])/number_models

	list_matriz_voting = h.tolist()

	labels = np.zeros((n), dtype=int)
	threshold = 0.5
	currCluster = 0
	for i in range(0, n):
		for j in range(i+1, n):
			if h[i, j] > threshold:
				if labels[i] and labels[j]:
					cluster_num = min(labels[i] , labels[j])
					cluster_toChange = max(labels[i] , labels[j])

					indices = [k for k, x in enumerate(labels) if x == cluster_toChange]
					labels[indices] = cluster_num

				elif not labels[i] and not labels[j]: #a new cluster
					labels[i] = currCluster
					labels[j] = currCluster
					currCluster += 1

				else: #one of them is in a cluster and one is not, one will be assigned the same thing, but saves an if
					cluster_num = max(labels[i] , labels[j])
					labels[i] = cluster_num
					labels[j] = cluster_num
					print(currCluster)
					print("final test")
			else: #else don't join them and give them an assignment the first time they are traversed
				if not labels[i]:
					labels[i] = currCluster
					currCluster += 1
				if not labels[j]:
					labels[j] = currCluster
					currCluster += 1

	clusters = np.sort(np.unique(labels))
	for ind in range(0,len(clusters)-1): #operating on ind+1
		if clusters[ind+1] != clusters[ind]+1:
			cluster_num = clusters[ind] + 1
			#print("updating cluster num %d to %d")%(clusters[ind+1], cluster_num)
			indices = [k for k, x in enumerate(labels) if x == clusters[ind+1]]
			labels[indices] = cluster_num
			clusters[ind+1] = cluster_num 

	final_label = []
	s = set(labels)
	min_value = min(labels)
	for i in range(n):
		final_label.append(labels[i]-min_value)

	print(labels)
	tolist_cspa = []
	ex   = 0

	test1 = Majority_voteAlgorithm(DataSet, {'Majority_voteAlgorithm':final_label} )
	if len(s)>1:
		test1.run()
		val3 = test1.m_resultMetrics['silhouette_score']
		val4 = test1.m_resultMetrics['Sum_Squared_Within']
		val5 = test1.m_resultMetrics['Sum_Squared_Between']
	else:
		val3 = 1
		val4 = 1
		val5 = 1
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)

	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(final_label).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_voting = matrix_general.tolist()

	
	print("list of metrics")
	print(val3)
	print(val4)
	print(val5)
	data = {'val1': tolist_voting, 'val2': list_matriz_voting, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex}
	return HttpResponse(json.dumps(data))