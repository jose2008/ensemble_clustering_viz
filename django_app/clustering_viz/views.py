from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.shortcuts import render
from clustering_viz.clustering.core import KMeansAlgorithm

from sklearn import datasets 
from clustering_viz.clustering.core.KMeansAlgorithm import *
from clustering_viz.clustering.core.SpectralAlgorithm import *
from clustering_viz.clustering.core.GmmAlgorithm import *
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
from clustering_viz.clustering.core.lamp import *
from django.views.decorators.csrf import csrf_exempt
from sklearn.manifold import TSNE, MDS

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from django.template import RequestContext
from sklearn.datasets import load_digits


from django.template.response import TemplateResponse 
from django.shortcuts import render_to_response
import metis
import time

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




from pandas import DataFrame, read_excel
import xlrd

'''
###################   DATASETS FROM UNIVERSITY SAN DIEGO   ##############################
'''
#seeds_dataset



#DataSet = load_dataset('seeds_dataset.txt') #



# buddymove_holidayiq.csv attributo 6 -n=249 , k=6
#df = pd.read_csv('buddymove_holidayiq.csv')
#df = df.drop('User Id', axis=1)
#DataSet = df.as_matrix()



# Wholesale_customers_data.csv attributo 6 -n=440 , D=7
#df = pd.read_csv('Wholesale_customers_data.csv')
#df = df.drop('Channel', axis=1)
#df = df.drop('Region', axis=1)
#DataSet = df.as_matrix()

#google_review_ratings.csv attributo 6 -n=249
#df = pd.read_csv('google_review_ratings.csv')
#df = df.drop('User', axis=1)
#DataSet = df.as_matrix()





# User_Knowledge_Modeling.csv attributo 5 -n=260, k=4
#my_sheet = 'Training_Data'
#file_name = 'User_Knowledge_Modeling.xls' # name of your excel file
#df = read_excel(file_name, sheet_name = my_sheet)
#df = df.drop(' UNS', axis=1)
#DataSet = df.as_matrix()




#print(DataSet)


'''
###################   DATASETS FROM SKLEARN   ##############################
'''

#circlesss
#X, Y = datasets.make_circles(n_samples=300, factor=.5, noise=.05)
#DataSet = StandardScaler().fit_transform(X)  #k=2


#moons
#X, y = datasets.make_moons(n_samples=150, shuffle=True, noise=0.02, random_state=None)
#DataSet = StandardScaler().fit_transform(X)  #k=2




#blobs
#X, y = datasets.make_blobs(n_samples=300, random_state=170)
#DataSet = StandardScaler().fit_transform(X)



#blobs with variance
#X, Y = datasets.make_blobs(n_samples=300, cluster_std=[1.0, 2.5, 0.5], random_state=170)
#DataSet = StandardScaler().fit_transform(X)


#aniso
#X, y = datasets.make_blobs(n_samples=300, random_state=170)
#transformation = [[0.6, -0.6], [-0.4, 0.8]]
#X_aniso = np.dot(X, transformation)
#X, y = (X_aniso, y)
#DataSet = StandardScaler().fit_transform(X)




'''
##############################  Clustering basic benchmark ##################################
'''
#DataSet = load_dataset('flame.txt')[:,[0,1]]		#N =240 k=2 , D=2


#DataSet = load_dataset('spiral.txt')[:,[0,1]]		#N=312, k=3, D=2 


#DataSet = load_dataset('wine.txt')					#N=178, k=3, D=13-----test((de hecho))(2).......................(3 tesis)


#DataSet = load_dataset('jain.txt')[:,[0,1]]			#N=373, k=2, D=2----test (4)(costoso)


#DataSet = load_dataset('unbalance.txt')				#N=6500(costoso)



#DataSet = load_dataset('aggregation.txt')[:,[0,1]]	#N=788 k=7 d=2(si)


#DataSet = load_dataset('thyroid.txt')	#N=215, k=2, D=5



#DataSet = datasets.load_iris().data                 #N = 150 puede ser (3)...........................................(4 tesis)




'''
##############################  Clustering from istar##################################
'''

#dermatology(si)
#DataSet = pd.read_csv('dermatology.txt', sep=";", header=None)
#DataSet = DataSet.as_matrix()	#n=358 D=34 k=6
#DataSet = np.delete(DataSet, -1, axis=1)
#DataSet = DataSet[:,1:]


#twonorm.txt(bien)
DataSet = pd.read_csv('twonorm.txt', sep=";", header=None)
DataSet = DataSet.as_matrix()	#n=1009 D=20 k=2
DataSet = np.delete(DataSet, -1, axis=1)
DataSet = DataSet[:,1:]


#VEHICLE(bien)
#DataSet = pd.read_csv('VEHICLE.txt', sep=";", header=None)
#DataSet = DataSet.as_matrix()	#n=846 D=18  k=4
#DataSet = np.delete(DataSet, -1, axis=1)
#DataSet = DataSet[:,1:]


#fiber-notnorm(mal)
#DataSet = pd.read_csv('fiber-notnorm.txt', sep=";", header=None)
#DataSet = DataSet.as_matrix()	#n=901 D=30 k=9
#DataSet = np.delete(DataSet, -1, axis=1)
#DataSet = DataSet[:,1:]


#ionosphere(malo)
#DataSet = pd.read_csv('ionosphere.txt', sep=";", header=None)
#DataSet = DataSet.as_matrix()	#n=351 D=33 k=10
#DataSet = np.delete(DataSet, -1, axis=1)
#DataSet = DataSet[:,1:]







'''
##############################  Clustering from istar##################################
'''
#case_study(si)
#DataSet = pd.read_csv('case_study.txt', sep=";", header=None)
#DataSet = DataSet.as_matrix()	#n=184 D=2 k=2
#DataSet = np.delete(DataSet, -1, axis=1)
#DataSet = DataSet[:,1:]

n_cspa = 3

def index(request):
	username = request.GET.get('username', None)
	'''
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
	'''
	return render_to_response('index.html',  {'model_list': "holaaaa"})



def kmean_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	print("datasets--------------")
	print(DataSet)
	if not username:
		test = KMeansAlgorithm(DataSet, {'kmeans':2} )
	else:
		test = KMeansAlgorithm(DataSet, {'kmeans':int(username)} )
	test.run()
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0,n_iter=20000).fit_transform(DataSet)



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
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex.tolist(),"name":'K-means'}
	return HttpResponse(json.dumps(data))




def spectral_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	if not username:
		test = SpectralAlgorithm(DataSet, {'spectral':2} )
	else:
		test = SpectralAlgorithm(DataSet, {'spectral':int(username)} )
	test.run()
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0,n_iter=20000).fit_transform(DataSet)



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
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex.tolist(),"name":'spectral'}
	return HttpResponse(json.dumps(data))



def gmm_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	if not username:
		test = GmmAlgorithm(DataSet, {'gmm':2} )
	else:
		test = GmmAlgorithm(DataSet, {'gmm':int(username)} )
	test.run()
	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0,n_iter=20000).fit_transform(DataSet)



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
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex.tolist(),"name":'gmm'}
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
	#
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex, 'name':"Kmedoids"}
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex, 'name':"Kmeans-clasic"}
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(test.m_resultLabels).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()
	n = len(test.m_resultLabels)
	print("labels")
	print(test.m_resultLabels)
	'''list_set = []
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



	'''
	h = np.zeros((n, n))
	'''for i in range(n):
		for j in range(n):
			if test.m_resultLabels[i] == test.m_resultLabels[j]:
				h[i][j] = 1
			else:
				h[i][j] = 0
	'''
	list_matriz_kmean = h.tolist()



	val3 = test.m_resultMetrics['silhouette_score']
	val4 = test.m_resultMetrics['Sum_Squared_Within']
	val5 = test.m_resultMetrics['Sum_Squared_Between']
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex, 'name':"DBScan"}
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex, "name":'Agglomerative'}
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':ex, "name":"MiniBatch"}
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels.tolist()
	data = {'val1': tolist_knn, 'val2': list_matriz_birch, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3),'list_labels':ex, "name":'Birch'}
	return HttpResponse(json.dumps(data))


def som_model(request):
	username = request.GET.get('username', None)
	#iris = datasets.load_iris()
	if not username:
		test = SomAlgorithm( DataSet, {'som_a':1, 'som_b':3} )
	else:
		test = SomAlgorithm( DataSet, {'som_a':1, 'som_b':int(username) } )
	test.run()

	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

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
	ex = test.m_resultLabels.tolist()
	data = {'val1': totlist_som, 'val2': list_matriz_som, 'val3': trun_n_d(val3,3), 'val4': trun_n_d(val4,3) ,'val5':trun_n_d(val5,3),'list_labels':ex, "name":"SOM" }
	return HttpResponse(json.dumps(data))


@csrf_exempt
def ensamble_model(request):

	username   = request.POST.getlist('username[]')
	points     = request.POST.getlist('points[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)

	listPoint = []
	k=0
	for i in range( int(len(points)/2)) :
		tmp = []
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)


	listWeight_test = Wachspress(listPoint)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)





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

	list_matriz_ensemble = h.tolist()

	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				continue
			else:
				t = (j, int(round(h[i][j]*10)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)


	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex , 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))



@csrf_exempt
def ensamble_model_test(request):
	input_models_list   = request.POST.getlist('username[]')
	points_vertex   = request.POST.getlist('vertex[]')
	points_large     = request.POST.getlist('points[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	listPoint = []

	n = int(len(input_models_list)/number_models)
	list_of_metrics = []
	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(input_models_list[j]) )
		listModel.append(tmp)

	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
		h_models.append(h)
	k = 0
	listPoint = []
	for i in range( int(len(points_vertex)/2)) :
		tmp = []
		tmp.append( float(points_vertex[k]))
		tmp.append( float(points_vertex[k+1]))
		k = k+2
		listPoint.append(tmp)

	for count in range(0,int(len(points_large)),2):
		listWeight_test = []
		listPoint.append( [ float(points_large[count]) , float(points_large[count+1]) ]  )
		listWeight_test = Wachspress(listPoint)

		listWeight_test = Wachspress(listPoint)
		total_weight = 0
		for i in range(len(listWeight_test)):
			total_weight = total_weight + listWeight_test[i]
		tmp2 = []
		for i in range(len(listWeight_test)):
			tmp2.append(listWeight_test[i]/total_weight)

		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				h[i][j] = 0
				for k in range(number_models):
					h[i][j] = h[i][j] + h_models[k][i][j]*tmp2[k]

		list_matriz_ensemble = h.tolist()

		list_ad = []
		for i in range(n):
			list_tmp = []
			for j in range(n):
				if(h[i][j] == 0):
					continue
				else:
					t = (j, int(round(h[i][j]*10)))
					list_tmp.append(t)
			list_ad.append(list_tmp)

		cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)

		test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
		test1.run()

		val3 = test1.m_resultMetrics['silhouette_score']
		val4 = test1.m_resultMetrics['Sum_Squared_Within']
		val5 = test1.m_resultMetrics['Sum_Squared_Between']

		list_of_metrics.append([val3, val4, val5])
		data = {'list_of_metrics':list_of_metrics}
		listPoint.pop()
		print("print metrics")
		print(data)
	return HttpResponse(json.dumps(data))



@csrf_exempt
def ensamble_model_initial(request):
	username   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	list_metrics = request.POST.getlist('list_metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)



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

	list_matriz_ensemble = h.tolist()

	list_ad = []
	for i in range(n):
		list_tmp = []
		for j in range(n):
			if(h[i][j] == 0):
				continue
			else:
				t = (j, int(round(h[i][j]*10)))
				list_tmp.append(t)
		list_ad.append(list_tmp)

	cuts, parts = metis.part_graph(list_ad, n_cspa, recursive = False, dbglvl=metis.METIS_DBG_ALL)





	sklearn_pca = sklearnPCA(n_components=2)
	std = StandardScaler().fit_transform(DataSet)
	feature = sklearn_pca.fit_transform(std)
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	#ex = test.m_resultLabels.tolist()
	ex = []


	data_metrics = []
	#data_metrics.append( [1,1,1] )
	#data_metrics.append( [0.5,0.5,0.5] )
	#data_metrics.append( [0,0,0] )
	array_metric1 = []
	array_metric2 = []
	array_metric3 = []
	for k in range(0,len(list_metrics),3):
		data_metrics.append( [ float(list_metrics[k]), float(list_metrics[k+1]), float(list_metrics[k+2])] )	
		array_metric1.append(float(list_metrics[k]))
		array_metric2.append(float(list_metrics[k+1]))
		array_metric3.append(float(list_metrics[k+2]))


	#data_metrics.append( [ max(array_metric1), max(array_metric2), max(array_metric3) ] )
	#data_metrics.append( [ (float(max(array_metric1)) + float(min(array_metric1)))/2, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2])  
	#data_metrics.append( [ min(array_metric1), min(array_metric2), min(array_metric3) ] )
	data_metrics.append([1,1,1])
	data_metrics.append([0.5+0.2,0.5-0.1,0.5])
	data_metrics.append([0,0,0])



	#control_point = [ [ max(array_metric1), max(array_metric2), max(array_metric3) ],
	#[ (float(max(array_metric1)) + float(min(array_metric1)))/2, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2],
	#[ min(array_metric1), min(array_metric2), min(array_metric3) ] ]

	control_point = [ [1,1,1],  [0.5 +0.2,0.5-0.1,0.5], [0,0,0] ]




	data_metrics = np.array(data_metrics)
	control_point = np.array(control_point)
	id = np.array([len(data_metrics) - 3  ,  len(data_metrics) - 2 , len(data_metrics) - 1])
	control_point = np.hstack((control_point, id.reshape(3,1) ))
	lamp_proj = Lamp(Xdata = data_metrics, control_points = control_point, label=False)
	metrics_reductions = lamp_proj.fit()
	metrics_reductions = metrics_reductions.tolist()
	for k in range(len( metrics_reductions ) ):
		metrics_reductions[k].append(k)
	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex, 'list_metrics': metrics_reductions}
	return HttpResponse(json.dumps(data))



@csrf_exempt
def ensamble_Agglomerative_initial(request):
	username   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)


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
	p = 0
	q = 0
	for i in range(n):
		for j in range(n):
			h[i][j] = 0
			for k in range(number_models):
					h[i][j] = h[i][j] + h_models[k][i][j]
			if (h[i][j]/number_models)>0.5:
				h[i][j] = 1
				p = p+1
			else:
				h[i][j] = 0
				q = q+1
	
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
	#feature = sklearn_pca.fit_transform(std)
	feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(test1.m_resultLabels).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	#test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	#test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex}
	return HttpResponse(json.dumps(data))





@csrf_exempt
def ensamble_Agglomerative(request):

	list_model   = request.POST.getlist('username[]')
	points     = request.POST.getlist('points[]')
	listMetric = request.POST.getlist('metrics[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	N = int(number_models)

	listPoint = []
	k=0
	for i in range( int(len(points)/2)) :
		tmp = []
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)


	listWeight_test = Wachspress(listPoint)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)





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

	list_matriz_ensemble = h.tolist()

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
	#feature = sklearn_pca.fit_transform(std)
	feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix__feature_cspa = np.matrix(feature)

	matrix_label_cspa = np.matrix(parts).transpose()
	matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
	tolist_cspa = matrix_general_cspa.tolist()


	test1 = EnsembleAlgorithm(DataSet, {'EnsembleAlgorithm':parts} )
	test1.run()


	val3 = test1.m_resultMetrics['silhouette_score']
	val4 = test1.m_resultMetrics['Sum_Squared_Within']
	val5 = test1.m_resultMetrics['Sum_Squared_Between']
	#ex = test.m_resultLabels.tolist()
	ex = []

	data = {'val1': tolist_cspa, 'val2': list_matriz_ensemble, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex , 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))






@csrf_exempt
def majority_vote(request):
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
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)

	listWeight_test = Wachspress(listPoint)
	total_weight = 0
	for i in range(len(listWeight_test)):
		total_weight = total_weight + listWeight_test[i]
	tmp2 = []
	for i in range(len(listWeight_test)):
		tmp2.append(listWeight_test[i]/total_weight)


	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(list_models_input[j]) )
		listModel.append(tmp)

	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		#h[np.where(listModel[k][i] == listModel[k][j]),1, 0]
		#np.where(listModel[k][i] == listModel[k][j], 1, 0)
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
				#else:
				#	h[i][j] = 0
		h_models.append(h)

	for k in range(number_models):
		h_models[k] = np.multiply( h_models[k], tmp2[k])

	h = np.zeros((n, n))
	for k in range(number_models):
		h = h + h_models[k]


	list_matriz_kmean = h.tolist()
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
			indices = [k for k, x in enumerate(labels) if x == clusters[ind+1]]
			labels[indices] = cluster_num
			clusters[ind+1] = cluster_num 

	final_label = []
	s = set(labels)
	min_value = min(labels)
	for i in range(n):
		final_label.append( labels[i] -min_value)

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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(final_label).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_knn = matrix_general.tolist()

	data = {'val1': tolist_knn, 'val2': list_matriz_kmean, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex, 'listWeight':tmp2}
	return HttpResponse(json.dumps(data))




@csrf_exempt
def majority_vote_test(request):
	list_models_input   = request.POST.getlist('username[]')
	list_models_input2   = request.POST.getlist('vertex[]')
	points_large     = request.POST.getlist('points[]')
	number_models = request.POST.get('numbers_model', None)
	number_models = int(number_models)
	
	listPoint = []

	n = int(len(list_models_input)/number_models)
	
	list_of_metrics = []
	points = list_models_input2
	#points.pop()
	#points.pop()

	listModel = []
	for i in range(number_models):
		tmp = []
		for j in range(  int(i*n), int(n+i*n)):
			tmp.append( int(list_models_input[j]) )
		listModel.append(tmp)

	#list_models_input = np.array(list_models_input)
	#for i in range(number_models):
	#	listModel.append(list_models_input[int(i*n): int(n+i*n)].tolist())


	h_models = []
	for k in range(number_models):
		h = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if listModel[k][i] == listModel[k][j]:
					h[i][j] = 1
		h_models.append(h)
	
	k = 0
	listPoint = []
	for i in range( int(len(points)/2)) :
		tmp = []
		tmp.append( float(points[k]))
		tmp.append( float(points[k+1]))
		k = k+2
		listPoint.append(tmp)
	start = time.time()
	for count in range(0,int(len(points_large)),2):
		print(count)
		listWeight_test = []
		'''k=0
		for i in range( int(len(points)/2)) :
			tmp = []
			tmp.append( float(points[k]))
			tmp.append( float(points[k+1]))
			k = k+2
			listPoint.append(tmp)
		'''
		listPoint.append( [ float(points_large[count]) , float(points_large[count+1]) ]  )
		listWeight_test = Wachspress(listPoint)
		total_weight = 0
		for i in range(len(listWeight_test)):
			total_weight = total_weight + listWeight_test[i]
		tmp2 = []
		for i in range(len(listWeight_test)):
			tmp2.append(listWeight_test[i]/total_weight)

		h_models_weighted = []
		for k in range(number_models):
			h_models_weighted.append(np.multiply( h_models[k], tmp2[k]))

		h = np.zeros((n, n))
		for k in range(number_models):
			h = h + h_models_weighted[k]
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
				indices = [k for k, x in enumerate(labels) if x == clusters[ind+1]]
				labels[indices] = cluster_num
				clusters[ind+1] = cluster_num 

		final_label = []
		s = set(labels)
		min_value = min(labels)
		for i in range(n):
			final_label.append( labels[i] -min_value)

		tolist_cspa = []
		ex   = 0
		test1 = Majority_voteAlgorithm( DataSet, {'Majority_voteAlgorithm':final_label} )
		if len(s)>1:
			test1.run()
			val3 = test1.m_resultMetrics['silhouette_score']
			val4 = test1.m_resultMetrics['Sum_Squared_Within']
			val5 = test1.m_resultMetrics['Sum_Squared_Between']
		else:
			val3 = 1
			val4 = 1
			val5 = 1

		list_of_metrics.append([val3, val4, val5])
		#points.pop()
		#points.pop()
		listPoint.pop()
	
	print("time")
	end = time.time()
	print(end - start)

	data = {'list_of_metrics':list_of_metrics}
	#print(list_of_metrics)
	return HttpResponse(json.dumps(data))




@csrf_exempt
def majority_vote_initial(request):
	list_model_input   = request.POST.getlist('username[]')
	listMetric = request.POST.getlist('metrics[]')
	list_metrics = request.POST.getlist('list_metrics[]')
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
	#feature = TSNE(n_components=2, init='random',random_state=0).fit_transform(DataSet)

	matrix_feature_kmean = np.matrix(feature)
	matrix_label_knn = np.matrix(final_label).transpose()

	matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
	tolist_voting = matrix_general.tolist()


	data_metrics = []
	#data_metrics.append( [1,1,1] )
	#data_metrics.append( [0.5,0.5,0.5] )
	#data_metrics.append( [0,0,0] )
	array_metric1 = []
	array_metric2 = []
	array_metric3 = []
	for k in range(0,len(list_metrics),3):
		data_metrics.append( [ float(list_metrics[k]), float(list_metrics[k+1]), float(list_metrics[k+2])] )	
		array_metric1.append(float(list_metrics[k]))
		array_metric2.append(float(list_metrics[k+1]))
		array_metric3.append(float(list_metrics[k+2]))


	#data_metrics.append( [ max(array_metric1), max(array_metric2), max(array_metric3) ] )
	#data_metrics.append( [ (float(max(array_metric1)) + float(min(array_metric1)))/2 + 0.02, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2])  
	#data_metrics.append( [ min(array_metric1), min(array_metric2), min(array_metric3) ] )

	data_metrics.append([1,1,1])
	data_metrics.append([0.5 +0.2,0.5 -0.1,0.5])
	data_metrics.append([0,0,0])


	#control_point = [ [ max(array_metric1), max(array_metric2), max(array_metric3) ],
	#[ (float(max(array_metric1)) + float(min(array_metric1)))/2 + 0.02, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2],
	#[ min(array_metric1), min(array_metric2), min(array_metric3) ] ]

	control_point = [ [1,1,1],  [0.5+0.2 ,0.5-0.1,0.5], [0,0,0] ]


	'''
	#metrics_reductions = TSNE(n_components=2, init='random',random_state=0).fit_transform(data_metrics);
	std = StandardScaler().fit_transform( data_metrics)
	metrics_reductions = sklearn_pca.fit_transform(std)

	print("return of metrics------")
	print(metrics_reductions)
	metrics_reductions = metrics_reductions.tolist()
	for k in range(len( metrics_reductions ) ):
		metrics_reductions[k].append(k)
		#metrics_reductions[k].concatenate(metrics_reductions[k] ,k)
	'''

	
	data_metrics = np.array(data_metrics)
	control_point = np.array(control_point)
	id = np.array([len(data_metrics) - 3  ,  len(data_metrics) - 2 , len(data_metrics) - 1])
	control_point = np.hstack((control_point, id.reshape(3,1) ))
	lamp_proj = Lamp(Xdata = data_metrics, control_points = control_point, label=False)
	metrics_reductions = lamp_proj.fit()
	metrics_reductions = metrics_reductions.tolist()
	for k in range(len( metrics_reductions ) ):
		metrics_reductions[k].append(k)
		#metrics_reductions[k].concatenate(metrics_reductions[k] ,k)
	print("control_point.....")
	print(metrics_reductions)
	data = {'val1': tolist_voting, 'val2': list_matriz_voting, 'val3':trun_n_d( val3, 3), 'val4': trun_n_d(val4 ,3),'val5':trun_n_d (val5,3),'list_labels':0 ,'list_labels':ex,'list_metrics': metrics_reductions}
	return HttpResponse(json.dumps(data))



@csrf_exempt
def dimensionalReduction(request):
	print("reductionnnnnnnnnnnnnnnnnnnn")
	list_metrics = request.POST.getlist('list_metrics[]')
	print(list_metrics)
	data_metrics = []
	array_metric1 = []
	array_metric2 = []
	array_metric3 = []
	for k in range(0,len(list_metrics),3):
		data_metrics.append( [ float(list_metrics[k]), float(list_metrics[k+1]), float(list_metrics[k+2])] )	
		array_metric1.append(float(list_metrics[k]))
		array_metric2.append(float(list_metrics[k+1]))
		array_metric3.append(float(list_metrics[k+2]))

	#data_metrics.append( [ max(array_metric1), max(array_metric2), max(array_metric3) ] )
	#data_metrics.append( [ (float(max(array_metric1)) + float(min(array_metric1)))/2 + 0.02, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2])  
	#data_metrics.append( [ min(array_metric1), min(array_metric2), min(array_metric3) ] )

	data_metrics.append([1,1,1])
	data_metrics.append([0.5+0.2,0.5-0.1,0.5])
	data_metrics.append([0,0,0])


	#control_point = [ [ max(array_metric1), max(array_metric2), max(array_metric3) ],
	#[ (float(max(array_metric1)) + float(min(array_metric1)))/2 + 0.02, (float(max(array_metric2)) + float(min(array_metric2)))/2, (float(max(array_metric3)) + float(min(array_metric3)))/2],
	#[ min(array_metric1), min(array_metric2), min(array_metric3) ] ]
	
	control_point = [ [1,1,1],  [0.5 +0.2,0.5-0.1,0.5], [0,0,0] ]

	data_metrics = np.array(data_metrics)
	control_point = np.array(control_point)
	id = np.array([len(data_metrics) - 3  ,  len(data_metrics) - 2 , len(data_metrics) - 1])
	control_point = np.hstack((control_point, id.reshape(3,1) ))
	lamp_proj = Lamp(Xdata = data_metrics, control_points = control_point, label=False)
	metrics_reductions = lamp_proj.fit()
	metrics_reductions = metrics_reductions.tolist()
	for k in range(len( metrics_reductions ) ):
		metrics_reductions[k].append(k)
		#metrics_reductions[k].concatenate(metrics_reductions[k] ,k)
	print("control_point.....")
	print(metrics_reductions)

	data = {'list_metrics': metrics_reductions}
	return HttpResponse(json.dumps(data))
