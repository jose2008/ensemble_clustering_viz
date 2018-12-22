"""django_app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from clustering_viz import views 
from clustering_viz import models
#from . import views 
urlpatterns = [
	path('clustering_viz/', include('clustering_viz.urls')),
    path('admin/', admin.site.urls),
	#path('', views.index, name='index'),
	path('models/model_clustering', models.model_clustering, name='model_clustering'),
	path('views/kmean_model', views.kmean_model, name='kmean_model'),
	path('views/birch_model', views.birch_model, name='birch_model'),
	path('views/som_model', views.birch_model, name='birch_model'),
	path('views/ensamble_model', views.ensamble_model, name='ensamble_model'),
]

#path('polls/', include('polls.urls')),