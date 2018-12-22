from django.urls import path
from . import views, models


urlpatterns = [
    path('', views.index, name='index'),
    path('/clustering_viz/models/model_clustering', models.model_clustering, name='model_clustering'),
    #path('/clustering_viz/views/kmean_model', views.kmean_model, name='kmean_model'),
    #path('/clustering_viz/views/kmean_model', views.kmean_model, name='kmean_model'),
    path('', views.kmean_model, name='kmean_model'),
]