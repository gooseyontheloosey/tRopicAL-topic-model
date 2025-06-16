# topicModeller/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Data management
    path('parquetsplitter/', views.PARQUETSplitter, name='PARQUETSplitter'),
    path('datasets/', views.datasetList, name='datasetList'),
    path('datasets/upload/', views.datasetUpload, name='datasetUpload'),
    path('datasets/<int:pk>/', views.datasetDetail, name='datasetDetail'),
    path('datasets/<int:pk>/preprocess/', views.preprocessDataset, name='preprocessDataset'),
    
    # Training
    path('training/new/', views.trainingNew, name='trainingNew'),
    path('training/', views.trainingList, name='trainingList'),
    path('training/<int:pk>/', views.trainingDetail, name='trainingDetail'),
    
    # Model operations
    path('models/', views.viewModels, name='viewModels'),
    path('models/load/', views.enhancedModelLoad, name='enhancedModelLoad'),
    path('training/<int:pk>/validate/', views.validateModel, name='validateModel'),
    path('training/<int:pk>/test/', views.testModel, name='testModel'),
    
    # Topic visualisation
    path('visualise/', views.visualiseTopics, name='visualiseTopics'),
    path('inference/<int:pk>/', views.inferenceDetail, name='inferenceDetail'),
    
    # API endpoints
    path('api/training/<int:pk>/status/', views.getTrainingStatus, name='trainingStatusApi'),
    path('api/inference/<int:pk>/status/', views.getInferenceStatus, name='inferenceStatusApi'),
]