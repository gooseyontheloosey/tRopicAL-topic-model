# topicModeller/apps.py
from django.apps import AppConfig

class topicModellerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'topicModeller'
    verbose_name = 'Topic Modeller'
    
    def ready(self):
        pass