# topicModeller/models.py
from django.db import models
from django.contrib.auth.models import User



class Dataset(models.Model):
    name = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
    location = models.CharField(max_length=500)
    uploadedFile = models.FileField(upload_to='datasets/', null=True, blank=True)
    isPreprocessed = models.BooleanField(default=False)
    vocabSize = models.IntegerField(null=True, blank=True)
    numBatches = models.IntegerField(null=True, blank=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    runName = models.CharField(max_length=255)
    hiddenDim1Size = models.IntegerField()
    hiddenDim2Size = models.IntegerField()
    latentDimSize = models.IntegerField()
    numEpochs = models.IntegerField()
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    # New field to mark models as trained for visualisation
    isMarkedTrained = models.BooleanField(default=False)
    modelPath = models.CharField(max_length=500, null=True, blank=True)
    finalLoss = models.FloatField(null=True, blank=True)
    coherenceScore = models.FloatField(null=True, blank=True)
    diversityScore = models.FloatField(null=True, blank=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    completedAt = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f'{self.runName} - {self.dataset.name}'
    
    @property
    def canBeMarkedTrained(self):
        '''Check if this training run can be marked as trained'''
        return self.status == 'completed' and self.modelPath


class ValidationResult(models.Model):
    trainingRun = models.ForeignKey(TrainingRun, on_delete=models.CASCADE)
    validationLoss = models.FloatField()
    reconstructionLoss = models.FloatField()
    klLoss = models.FloatField()
    coherenceScore = models.FloatField(null=True, blank=True)
    diversityScore = models.FloatField(null=True, blank=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    

class TestResult(models.Model):
    trainingRun = models.ForeignKey(TrainingRun, on_delete=models.CASCADE)
    testLoss = models.FloatField()
    reconstructionLoss = models.FloatField()
    klLoss = models.FloatField()
    coherenceScore = models.FloatField(null=True, blank=True)
    diversityScore = models.FloatField(null=True, blank=True)
    createdAt = models.DateTimeField(auto_now_add=True)


class InferenceResult(models.Model):
    '''New model to track visualisation/inference results'''
    trainingRun = models.ForeignKey(TrainingRun, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    resultsPath = models.CharField(max_length=500, null=True, blank=True)
    totalLoss = models.FloatField(null=True, blank=True)
    coherenceScore = models.FloatField(null=True, blank=True)
    diversityScore = models.FloatField(null=True, blank=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    completedAt = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f'Inference: {self.trainingRun.runName} on {self.dataset.name}'