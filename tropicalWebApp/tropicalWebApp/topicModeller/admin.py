# topicModeller/admin.py
from django.contrib import admin
from .models import Dataset, TrainingRun, ValidationResult, TestResult

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'isPreprocessed', 'vocabSize', 'numBatches', 'createdAt')
    list_filter = ('isPreprocessed', 'createdAt')
    search_fields = ('name', 'filename')
    readonly_fields = ('createdAt', 'vocabSize', 'numBatches')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'location', 'filename', 'uploadedFile')
        }),
        ('Processing Status', {
            'fields': ('isPreprocessed', 'vocabSize', 'numBatches')
        }),
        ('Timestamps', {
            'fields': ('createdAt',),
            'classes': ('collapse',)
        }),
    )

@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ('runName', 'dataset', 'status', 'numEpochs', 'finalLoss', 'coherenceScore', 'createdAt')
    list_filter = ('status', 'createdAt', 'dataset')
    search_fields = ('runName', 'dataset__name')
    readonly_fields = ('createdAt', 'completedAt', 'finalLoss', 'coherenceScore', 'diversityScore')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('runName', 'dataset', 'status')
        }),
        ('Model Parameters', {
            'fields': ('hiddenDim1Size', 'hiddenDim2Size', 'latentDimSize', 'numEpochs')
        }),
        ('Results', {
            'fields': ('modelPath', 'finalLoss', 'coherenceScore', 'diversityScore')
        }),
        ('Timestamps', {
            'fields': ('createdAt', 'completedAt'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ValidationResult)
class ValidationResultAdmin(admin.ModelAdmin):
    list_display = ('trainingRun', 'validationLoss', 'coherenceScore', 'diversityScore', 'createdAt')
    list_filter = ('createdAt', 'trainingRun__dataset')
    search_fields = ('trainingRun__runName', 'trainingRun__dataset__name')
    readonly_fields = ('createdAt',)
    
    fieldsets = (
        ('Training Run', {
            'fields': ('trainingRun',)
        }),
        ('Loss Metrics', {
            'fields': ('validationLoss', 'reconstructionLoss', 'klLoss')
        }),
        ('Topic Metrics', {
            'fields': ('coherenceScore', 'diversityScore')
        }),
        ('Timestamp', {
            'fields': ('createdAt',)
        }),
    )

@admin.register(TestResult)
class TestResultAdmin(admin.ModelAdmin):
    list_display = ('trainingRun', 'testLoss', 'coherenceScore', 'diversityScore', 'createdAt')
    list_filter = ('createdAt', 'trainingRun__dataset')
    search_fields = ('trainingRun__runName', 'trainingRun__dataset__name')
    readonly_fields = ('createdAt',)
    
    fieldsets = (
        ('Training Run', {
            'fields': ('trainingRun',)
        }),
        ('Loss Metrics', {
            'fields': ('testLoss', 'reconstructionLoss', 'klLoss')
        }),
        ('Topic Metrics', {
            'fields': ('coherenceScore', 'diversityScore')
        }),
        ('Timestamp', {
            'fields': ('createdAt',)
        }),
    )

admin.site.siteHeader = 'tRopicAL Administration'
admin.site.siteTitle = 'tRopicAL Admin'
admin.site.indexTitle = 'Topic Modeling Administration'