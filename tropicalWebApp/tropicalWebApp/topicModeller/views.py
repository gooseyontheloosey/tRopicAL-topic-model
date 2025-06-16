# topicModeller/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone
import json
import os
import threading
import time
import pandas as pd

from .models import Dataset, TrainingRun, ValidationResult, TestResult, InferenceResult
from .forms import DatasetForm, TrainingForm, VisualiseTopicsForm, EnhancedModelLoadForm
from .parquetSplitter import splitPARQUET
from .parquetConverter import convertUploadedFile, convertMultipleDocxFiles

import sys
import os

try:
    from . import preprocess
    from . import train 
    from . import validate
    from . import test
    from . import modelTopics
    from . import utils
    from .vae import VAE  
    MODULES_LOADED = True
except ImportError as e:
    print(f'Could not import topic modelling modules: {e}')
    MODULES_LOADED = False


def dashboard(request):
    '''Main dashboard view'''
    datasets = Dataset.objects.all().order_by('-createdAt')
    trainingRuns = TrainingRun.objects.all().order_by('-createdAt')[:5]
    
    # Calculate counts for dashboard metrics
    totalDatasets = datasets.count()
    preprocessedDatasets = datasets.filter(isPreprocessed=True).count()
    totalTrainingRuns = TrainingRun.objects.count()
    completedTrainingRuns = TrainingRun.objects.filter(status='completed').count()
    
    context = {
        'datasets': datasets,
        'recentTrainingRuns': trainingRuns,
        'totalDatasets': totalDatasets,
        'preprocessedDatasets': preprocessedDatasets,
        'totalTrainingRuns': totalTrainingRuns,
        'completedTrainingRuns': completedTrainingRuns,
    }
    return render(request, 'topicModeller/dashboard.html', context)


def viewModels(request):
    '''View all models with ability to mark as trained'''
    if request.method == 'POST':
        action = request.POST.get('action')
        modelId = request.POST.get('modelId')
        
        if action == 'mark_trained' and modelId:
            try:
                trainingRun = get_object_or_404(TrainingRun, pk=modelId)
                if trainingRun.canBeMarkedTrained:
                    trainingRun.isMarkedTrained = True
                    trainingRun.save()
                    messages.success(request, f'Model "{trainingRun.runName}" marked as trained for visualisation.')
                else:
                    messages.error(request, 'Only completed models with saved paths can be marked as trained.')
            except Exception as e:
                messages.error(request, f'Error marking model as trained: {str(e)}')
        
        elif action == 'unmark_trained' and modelId:
            try:
                trainingRun = get_object_or_404(TrainingRun, pk=modelId)
                trainingRun.isMarkedTrained = False
                trainingRun.save()
                messages.success(request, f'Model "{trainingRun.runName}" unmarked as trained.')
            except Exception as e:
                messages.error(request, f'Error unmarking model: {str(e)}')
        
        return redirect('viewModels')
    
    # Get all training runs with related data
    trainingRuns = TrainingRun.objects.all().order_by('-createdAt')
    
    context = {
        'trainingRuns': trainingRuns,
        'completedRuns': trainingRuns.filter(status='completed'),
        'trainedRuns': trainingRuns.filter(isMarkedTrained=True),
    }
    return render(request, 'topicModeller/viewModels.html', context)


def visualiseTopics(request):
    '''Visualise topics using a trained model on a dataset'''
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = VisualiseTopicsForm(request.POST)
        if form.is_valid():
            trainingRun = form.cleaned_data['trainingRun']
            dataset = form.cleaned_data['dataset']
            
            # Create inference result record
            inference = InferenceResult.objects.create(
                trainingRun=trainingRun,
                dataset=dataset,
                status='pending'
            )
            
            # Run inference in background
            def runInference():
                try:
                    inference.status = 'running'
                    inference.save()
                    
                    # Load model architecture
                    model, _, device = utils.initialiseModel(
                        dataset.vocabSize,
                        trainingRun.hiddenDim1Size,
                        trainingRun.hiddenDim2Size,
                        trainingRun.latentDimSize
                    )
                    
                    # Load trained model weights
                    modelFiles = [f for f in os.listdir(trainingRun.modelPath) if f.endswith('.pt')]
                    if modelFiles:
                        # Try to get the best model first, otherwise use the latest
                        bestModels = [f for f in modelFiles if 'best' in f.lower()]
                        if bestModels:
                            modelFile = bestModels[0]  # Use first best model found
                        else:
                            # Get the latest model file
                            modelFile = max(modelFiles, key=lambda x: os.path.getctime(os.path.join(trainingRun.modelPath, x)))
                        
                        model = utils.loadModel(trainingRun.modelPath, modelFile, model, device)
                    else:
                        raise Exception('No model files found in model path')
                    
                    # Get vocabulary
                    path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                    vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                    
                    # Create save directory for inference results
                    timestamp = utils.getRunTimestamp()
                    saveDir = os.path.join(settings.MEDIA_ROOT, 'inference_results', f'{trainingRun.runName}_{dataset.name}_{timestamp}')
                    os.makedirs(saveDir, exist_ok=True)
                    
                    # Run inference
                    results = modelTopics.modelTopics(
                        location=dataset.location,
                        filename=dataset.filename,
                        model=model,
                        vocabulary=vocabulary,
                        device=device,
                        saveDir=saveDir
                    )
                    
                    # Update inference record with results
                    if results:
                        inference.status = 'completed'
                        inference.resultsPath = saveDir
                        inference.totalLoss = results['losses']['totalLoss']
                        if results.get('topicMetrics'):
                            inference.coherenceScore = results['topicMetrics']['coherence'].get('averageCoherence')
                            inference.diversityScore = results['topicMetrics']['diversity'].get('averageDiversity')
                        inference.completedAt = timezone.now()
                        inference.save()
                    else:
                        inference.status = 'failed'
                        inference.save()
                        
                except Exception as e:
                    print(f'Inference error: {e}')
                    import traceback
                    traceback.print_exc()
                    inference.status = 'failed'
                    inference.save()
            
            thread = threading.Thread(target=runInference)
            thread.start()
            
            messages.success(request, f'Topic visualisation started for model "{trainingRun.runName}" on dataset "{dataset.name}"!')
            return redirect('inferenceDetail', pk=inference.pk)
    else:
        form = VisualiseTopicsForm()
    
    # Get recent inference results for display
    recentInferences = InferenceResult.objects.all().order_by('-createdAt')[:5]
    
    context = {
        'form': form,
        'recentInferences': recentInferences,
        'modulesLoaded': MODULES_LOADED,
    }
    return render(request, 'topicModeller/visualiseTopics.html', context)


def inferenceDetail(request, pk):
    '''Detail view for inference results'''
    inference = get_object_or_404(InferenceResult, pk=pk)
    
    # Get topic hierarchies if available
    topicHierarchies = None
    if inference.status == 'completed' and inference.resultsPath:
        hierarchyFile = None
        for filename in os.listdir(inference.resultsPath):
            if filename.startswith('topic-hierarchies') and filename.endswith('.txt'):
                hierarchyFile = os.path.join(inference.resultsPath, filename)
                break
        
        if hierarchyFile and os.path.exists(hierarchyFile):
            try:
                from .topicVisualiser import TopicHierarchyGenerator
                generator = TopicHierarchyGenerator(hierarchyFile)
                topicHierarchies = generator.generateAllForTemplate(maxCount=10)
            except Exception as e:
                print(f'Error loading topic hierarchies: {e}')
    
    context = {
        'inference': inference,
        'topicHierarchies': topicHierarchies,
    }
    return render(request, 'topicModeller/inferenceDetail.html', context)


def enhancedModelLoad(request):
    '''Enhanced model loading with more options'''
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = EnhancedModelLoadForm(request.POST)
        if form.is_valid():
            trainingRun = form.cleaned_data['trainingRun']
            dataset = form.cleaned_data['dataset']
            action = form.cleaned_data['action']
            uploadedModelFile = form.cleaned_data.get('modelFile')  # This is now a file upload
            
            # Get model parameters
            hiddenDim1Size = form.cleaned_data['hiddenDim1Size']
            hiddenDim2Size = form.cleaned_data['hiddenDim2Size']
            latentDimSize = form.cleaned_data['latentDimSize']
            
            # Determine which model file to use
            if uploadedModelFile:
                # User uploaded a model file - save it temporarily
                import tempfile
                tempDir = tempfile.mkdtemp()
                tempModelPath = os.path.join(tempDir, uploadedModelFile.name)
                
                with open(tempModelPath, 'wb') as tempFile:
                    for chunk in uploadedModelFile.chunks():
                        tempFile.write(chunk)
                
                selectedModelPath = tempModelPath
                selectedModelFile = uploadedModelFile.name
                modelPath = tempDir
                
                messages.info(request, f'Using uploaded model file: {selectedModelFile}')
            else:
                # Auto-select latest model file from training run
                if not trainingRun.modelPath:
                    messages.error(request, 'Selected training run has no saved models.')
                    return render(request, 'topicModeller/enhancedModelLoad.html', {'form': form, 'modulesLoaded': MODULES_LOADED})
                
                modelFiles = [f for f in os.listdir(trainingRun.modelPath) if f.endswith('.pt')]
                if modelFiles:
                    selectedModelFile = max(modelFiles, key=lambda x: os.path.getctime(os.path.join(trainingRun.modelPath, x)))
                    selectedModelPath = os.path.join(trainingRun.modelPath, selectedModelFile)
                    modelPath = trainingRun.modelPath
                    messages.info(request, f'Using latest saved model: {selectedModelFile}')
                else:
                    messages.error(request, 'No model files found in the training run directory.')
                    return render(request, 'topicModeller/enhancedModelLoad.html', {'form': form, 'modulesLoaded': MODULES_LOADED})
            
            # Verify the model file exists
            if not os.path.exists(selectedModelPath):
                messages.error(request, f'Model file "{selectedModelFile}" not found.')
                return render(request, 'topicModeller/enhancedModelLoad.html', {'form': form, 'modulesLoaded': MODULES_LOADED})
            
            if action == 'train':
                # Create new training run for continuation
                startEpoch = form.cleaned_data['startEpoch']
                numEpochs = form.cleaned_data['numEpochs']
                
                newTrainingRun = TrainingRun.objects.create(
                    dataset=dataset,
                    runName=f'{trainingRun.runName}_continued_{utils.getRunTimestamp()}',
                    hiddenDim1Size=hiddenDim1Size,
                    hiddenDim2Size=hiddenDim2Size,
                    latentDimSize=latentDimSize,
                    numEpochs=numEpochs,
                    status='pending'
                )
                
                # Run training in background
                def runContinuedTraining():
                    try:
                        newTrainingRun.status = 'running'
                        newTrainingRun.save()
                        
                        # Load model and optimiser
                        model, optimiser, device = utils.initialiseModel(
                            dataset.vocabSize,
                            hiddenDim1Size,
                            hiddenDim2Size,
                            latentDimSize
                        )
                        
                        # Load the specific model file
                        model = utils.loadModel(modelPath, selectedModelFile, model, device)
                        messages.info(request, f'Loaded model from: {selectedModelFile}')
                        
                        # Get vocabulary
                        path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                        df = pd.read_parquet(path)
                        vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                        
                        # Create save directory
                        timestamp = utils.getRunTimestamp()
                        saveDir = utils.getModelSaveDir(dataset.filename, timestamp)
                        os.makedirs(saveDir, exist_ok=True)
                        
                        # Continue training
                        model, results = train.resumeTrainingModel(
                            dataset.location,
                            dataset.filename,
                            model,
                            optimiser,
                            startEpoch,
                            numEpochs,
                            dataset.numBatches,
                            vocabulary,
                            evalFrequency=5,
                            device=device,
                            saveDir=saveDir
                        )
                        
                        # Update training run
                        newTrainingRun.status = 'completed'
                        newTrainingRun.modelPath = saveDir
                        newTrainingRun.finalLoss = results.get('epochLosses', [0])[-1] if results.get('epochLosses') else None
                        newTrainingRun.coherenceScore = results.get('finalCoherence')
                        newTrainingRun.diversityScore = results.get('finalDiversity')
                        newTrainingRun.completedAt = timezone.now()
                        newTrainingRun.save()
                        
                    except Exception as e:
                        newTrainingRun.status = 'failed'
                        newTrainingRun.save()
                        print(f'Training error: {e}')
                        import traceback
                        traceback.print_exc()
                
                thread = threading.Thread(target=runContinuedTraining)
                thread.start()
                
                messages.success(request, f'Continued training started for "{newTrainingRun.runName}" using model "{selectedModelFile}"!')
                return redirect('trainingDetail', pk=newTrainingRun.pk)
            
            elif action == 'validate':
                # Run validation
                def runValidation():
                    try:
                        model, _, device = utils.initialiseModel(
                            dataset.vocabSize,
                            hiddenDim1Size,
                            hiddenDim2Size,
                            latentDimSize
                        )
                        
                        # Load the specific model file
                        model = utils.loadModel(modelPath, selectedModelFile, model, device)
                        
                        # Get vocabulary
                        path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                        df = pd.read_parquet(path)
                        vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                        
                        # Run validation
                        results = validate.validateModel(
                            dataset.location,
                            dataset.filename,
                            model,
                            vocabulary,
                            device,
                            dataset.numBatches
                        )
                        
                        # Save validation results
                        if results:
                            ValidationResult.objects.create(
                                trainingRun=trainingRun,
                                validationLoss=results['losses']['totalLoss'],
                                reconstructionLoss=results['losses']['reconstructionLoss'],
                                klLoss=results['losses']['klLoss'],
                                coherenceScore=results['topicMetrics']['coherence'].get('averageCoherence') if results.get('topicMetrics') else None,
                                diversityScore=results['topicMetrics']['diversity'].get('averageDiversity') if results.get('topicMetrics') else None,
                            )
                    except Exception as e:
                        print(f'Validation error: {e}')
                        import traceback
                        traceback.print_exc()
                
                thread = threading.Thread(target=runValidation)
                thread.start()
                
                messages.success(request, f'Validation started using model "{selectedModelFile}"!')
                return redirect('trainingDetail', pk=trainingRun.pk)
            
            elif action == 'test':
                # Run testing
                def runTesting():
                    try:
                        model, _, device = utils.initialiseModel(
                            dataset.vocabSize,
                            hiddenDim1Size,
                            hiddenDim2Size,
                            latentDimSize
                        )
                        
                        # Load the specific model file
                        model = utils.loadModel(modelPath, selectedModelFile, model, device)
                        
                        # Get vocabulary
                        path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                        df = pd.read_parquet(path)
                        vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                        
                        # Run testing
                        results = test.testModel(
                            dataset.location,
                            dataset.filename,
                            model,
                            vocabulary,
                            device,
                            dataset.numBatches
                        )
                        
                        # Save test results
                        if results:
                            TestResult.objects.create(
                                trainingRun=trainingRun,
                                testLoss=results['losses']['totalLoss'],
                                reconstructionLoss=results['losses']['reconstructionLoss'],
                                klLoss=results['losses']['klLoss'],
                                coherenceScore=results['topicMetrics']['coherence'].get('averageCoherence') if results.get('topicMetrics') else None,
                                diversityScore=results['topicMetrics']['diversity'].get('averageDiversity') if results.get('topicMetrics') else None,
                            )
                    except Exception as e:
                        print(f'Testing error: {e}')
                        import traceback
                        traceback.print_exc()
                
                thread = threading.Thread(target=runTesting)
                thread.start()
                
                messages.success(request, f'Testing started using model "{selectedModelFile}"!')
                return redirect('trainingDetail', pk=trainingRun.pk)
    else:
        form = EnhancedModelLoadForm()
    
    context = {
        'form': form,
        'modulesLoaded': MODULES_LOADED,
    }
    return render(request, 'topicModeller/enhancedModelLoad.html', context)


# ... [rest of the existing views remain the same] ...

def PARQUETSplitter(request):
    '''Split a training dataset into 80% training and 20% validation'''
    
    print("=== PARQUETSplitter view called ===")
    print(f"Request method: {request.method}")
    
    if request.method == 'POST':
        print("=== Processing POST request ===")
        # Import the new form
        from .forms import SplitDatasetForm
        form = SplitDatasetForm(request.POST, request.FILES)
        
        print(f"Form is valid: {form.is_valid()}")
        
        if form.is_valid():
            print("Form validation passed")
            uploadedFile = form.cleaned_data['uploadedFile']
            print(f"Processing file: {uploadedFile.name}")
            
            try:
                # Save the uploaded file temporarily with original name preserved
                import tempfile
                tempDir = tempfile.mkdtemp()
                tempPath = os.path.join(tempDir, uploadedFile.name)
                
                with open(tempPath, 'wb') as tempFile:
                    for chunk in uploadedFile.chunks():
                        tempFile.write(chunk)
                
                print(f"Temp file created: {tempPath}")
                
                # Test if file is readable first
                import pandas as pd
                df_test = pd.read_parquet(tempPath)
                print(f"File is readable, shape: {df_test.shape}")
                
                print(f"Calling splitPARQUET with: {tempDir}, {uploadedFile.name}")
                
                # Run the splitting function
                success = splitPARQUET(tempDir, uploadedFile.name)
                print(f"Split result: {success}")
                
                if success:
                    # Create validation filename
                    valFilename = uploadedFile.name.replace('train', 'validate')
                    valPath = os.path.join(tempDir, valFilename)
                    
                    print(f"Looking for validation file: {valPath}")
                    print(f"Validation file exists: {os.path.exists(valPath)}")
                    
                    if os.path.exists(valPath):
                        # Create media directory
                        mediaPath = os.path.join(settings.MEDIA_ROOT, 'datasets')
                        os.makedirs(mediaPath, exist_ok=True)
                        
                        # Create final file paths
                        finalTrainPath = os.path.join(mediaPath, uploadedFile.name)
                        finalValPath = os.path.join(mediaPath, valFilename)
                        
                        print(f"Moving {tempPath} to {finalTrainPath}")
                        print(f"Moving {valPath} to {finalValPath}")
                        
                        # Move files
                        import shutil
                        shutil.move(tempPath, finalTrainPath)
                        shutil.move(valPath, finalValPath)
                        
                        # Create dataset entries
                        trainDatasetName = os.path.splitext(uploadedFile.name)[0]
                        valDatasetName = trainDatasetName.replace('train', 'validate')
                        
                        from .models import Dataset
                        
                        # Create training dataset
                        trainDataset, created = Dataset.objects.get_or_create(
                            name=trainDatasetName,
                            defaults={
                                'filename': trainDatasetName,
                                'location': mediaPath
                            }
                        )
                        trainDataset.uploadedFile.name = os.path.relpath(finalTrainPath, settings.MEDIA_ROOT)
                        trainDataset.save()
                        
                        # Create validation dataset
                        valDataset, created = Dataset.objects.get_or_create(
                            name=valDatasetName,
                            defaults={
                                'filename': valDatasetName,
                                'location': mediaPath
                            }
                        )
                        valDataset.uploadedFile.name = os.path.relpath(finalValPath, settings.MEDIA_ROOT)
                        valDataset.save()
                        
                        # Cleanup temp directory
                        shutil.rmtree(tempDir)
                        
                        messages.success(request, f'Successfully split "{uploadedFile.name}" into training (980 rows) and validation (245 rows) datasets!')
                        return redirect('datasetList')
                    else:
                        messages.error(request, f'Split appeared to complete but validation file "{valFilename}" was not found.')
                        shutil.rmtree(tempDir)
                else:
                    messages.error(request, 'Failed to split the dataset.')
                    shutil.rmtree(tempDir)
                    
            except Exception as e:
                print(f"ERROR in processing: {e}")
                import traceback
                traceback.print_exc()
                messages.error(request, f'Error processing file: {str(e)}')
                # Clean up temp directory on error
                if 'tempDir' in locals() and os.path.exists(tempDir):
                    shutil.rmtree(tempDir)
                    
        else:
            print("Form validation failed")
            print(f"Form errors: {form.errors}")
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        print("=== Showing form (GET request) ===")
        from .forms import SplitDatasetForm
        form = SplitDatasetForm()
    
    print("=== Rendering template ===")
    return render(request, 'topicModeller/PARQUETSplitter.html', {'form': form})


def datasetList(request):
    '''List all datasets'''
    datasets = Dataset.objects.all().order_by('-createdAt')
    return render(request, 'topicModeller/datasetList.html', {'datasets': datasets})


def datasetUpload(request):
    '''Upload and preprocess dataset'''
    if request.method == 'POST':
        # Handle multiple file uploads for DOCX
        fileType = request.POST.get('fileType', 'parquet')
        files = request.FILES.getlist('uploadedFile')
        
        if not files:
            messages.error(request, 'Please select file(s) to upload.')
            return render(request, 'topicModeller/datasetUpload.html', {'form': DatasetForm()})
        
        try:
            if fileType == 'docx':
                # Handle DOCX files
                if len(files) > 1:
                    # Multiple DOCX files - use 'combined' as dataset name
                    datasetName = 'combined'
                    mediaPath = os.path.join(settings.MEDIA_ROOT, 'datasets')
                    convertedPath = convertMultipleDocxFiles(files, mediaPath, datasetName)
                    
                    if convertedPath:
                        # Create dataset object
                        dataset = Dataset(name=datasetName)
                        relativePath = os.path.relpath(convertedPath, settings.MEDIA_ROOT)
                        dataset.uploadedFile.name = relativePath
                        dataset.location = os.path.dirname(convertedPath)
                        dataset.filename = datasetName
                        dataset.save()
                        
                        messages.success(request, f'Successfully combined {len(files)} DOCX files and converted to Parquet format!')
                    else:
                        messages.error(request, 'Failed to convert DOCX files. Please check the file formats.')
                        return render(request, 'topicModeller/datasetUpload.html', {'form': DatasetForm()})
                else:
                    # Single DOCX file - use filename without extension
                    uploadedFile = files[0]
                    datasetName = os.path.splitext(uploadedFile.name)[0]
                    mediaPath = os.path.join(settings.MEDIA_ROOT, 'datasets')
                    convertedPath = convertUploadedFile(uploadedFile, mediaPath)
                    
                    if convertedPath:
                        # Create dataset object
                        dataset = Dataset(name=datasetName)
                        relativePath = os.path.relpath(convertedPath, settings.MEDIA_ROOT)
                        dataset.uploadedFile.name = relativePath
                        dataset.location = os.path.dirname(convertedPath)
                        dataset.filename = datasetName
                        dataset.save()
                        
                        messages.success(request, 'DOCX file converted to Parquet format successfully!')
                    else:
                        messages.error(request, 'Failed to convert DOCX file. Please check the file format.')
                        return render(request, 'topicModeller/datasetUpload.html', {'form': DatasetForm()})
            else:
                # Handle single Parquet file
                if len(files) > 1:
                    messages.error(request, 'Only one Parquet file can be uploaded at a time.')
                    return render(request, 'topicModeller/datasetUpload.html', {'form': DatasetForm()})
                
                uploadedFile = files[0]
                datasetName = os.path.splitext(uploadedFile.name)[0]
                
                # Create dataset object
                dataset = Dataset(name=datasetName)
                dataset.uploadedFile = uploadedFile
                dataset.filename = datasetName
                dataset.save()
                
                # Set location after file is saved
                dataset.location = os.path.dirname(dataset.uploadedFile.path)
                dataset.save()
                
                messages.success(request, f'Parquet file uploaded successfully!')
            
            return redirect('datasetDetail', pk=dataset.pk)
            
        except Exception as e:
            messages.error(request, f'Error uploading dataset: {str(e)}')
            return render(request, 'topicModeller/datasetUpload.html', {'form': DatasetForm()})
    else:
        form = DatasetForm()
    
    return render(request, 'topicModeller/datasetUpload.html', {'form': form})


def datasetDetail(request, pk):
    '''Dataset detail view'''
    dataset = get_object_or_404(Dataset, pk=pk)
    trainingRuns = TrainingRun.objects.filter(dataset=dataset).order_by('-createdAt')
    
    context = {
        'dataset': dataset,
        'trainingRuns': trainingRuns,
    }
    return render(request, 'topicModeller/datasetDetail.html', context)


def preprocessDataset(request, pk):
    '''Preprocess dataset'''
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('datasetDetail', pk=pk)
    
    if request.method == 'POST':
        try:
            # Run preprocessing in a separate thread
            def runPreprocessing():
                try:
                    # Use your actual preprocessing function
                    vocabSize, numBatches = preprocess.preprocess(dataset.location, dataset.filename)
                    
                    # Update dataset with actual values
                    dataset.isPreprocessed = True
                    dataset.vocabSize = vocabSize
                    dataset.numBatches = numBatches
                    dataset.save()
                    
                except Exception as e:
                    print(f'Preprocessing error: {e}')
                    # You might want to set an error status on the dataset model
            
            thread = threading.Thread(target=runPreprocessing)
            thread.start()
            
            messages.success(request, 'Preprocessing started! This may take a while.')
            return redirect('datasetDetail', pk=pk)
            
        except Exception as e:
            messages.error(request, f'Error starting preprocessing: {e}')
    
    return render(request, 'topicModeller/preprocessConfirm.html', {'dataset': dataset})


def trainingNew(request):
    '''Start new training'''
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            trainingRun = form.save()
            
            # Start training in background
            def runTraining():
                try:
                    trainingRun.status = 'running'
                    trainingRun.save()
                    
                    # Initialise model with actual parameters
                    dataset = trainingRun.dataset
                    model, optimiser, device = utils.initialiseModel(
                        dataset.vocabSize,
                        trainingRun.hiddenDim1Size,
                        trainingRun.hiddenDim2Size,
                        trainingRun.latentDimSize
                    )
                    
                    # Get vocabulary
                    path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                    vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                    
                    # Get timestamp and create save directory
                    timestamp = utils.getRunTimestamp()
                    saveDir = utils.getModelSaveDir(dataset.filename, timestamp)
                    os.makedirs(saveDir, exist_ok=True)
                    
                    # Run actual training
                    model, results = train.trainNewModel(
                        dataset.location, 
                        dataset.filename, 
                        model, 
                        optimiser, 
                        trainingRun.numEpochs, 
                        dataset.numBatches, 
                        vocabulary, 
                        evalFrequency=5, 
                        device=device, 
                        saveDir=saveDir
                    )
                    
                    # Save results
                    trainingRun.status = 'completed'
                    trainingRun.modelPath = saveDir
                    trainingRun.finalLoss = results.get('epochLosses', [0])[-1] if results.get('epochLosses') else None
                    trainingRun.coherenceScore = results.get('finalCoherence')
                    trainingRun.diversityScore = results.get('finalDiversity')
                    trainingRun.completedAt = timezone.now()
                    trainingRun.save()
                    
                except Exception as e:
                    trainingRun.status = 'failed'
                    trainingRun.save()
                    print(f'Training error: {e}')
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=runTraining)
            thread.start()
            
            messages.success(request, 'Training started successfully!')
            return redirect('trainingDetail', pk=trainingRun.pk)
    else:
        form = TrainingForm()
    
    return render(request, 'topicModeller/trainingNew.html', {'form': form, 'modulesLoaded': MODULES_LOADED})


def trainingList(request):
    '''List all training runs'''
    trainingRuns = TrainingRun.objects.all().order_by('-createdAt')
    
    # Calculate status counts for summary
    statusCounts = {
        'completed': trainingRuns.filter(status='completed').count(),
        'running': trainingRuns.filter(status='running').count(),
        'failed': trainingRuns.filter(status='failed').count(),
        'pending': trainingRuns.filter(status='pending').count(),
    }
    
    context = {
        'trainingRuns': trainingRuns,
        'statusCounts': statusCounts,
    }
    return render(request, 'topicModeller/trainingList.html', context)


def trainingDetail(request, pk):
    '''Training run detail view'''
    trainingRun = get_object_or_404(TrainingRun, pk=pk)
    validationResults = ValidationResult.objects.filter(trainingRun=trainingRun)
    testResults = TestResult.objects.filter(trainingRun=trainingRun)
    
    context = {
        'trainingRun': trainingRun,
        'validationResults': validationResults,
        'testResults': testResults,
    }
    return render(request, 'topicModeller/trainingDetail.html', context)


def modelLoad(request):
    '''Load existing model'''
    if request.method == 'POST':
        form = ModelLoadForm(request.POST)
        if form.is_valid():
            # Handle model loading logic here
            messages.success(request, 'Model loaded successfully!')
            return redirect('dashboard')
    else:
        form = ModelLoadForm()
    
    return render(request, 'topicModeller/modelLoad.html', {'form': form})


def validateModel(request, pk):
    '''Validate a trained model'''
    trainingRun = get_object_or_404(TrainingRun, pk=pk)
    
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('trainingDetail', pk=pk)
    
    if request.method == 'POST':
        try:
            # Run validation in background
            def runValidation():
                try:
                    # Load model
                    dataset = trainingRun.dataset
                    model, _, device = utils.initialiseModel(
                        dataset.vocabSize,
                        trainingRun.hiddenDim1Size,
                        trainingRun.hiddenDim2Size,
                        trainingRun.latentDimSize
                    )
                    
                    # Load trained model weights
                    modelFiles = [f for f in os.listdir(trainingRun.modelPath) if f.endswith('.pt')]
                    if modelFiles:
                        latestModel = max(modelFiles, key=lambda x: os.path.getctime(os.path.join(trainingRun.modelPath, x)))
                        model = utils.loadModel(trainingRun.modelPath, latestModel, model, device)
                    
                    # Get vocabulary
                    path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                    vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                    
                    # Run validation
                    results = validate.validateModel(
                        dataset.location, 
                        dataset.filename, 
                        model, 
                        vocabulary, 
                        device, 
                        dataset.numBatches
                    )
                    
                    # Save validation results
                    if results:
                        ValidationResult.objects.create(
                            trainingRun=trainingRun,
                            validationLoss=results['losses']['totalLoss'],
                            reconstructionLoss=results['losses']['reconstructionLoss'],
                            klLoss=results['losses']['klLoss'],
                            coherenceScore=results['topicMetrics']['coherence'].get('averageCoherence') if results.get('topicMetrics') else None,
                            diversityScore=results['topicMetrics']['diversity'].get('averageDiversity') if results.get('topicMetrics') else None,
                        )
                        
                except Exception as e:
                    print(f'Validation error: {e}')
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=runValidation)
            thread.start()
            
            messages.success(request, 'Validation started!')
            return redirect('trainingDetail', pk=pk)
            
        except Exception as e:
            messages.error(request, f'Error starting validation: {e}')
    
    return render(request, 'topicModeller/validateConfirm.html', {'trainingRun': trainingRun})


def testModel(request, pk):
    '''Test a trained model'''
    trainingRun = get_object_or_404(TrainingRun, pk=pk)
    
    if not MODULES_LOADED:
        messages.error(request, 'Topic modelling modules not available. Please check your setup.')
        return redirect('trainingDetail', pk=pk)
    
    if request.method == 'POST':
        try:
            # Run testing in background
            def runTesting():
                try:
                    # Load model
                    dataset = trainingRun.dataset
                    model, _, device = utils.initialiseModel(
                        dataset.vocabSize,
                        trainingRun.hiddenDim1Size,
                        trainingRun.hiddenDim2Size,
                        trainingRun.latentDimSize
                    )
                    
                    # Load trained model weights
                    modelFiles = [f for f in os.listdir(trainingRun.modelPath) if f.endswith('.pt')]
                    if modelFiles:
                        latestModel = max(modelFiles, key=lambda x: os.path.getctime(os.path.join(trainingRun.modelPath, x)))
                        model = utils.loadModel(trainingRun.modelPath, latestModel, model, device)
                    
                    # Get vocabulary
                    path = os.path.join(dataset.location, f'TF-IDF_SCORES - {dataset.filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                    vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]
                    
                    # Run testing
                    results = test.testModel(
                        dataset.location, 
                        dataset.filename, 
                        model, 
                        vocabulary, 
                        device, 
                        dataset.numBatches
                    )
                    
                    # Save test results
                    if results:
                        TestResult.objects.create(
                            trainingRun=trainingRun,
                            testLoss=results['losses']['totalLoss'],
                            reconstructionLoss=results['losses']['reconstructionLoss'],
                            klLoss=results['losses']['klLoss'],
                            coherenceScore=results['topicMetrics']['coherence'].get('averageCoherence') if results.get('topicMetrics') else None,
                            diversityScore=results['topicMetrics']['diversity'].get('averageDiversity') if results.get('topicMetrics') else None,
                        )
                        
                except Exception as e:
                    print(f'Testing error: {e}')
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=runTesting)
            thread.start()
            
            messages.success(request, 'Testing started!')
            return redirect('trainingDetail', pk=pk)
            
        except Exception as e:
            messages.error(request, f'Error starting testing: {e}')
    
    return render(request, 'topicModeller/testConfirm.html', {'trainingRun': trainingRun})


@csrf_exempt
def getTrainingStatus(request, pk):
    '''API endpoint to get training status'''
    trainingRun = get_object_or_404(TrainingRun, pk=pk)
    return JsonResponse({
        'status': trainingRun.status,
        'finalLoss': trainingRun.finalLoss,
        'coherenceScore': trainingRun.coherenceScore,
        'diversityScore': trainingRun.diversityScore,
    })


@csrf_exempt
def getInferenceStatus(request, pk):
    '''API endpoint to get inference status'''
    inference = get_object_or_404(InferenceResult, pk=pk)
    return JsonResponse({
        'status': inference.status,
        'totalLoss': inference.totalLoss,
        'coherenceScore': inference.coherenceScore,
        'diversityScore': inference.diversityScore,
    })


@csrf_exempt
def getModelFiles(request, pk):
    '''API endpoint to get available model files for a training run'''
    try:
        trainingRun = get_object_or_404(TrainingRun, pk=pk)
        
        if not trainingRun.modelPath or not os.path.exists(trainingRun.modelPath):
            return JsonResponse({'files': []})
        
        model_files = [f for f in os.listdir(trainingRun.modelPath) if f.endswith('.pt')]
        
        # Sort files by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(trainingRun.modelPath, x)), reverse=True)
        
        files_with_info = []
        for file in model_files:
            file_path = os.path.join(trainingRun.modelPath, file)
            file_size = os.path.getsize(file_path)
            modified_time = os.path.getctime(file_path)
            
            # Create friendly names
            if 'best-coherence' in file:
                display_name = f"{file} (Best Coherence)"
                category = "best"
            elif 'best-diversity' in file:
                display_name = f"{file} (Best Diversity)"
                category = "best"
            elif 'final' in file:
                display_name = f"{file} (Final Model)"
                category = "final"
            elif 'epoch' in file:
                epoch_num = file.split('epoch-')[1].split('.')[0] if 'epoch-' in file else 'Unknown'
                display_name = f"{file} (Epoch {epoch_num})"
                category = "epoch"
            else:
                display_name = file
                category = "other"
            
            files_with_info.append({
                'filename': file,
                'display_name': display_name,
                'category': category,
                'size': file_size,
                'modified': modified_time,
                'size_human': formatFileSize(file_size)
            })
        
        return JsonResponse({'files': files_with_info})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def formatFileSize(bytes):
    '''Helper function to format file size'''
    if bytes == 0:
        return '0 B'
    size_names = ['B', 'KB', 'MB', 'GB']
    i = 0
    while bytes >= 1024 and i < len(size_names) - 1:
        bytes /= 1024.0
        i += 1
    return f'{bytes:.1f} {size_names[i]}'