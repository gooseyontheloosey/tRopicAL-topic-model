# topicModeller/forms.py
from django import forms
from .models import Dataset, TrainingRun, InferenceResult
import os


class SplitDatasetForm(forms.Form):
    """Simple form for splitting datasets - not tied to a model"""
    uploadedFile = forms.FileField(
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.parquet'
        }),
        help_text='Select your PARQUET training dataset to split (80% training, 20% validation)'
    )
    
    def clean_uploadedFile(self):
        uploadedFile = self.cleaned_data.get('uploadedFile')
        
        if uploadedFile:
            # Check file extension
            if not uploadedFile.name.lower().endswith('.parquet'):
                raise forms.ValidationError('Please select a .parquet file.')
            
            # Check file size (500MB limit)
            if uploadedFile.size > 500 * 1024 * 1024:
                raise forms.ValidationError('File too large. Maximum size is 500MB.')
            
            # Check if filename contains 'train'
            if 'train' not in uploadedFile.name.lower():
                raise forms.ValidationError('Filename should contain "train" for proper validation file naming.')
        
        return uploadedFile


class DatasetForm(forms.ModelForm):
    fileType = forms.ChoiceField(
        choices=[('parquet', 'Parquet'), ('docx', 'DOCX')],
        widget=forms.HiddenInput(),
        initial='parquet'
    )
    
    class Meta:
        model = Dataset
        fields = ['uploadedFile']
        widgets = {
            'uploadedFile': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': '.parquet'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['uploadedFile'].required = True
        
        # Add help text
        self.fields['uploadedFile'].help_text = 'Select your dataset file(s) - names auto-generated from filenames'
    
    def clean_uploadedFile(self):
        uploadedFile = self.cleaned_data.get('uploadedFile')
        fileType = self.data.get('fileType', 'parquet')
        
        if uploadedFile:
            # Check file extension based on selected type
            name = uploadedFile.name.lower()
            if fileType == 'parquet' and not name.endswith('.parquet'):
                raise forms.ValidationError('Please select a .parquet file.')
            elif fileType == 'docx' and not name.endswith('.docx'):
                raise forms.ValidationError('Please select .docx file(s).')
            
            # Check file size (500MB limit per file)
            if uploadedFile.size > 500 * 1024 * 1024:
                raise forms.ValidationError('File too large. Maximum size is 500MB per file.')
        
        return uploadedFile


class TrainingForm(forms.ModelForm):
    class Meta:
        model = TrainingRun
        fields = [
            'dataset', 'runName', 'hiddenDim1Size', 
            'hiddenDim2Size', 'latentDimSize', 'numEpochs'
        ]
        widgets = {
            'dataset': forms.Select(attrs={'class': 'form-select'}),
            'runName': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter run name'
            }),
            'hiddenDim1Size': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '384',
                'value': 384
            }),
            'hiddenDim2Size': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '192',
                'value': 192
            }),
            'latentDimSize': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '75',
                'value': 75
            }),
            'numEpochs': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '10',
                'value': 10
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show preprocessed datasets
        self.fields['dataset'].queryset = Dataset.objects.filter(isPreprocessed=True)


class VisualiseTopicsForm(forms.Form):
    '''Form for visualising topics using a trained model'''
    trainingRun = forms.ModelChoiceField(
        queryset=TrainingRun.objects.none(),  # Will be set in __init__
        empty_label="Select a trained model...",
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Trained Model'
    )
    
    dataset = forms.ModelChoiceField(
        queryset=Dataset.objects.none(),  # Will be set in __init__
        empty_label="Select a preprocessed dataset...",
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Dataset for Inference'
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show training runs marked as trained
        self.fields['trainingRun'].queryset = TrainingRun.objects.filter(
            isMarkedTrained=True,
            status='completed'
        ).order_by('-createdAt')
        
        # Only show preprocessed datasets
        self.fields['dataset'].queryset = Dataset.objects.filter(
            isPreprocessed=True
        ).order_by('-createdAt')
        
        # Update help text based on available options
        if not self.fields['trainingRun'].queryset.exists():
            self.fields['trainingRun'].empty_label = "No trained models available - mark a completed model as 'Trained' first"
        
        if not self.fields['dataset'].queryset.exists():
            self.fields['dataset'].empty_label = "No preprocessed datasets available - upload and preprocess a dataset first"


class EnhancedModelLoadForm(forms.Form):
    '''Enhanced form for loading models with more options'''
    trainingRun = forms.ModelChoiceField(
        queryset=TrainingRun.objects.filter(status='completed').order_by('-createdAt'),
        empty_label="Select a completed training run...",
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Select Previously Trained Model'
    )
    
    # File upload field for selecting model from PC
    modelFile = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.pt'
        }),
        label='Upload Model File (Optional)',
        help_text='Select a .pt model file from your computer, or leave blank to auto-select from saved models'
    )
    
    dataset = forms.ModelChoiceField(
        queryset=Dataset.objects.filter(isPreprocessed=True).order_by('-createdAt'),
        empty_label="Select a preprocessed dataset...",
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Select Dataset'
    )
    
    action = forms.ChoiceField(
        choices=[
            ('train', 'Train (Continue Training)'),
            ('validate', 'Validate'),
            ('test', 'Test'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Action to Perform'
    )
    
    # Model parameters (will be populated from selected training run)
    hiddenDim1Size = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '384'
        }),
        label='Hidden Dimension 1 Size',
        initial=384
    )
    
    hiddenDim2Size = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '192'
        }),
        label='Hidden Dimension 2 Size',
        initial=192
    )
    
    latentDimSize = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '75'
        }),
        label='Latent Dimension Size',
        initial=75
    )
    
    # Training-specific parameters
    startEpoch = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1'
        }),
        label='Start Epoch (for training)',
        initial=1
    )
    
    numEpochs = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '10'
        }),
        label='Number of Epochs (for training)',
        initial=10
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def populate_model_files(self, training_run_id):
        '''Populate model file choices based on selected training run'''
        try:
            training_run = TrainingRun.objects.get(pk=training_run_id)
            if training_run.modelPath and os.path.exists(training_run.modelPath):
                model_files = [f for f in os.listdir(training_run.modelPath) if f.endswith('.pt')]
                
                # Sort files by modification time (newest first)
                model_files.sort(key=lambda x: os.path.getctime(os.path.join(training_run.modelPath, x)), reverse=True)
                
                choices = []
                for file in model_files:
                    # Create friendly names for common file patterns
                    if 'best-coherence' in file:
                        display_name = f"{file} (Best Coherence)"
                    elif 'best-diversity' in file:
                        display_name = f"{file} (Best Diversity)"
                    elif 'final' in file:
                        display_name = f"{file} (Final Model)"
                    elif 'epoch' in file:
                        epoch_num = file.split('epoch-')[1].split('.')[0] if 'epoch-' in file else 'Unknown'
                        display_name = f"{file} (Epoch {epoch_num})"
                    else:
                        display_name = file
                    
                    choices.append((file, display_name))
                
                # Add automatic selection option
                choices.insert(0, ('auto', 'Auto-select latest model (recommended)'))
                self.fields['modelFile'].choices = choices
            else:
                self.fields['modelFile'].choices = [('', 'No model files found')]
        except TrainingRun.DoesNotExist:
            self.fields['modelFile'].choices = [('', 'Please select a training run first')]
    
    def clean(self):
        cleaned_data = super().clean()
        action = cleaned_data.get('action')
        
        # Validate training-specific fields
        if action == 'train':
            startEpoch = cleaned_data.get('startEpoch')
            numEpochs = cleaned_data.get('numEpochs')
            
            if not startEpoch:
                self.add_error('startEpoch', 'Start epoch is required for training.')
            if not numEpochs:
                self.add_error('numEpochs', 'Number of epochs is required for training.')
            elif startEpoch and numEpochs and startEpoch >= numEpochs:
                self.add_error('numEpochs', 'Number of epochs must be greater than start epoch.')
        
        return cleaned_data

