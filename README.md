A topic modelling tool, consisting of an originally designed topic model framework (in pyTorch) and a Django web-app interface allowing users to:
- Create instances of a model
- Train models
- Upload datasets
- Conduct topic modelling


# Application Requirements

## FRONT END FRAMEWORK
Django>=4.2.0,<5.0.0

## MACHINE LEARNING & DEEP LEARNING
torch>=2.1.0
<br/>torchvision>=0.16.0
<br/>scikit-learn>=1.3.0

## NATURAL LANGUAGE PROCESSING
nltk>=3.8.1

## DATA PROCESSING & ANALYSIS
pandas>=2.0.0
<br/>numpy>=1.24.0
<br/>pyarrow>=12.0.0

## DOCUMENT PROCESSING
python-docx>=0.8.11

## SYSTEM MONITORING & UTILITIES
psutil>=5.9.0

## GUI AUTOMATION
pyautogui>=0.9.54

## VISUALIZATION
matplotlib>=3.7.0

## ADDITIONAL DEPENDENCIES
Pillow>=10.0.0  # Image processing (required by Django and matplotlib)
<br/>setuptools>=68.0.0  # Package management utilities


# INDIVIDUAL PIP INSTALL COMMANDS
## Copy and paste for individual installation:

pip install "Django>=4.2.0,<5.0.0"
<br/>pip install "torch>=2.1.0"
<br/>pip install "torchvision>=0.16.0"
<br/>pip install "scikit-learn>=1.3.0"
<br/>pip install "nltk>=3.8.1"
<br/>pip install "pandas>=2.0.0"
<br/>pip install "numpy>=1.24.0"
<br/>pip install "pyarrow>=12.0.0"
<br/>pip install "python-docx>=0.8.11"
<br/>pip install "psutil>=5.9.0"
<br/>pip install "pyautogui>=0.9.54"
<br/>pip install "matplotlib>=3.7.0"
<br/>pip install "Pillow>=10.0.0"
<br/>pip install "setuptools>=68.0.0"


To run web app:
- In a terminal, navigate to C:\Users\...\tRopicAL\tropicalWebApp\tropicalWebApp (folder where manage.py is)
- Run the following: ```python manage.py makemigrations```, ```python manage.py migrate```, ```python manage.py runserver```


If some features of web app are buggy // to run only backend code:
- In a terminal, navigate to C:\Users\...\tRopicAL\tropicalWebApp\tropicalWebApp\topicModeller
- Run ```python main.py``` and follow the prompts


There will be saved models you can use to run in C:\Users\...\tRopicAL\tropicalWebApp\tropicalWebApp\Saved Models
There will be datasets to use in C:\Users\...\tRopicAL\tropicalWebApp\tropicalWebApp\media\datasets (note: may not always show up in web app)
