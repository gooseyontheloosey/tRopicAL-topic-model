import os
import pandas as pd
import glob
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend for matplotlib
import itertools
import datetime
import torch.optim as optim
import numpy as np


from .vae import VAE
from .logger import logger  # Fix: Import logger object directly
from .memory import checkMemory



def loadBatchWithSentimentScores(location, filename, batchId, device):
    ''' Loads in one batch of data from TF-IDF PARQUET file, and corresponding sentiment analysis scores. '''
    
    # Load TF-IDF batch
    tfidfPath = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
    tfidfDf = pd.read_parquet(tfidfPath)
    batch = tfidfDf[tfidfDf['batchId'] == batchId]
    
    # Extract features
    features = [col for col in batch.columns if col not in ['rowId', 'batchId']]
    tfidfFeatures = batch[features].values
    
    # Load sentiment scores
    sentimentPath = os.path.join(location, f'SENTIMENT_SCORES - {filename} - DATASET', f'part-{batchId}.parquet')
    sentimentDf = pd.read_parquet(sentimentPath)
    
    sentimentScoresByRow = []
    
    # Group sentiment scores by rowId
    for rowId in batch['rowId']:
        rowSentiments = sentimentDf[sentimentDf['rowId'] == rowId]['sentenceSentimentScore'].values
        # Convert sentiment analysis scores to tensor
        sentimentScoresByRow.append(torch.tensor(
            rowSentiments, dtype = torch.float32)
                                    .to(device))
    
    # Convert TF-IDF to tensor
    tfIdfTensor = torch.tensor(tfidfFeatures, dtype = torch.float32).to(device)
    
    return tfIdfTensor, sentimentScoresByRow


def initialiseModel(vocabSize, hiddenDim1Size, hiddenDim2Size, latentDimSize):
    ''' Initialises model with an optimiser. '''
    
    # Initialise model
    model = VAE(vocabSize, hiddenDim1Size, hiddenDim2Size, latentDimSize)

    # Ensure model placed on device GPU where possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimiser = optim.Adam(model.parameters(), lr = 0.0005)

    return model, optimiser, device


def getRunTimestamp():
    ''' Returns a string timestamp for current run. '''
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def getModelSaveDir(datasetName, runTimestamp):
    ''' Returns directory path for saving models for a dataset/run. '''
    return os.path.join(os.getcwd(), 'Saved Models', f'{datasetName}-{runTimestamp}')


def saveModel(model, datasetName, saveDir, epoch = None, metric = None, isFinal = False):
    ''' Saves model according to a naming convention in a timestamped subfolder. '''

    if isFinal:
        filename = 'tRopicAL-model.pt'
    elif metric is not None and epoch is not None:
        filename = f'tRopicAL-model-epoch-{epoch}-best-{metric}.pt'
    elif epoch is not None:
        filename = f'tRopicAL-model-epoch-{epoch}.pt'
    else:
        filename = 'tRopicAL-model.pt'

    path = os.path.join(saveDir, filename)
    torch.save(model.state_dict(), path)
    logger.info(f'Model saved to {path}')
    return path


def loadModel(modelFolder, modelName, model, device = None):
    ''' Loads a model from a user-specified folder and filename. '''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join(modelFolder, modelName)

    if not os.path.exists(path):
        logger.error(f'Model file not found: {path}')
        raise FileNotFoundError(f'Model file not found: {path}')
    
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval() # Loads model in evaluation mode by default
    logger.info(f'Model loaded from {path}')

    return model


def getDataSamples(location, filename, model, device, maxSamples = 20):
    ''' Get samples from dataset to use for evaluating topic model. '''

    logger.info('In getDataSamples()...')

    # Get total number of available batches
    path = os.path.join(location, f'CLEANED - {filename} - DATASET')
    allBatchFiles = glob.glob(os.path.join(path, '*.parquet'))
    totalBatches = len(allBatchFiles)
    
    if totalBatches == 0:
        logger.warning(f'No batch files found in {path}')
        return None, []

    numBatches = min(5, totalBatches)  # Limit to 5 batches for efficiency
    batchIds = np.random.choice(totalBatches, numBatches, replace = False)
    
    tfIdfSamples = []
    textSamples = []
    
    # Process each batch
    for batchId in batchIds:
        try:
            # Load just TF-IDF scores
            batch, _ = loadBatchWithSentimentScores(location, filename, batchId, device)
            
            # Sample rows from this batch
            indices = np.random.choice(
                batch.shape[0], 
                min(20, batch.shape[0]),  # Take up to 20 rows per batch
                replace = False
            )

            # Add to list
            for i in indices:
                if len(tfIdfSamples) < maxSamples:
                    tfIdfSamples.append(batch[i])
            
            # Load corresponding cleaned text
            try:
                path = os.path.join(location, f'CLEANED - {filename} - DATASET', f'part-{batchId}.parquet')
                text = pd.read_parquet(path)

                # Add to list
                for i in indices:
                    if i < len(text) and len(textSamples) < maxSamples:
                        textSamples.append(text['text'].iloc[i])
                        
            except Exception as e:
                logger.error(f'Error loading texts for batch {batchId}: {e}')
        
        except Exception as e:
            logger.error(f'Error sampling batch {batchId}: {e}')
    
    logger.info(f'Sampled {len(tfIdfSamples)} documents for evaluation')
    
    # Convert to tensor if data loaded
    if tfIdfSamples:
        tfIdfTensor = torch.stack(tfIdfSamples).to(device)
        return tfIdfTensor, textSamples
    else:
        return None, textSamples


def calculateTopicCoherence(model, vocabulary, texts, levelIdx=None, maxTopics=5, topWords=5):
    ''' Calculates a simple topic coherence: average overlap between topic words and document words. '''
    
    logger.info('In calculateTopicCoherence()...')
    device = next(model.parameters()).device
    model.eval()
    processedTexts = [set(text.split()) for text in texts]
    results = {'coherenceByLevel': {}, 'topicsByLevel': {}}
    with torch.no_grad():
        numLevels = 3
        if levelIdx is not None:
            levelsToProcess = [levelIdx]
        else:
            levelsToProcess = range(numLevels)
        for level in levelsToProcess:
            topics = []
            for i in range(min(model.latentDimSize, maxTopics)):
                basisZ = torch.zeros(1, model.latentDimSize).to(device)
                basisZ[0, i] = 1.0
                topicDist = model.finalDecoder(basisZ)
                topIndices = topicDist[0].topk(topWords).indices.cpu().numpy()
                topics.append([vocabulary[idx] for idx in topIndices])
            # Simple overlap-based coherence
            topic_sets = [set(t) for t in topics]
            scores = []
            for topic in topic_sets:
                overlap = [len(topic & doc) / len(topic) for doc in processedTexts if len(topic) > 0]
                if overlap:
                    scores.append(sum(overlap) / len(overlap))
            coherence = sum(scores) / len(scores) if scores else 0.0
            results['coherenceByLevel'][f'level_{level}'] = coherence
            results['topicsByLevel'][f'level_{level}'] = topics
    if levelIdx is None:
        results['averageCoherence'] = sum(results['coherenceByLevel'].values()) / max(1, len(results['coherenceByLevel']))
    return results


def calculateTopicDiversity(model, vocabulary, levelIdx=None, maxTopics=5, topWords=5):
    ''' Calculates topic diversity using average pairwise Jaccard distance between topic word sets. '''
    logger.info('In calculateTopicDiversity()...')
    device = next(model.parameters()).device
    model.eval()
    results = {'diversityByLevel': {}, 'uniqueWordsByLevel': {}}
    with torch.no_grad():
        numLevels = 3
        if levelIdx is not None:
            levelsToProcess = [levelIdx]
        else:
            levelsToProcess = range(numLevels)
        for level in levelsToProcess:
            topics = []
            for i in range(min(model.latentDimSize, maxTopics)):
                basisZ = torch.zeros(1, model.latentDimSize).to(device)
                basisZ[0, i] = 1.0
                topicDist = model.finalDecoder(basisZ)
                topIndices = topicDist[0].topk(topWords).indices.cpu().numpy()
                topics.append(set([vocabulary[idx] for idx in topIndices]))
            # Jaccard diversity
            if len(topics) < 2:
                diversity = 0.0
            else:
                pairs = list(itertools.combinations(topics, 2))
                jaccard_scores = [1 - len(a & b) / len(a | b) if len(a | b) > 0 else 0 for a, b in pairs]
                diversity = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
            uniqueWords = set().union(*topics)
            results['diversityByLevel'][f'level_{level}'] = diversity
            results['uniqueWordsByLevel'][f'level_{level}'] = len(uniqueWords)
    if levelIdx is None:
        results['averageDiversity'] = sum(results['diversityByLevel'].values()) / max(1, len(results['diversityByLevel']))
    return results


def evaluateTopicMetrics(model, data, texts, vocabulary, levelIdx = None):

    ''' Calculate topic coherence and diversity metrics. '''

    logger.info('In evaluateTopicMetrics()...')
    
    # Calculate coherence and diversity metrics
    coherenceResults = calculateTopicCoherence(model, vocabulary, texts, levelIdx)
    diversityResults = calculateTopicDiversity(model, vocabulary, levelIdx)
    
    # Combine results
    combinedResults = {
        'coherence': coherenceResults,
        'diversity': diversityResults
    }
    
    # Log summary
    print('\n')
    logger.info('===== Topic Model Evaluation Summary =====')
    if levelIdx is not None:
        levelStr = f'level_{levelIdx}'
        logger.info(f'Level {levelIdx} Metrics:')
        logger.info(f'  Topic Coherence (simple): {coherenceResults["coherenceByLevel"][levelStr]:.4f}')
        logger.info(f'  Topic Diversity: {diversityResults["diversityByLevel"][levelStr]:.4f}')
    else:
        print('\n')
        logger.info('Average Metrics Across All Levels:')
        logger.info(f'  Topic Coherence (simple): {coherenceResults["averageCoherence"]:.4f}')
        logger.info(f'  Topic Diversity: {diversityResults["averageDiversity"]:.4f}')
    
    return combinedResults
  

def printAndLogMetrics(metrics, logger = logger):
    ''' Prints and logs topic coherence and diversity metrics. '''
    if not metrics:
        logger.warning('No metrics to display.')
        print('No metrics to display.')
        return
    coherence = metrics.get('coherence', {})
    diversity = metrics.get('diversity', {})
    if 'averageCoherence' in coherence:
        logger.info(f'Average Topic Coherence (NPMI): {coherence["averageCoherence"]:.4f}')
        print(f'Average Topic Coherence (NPMI): {coherence["averageCoherence"]:.4f}')
    if 'averageDiversity' in diversity:
        logger.info(f'Average Topic Diversity: {diversity["averageDiversity"]:.4f}')
        print(f'Average Topic Diversity: {diversity["averageDiversity"]:.4f}')
    # Optionally print per-level metrics
    for level in coherence.get('coherenceByLevel', {}):
        logger.info(f'{level}: Coherence={coherence["coherenceByLevel"][level]:.4f}')
    for level in diversity.get('diversityByLevel', {}):
        logger.info(f'{level}: Diversity={diversity["diversityByLevel"][level]:.4f}')


def visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = True):
    ''' Visualises topic hierarchies produced by model, either for entire corpus or most recent batch. '''

    logger.debug('In visualiseTopics()...')
    
    checkMemory('Pre-visualisation', forceGc=True)
    wordsPerTopic = 5
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Visualise topics based on batchedMode
    topicHierarchies = model.recordTopicHierarchies(topK = wordsPerTopic, vocab = vocabulary, batchedMode = batchedMode)

    uniqueHierarchies = set(tuple(tuple(level) for level in hierarchy) for hierarchy in topicHierarchies)
    print('\n')
    logger.info(f'Unique topic hierarchies: {len(uniqueHierarchies)} / {len(topicHierarchies)}')

    # Ensure save directory exists
    os.makedirs(saveDir, exist_ok=True)
    
    # Use shorter filename to avoid Windows path length limit
    outPath = os.path.join(saveDir, f'topic-hierarchies-{timestamp}.txt')
    
    # Write the file
    with open(outPath, 'w', encoding='utf-8') as f:
        f.write(f'Topic hierarchies collected ({timestamp}):\n\n')
        for i, hierarchy in enumerate(topicHierarchies, 1):
            levelsStr = ' -> '.join(f"[{', '.join(words)}]" for words in hierarchy[:maxLevels])
            f.write(f'{i}. {levelsStr}\n')

    logger.info(f'Topic hierarchies saved to {outPath}')
    return outPath