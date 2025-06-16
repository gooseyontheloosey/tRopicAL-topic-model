import torch
import os
from .utils import *
from .preprocess import *
from .logger import logger



def modelTopics(location, filename, model, vocabulary, device = None, numBatches = None, saveDir = None):
    ''' Run inference on new dataset using trained model to extract topic hierarchies. '''
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory for results
    if saveDir is None:
        timestamp = getRunTimestamp()
        saveDir = os.path.join(os.getcwd(), 'Inference Results', f'{filename}-inference-{timestamp}')
    os.makedirs(saveDir, exist_ok = True)
    
    # Check if data is already preprocessed
    tfidfPath = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
    if not os.path.exists(tfidfPath):
        logger.info(f'Preprocessed data not found. Starting preprocessing for {filename}...')
        print(f'Preprocessing {filename}...')
        preprocess(location, filename)
        logger.info('Preprocessing complete.')
    
    # Get number of batches to process
    totalBatches = getNumBatches(location, filename)
    if numBatches is None:
        numBatches = totalBatches
    else:
        numBatches = min(numBatches, totalBatches)
    
    model.eval()  # Set model to evaluation mode
    
    print(f'\nBeginning TOPIC INFERENCE on {numBatches} batches...')
    logger.info(f'Starting topic inference on {numBatches} batches')
    
    # Storage for loss tracking
    totalReconstructionLoss = 0
    totalKLLoss = 0
    totalLoss = 0
    processedBatches = 0
    
    # Memory cleanup before starting
    checkMemory('Pre-inference', forceGc = True)
    
    # Process batches for inference
    with torch.no_grad():
        for batchId in range(numBatches):
            try:
                print(f'Processing batch {batchId+1}/{numBatches}...')
                
                # Load batch data
                tfIdfBatch, sentimentScoresByRow = loadBatchWithSentimentScores(location, filename, batchId, device)
                
                # Forward pass through model
                reconstruction, allMus, allLogVars, allZs = model.feedforward(tfIdfBatch, sentimentScoresByRow)
                
                # Calculate loss for monitoring
                reconLoss, klLoss, batchLoss = model.calculateLoss(tfIdfBatch, reconstruction, allMus, allLogVars, allZs, klWeight = 1.0)
                
                # Accumulate losses
                totalReconstructionLoss += reconLoss.item()
                totalKLLoss += klLoss.item()
                totalLoss += batchLoss.item()
                processedBatches += 1
                
                # Memory cleanup
                del reconstruction, allMus, allLogVars, allZs, tfIdfBatch, sentimentScoresByRow
                del reconLoss, klLoss, batchLoss
                
                # Periodic memory check
                if batchId % 10 == 0:
                    checkMemory(f'After inference batch {batchId}', forceGc = True)
                
            except FileNotFoundError:
                logger.warning(f'Inference batch {batchId} file not found')
                break
            except Exception as e:
                logger.error(f'Error processing inference batch {batchId}: {e}')
                continue
    
    if processedBatches == 0:
        print('No batches processed successfully!')
        logger.error('No batches processed successfully!')
        return None
    
    # Calculate average losses
    avgReconstructionLoss = totalReconstructionLoss / processedBatches
    avgKLLoss = totalKLLoss / processedBatches
    avgTotalLoss = totalLoss / processedBatches
    
    print(f'\nInference Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    logger.info(f'Inference Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    
    # Generate topic hierarchies using existing function
    print('Generating topic hierarchies...')
    visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = False)
    
    # Evaluate topic metrics using existing function
    print('Evaluating topic metrics...')
    sampleData, sampleTexts = getDataSamples(location, filename, model, device)
    
    topicMetrics = None
    if sampleData is not None and sampleTexts:
        topicMetrics = evaluateTopicMetrics(model, sampleData, sampleTexts, vocabulary)
        printAndLogMetrics(topicMetrics)
    else:
        print('Could not calculate topic metrics: insufficient sample data')
        logger.warning('Could not calculate topic metrics: insufficient sample data')
    
    # Final memory cleanup
    checkMemory('Post-inference', forceGc = True)
    
    print(f'\nInference complete! Results saved to: {saveDir}')
    logger.info(f'Topic inference complete for {filename}')
    
    # Return results
    return {
        'losses': {
            'reconstructionLoss': avgReconstructionLoss,
            'klLoss': avgKLLoss,
            'totalLoss': avgTotalLoss,
            'processedBatches': processedBatches
        },
        'topicMetrics': topicMetrics,
        'saveDir': saveDir
    }