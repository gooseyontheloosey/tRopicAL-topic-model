import torch
from .utils import *
from .preprocess import *
from .logger import logger



def testModel(location, filename, model, vocabulary, device = None, numBatches = None):
    ''' Evaluate a trained model on test data. '''
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If numBatches not specified, find how many are available
    if numBatches is None:
        numBatches = getNumBatches(location, filename)
    
    model.eval()  # Set model to evaluation mode

    print(f'\nBeginning TESTING on {numBatches} batches...')
    logger.info(f'Starting testing on {numBatches} batches')
    
    # Calculate loss on test data
    totalReconstructionLoss = 0
    totalKLLoss = 0
    totalLoss = 0
    processedBatches = 0
    
    # Memory cleanup before starting
    checkMemory('Pre-testing', forceGc = True)
    
    with torch.no_grad():
        for batchId in range(numBatches):
            try:
                print(f'\nProcessing batch {batchId+1}/{numBatches}...')
                
                # Load batch
                tfIdfBatch, sentimentScoresByRow = loadBatchWithSentimentScores(location, filename, batchId, device)

                print('Feeding forward...')
                # Forward pass
                reconstruction, allMus, allLogVars, allZs = model.feedforward(tfIdfBatch, sentimentScoresByRow)

                # Calculate loss
                reconstructionLoss, klLoss, batchLoss = model.calculateLoss(tfIdfBatch, reconstruction, allMus, allLogVars, allZs, klWeight = 1.0)
                
                # Accumulate losses
                totalReconstructionLoss += reconstructionLoss.item()
                totalKLLoss += klLoss.item()
                totalLoss += batchLoss.item()
                processedBatches += 1
                
                # Memory cleanup
                del reconstruction, allMus, allLogVars, allZs, tfIdfBatch, sentimentScoresByRow
                del reconstructionLoss, klLoss, batchLoss
                
                # Periodic memory check
                if batchId % 10 == 0:
                    checkMemory(f'After test batch {batchId}', forceGc = True)
                
            except FileNotFoundError:
                logger.warning(f'Test batch {batchId} file not found')
                break  # No more batches
            except Exception as e:
                logger.error(f'Error processing test batch {batchId}: {e}')
    
    if processedBatches == 0:
        print('No test batches found!')
        logger.error('No test batches found!')
        return None
    
    # Calculate average losses
    avgReconstructionLoss = totalReconstructionLoss / processedBatches
    avgKLLoss = totalKLLoss / processedBatches
    avgTotalLoss = totalLoss / processedBatches
    
    print(f'Test Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    logger.info(f'Test Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    
    # Evaluate topic metrics on test data
    testData, testTexts = getDataSamples(location, filename, model, device)
    
    # Calculate and print topic metrics
    topicMetrics = None
    if testData is not None and testTexts:
        logger.info('Calculating topic metrics on test data...')
        topicMetrics = evaluateTopicMetrics(model, testData, testTexts, vocabulary)
    else:
        print('Could not calculate topic metrics: insufficient test data')
        logger.warning('Could not calculate topic metrics: insufficient test data')
    
    # Final memory cleanup
    checkMemory('Post-testing', forceGc = True)
    
    # Save training results
    results = {
        'losses': {
            'reconstructionLoss': avgReconstructionLoss,
            'klLoss': avgKLLoss,
            'totalLoss': avgTotalLoss,
            'numBatches': processedBatches
        },
        'topicMetrics': topicMetrics
    }

    return model, results