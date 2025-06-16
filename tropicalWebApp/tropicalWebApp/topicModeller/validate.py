import torch
from .utils import *
from .logger import logger



def validateModel(location, filename, model, vocabulary, device = None, numBatches = 100):
    ''' Validate a trained model on validation data. '''
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()  # Set model to evaluation mode

    print(f'\n Beginning VALIDATION on {numBatches} batches...')
    logger.info(f'Starting validation on {numBatches} batches')
    
    # Step 1: Calculate loss on validation data
    totalReconstructionLoss = 0
    totalKLLoss = 0
    totalLoss = 0
    processedBatches = 0
    
    # Memory cleanup before starting
    checkMemory('Pre-validation', forceGc = True)
    
    with torch.no_grad():
        for batchId in range(numBatches):  # Limit to numBatches max
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
                    checkMemory(f'After validation batch {batchId}', forceGc = True)
                
            except FileNotFoundError:
                logger.warning(f'Validation batch {batchId} file not found')
                break  # No more batches
            except Exception as e:
                logger.error(f'Error processing validation batch {batchId}: {e}')
    
    if processedBatches == 0:
        print('No validation batches found!')
        logger.error('No validation batches found!')
        return None
    
    # Calculate average losses
    avgReconstructionLoss = totalReconstructionLoss / processedBatches
    avgKLLoss = totalKLLoss / processedBatches
    avgTotalLoss = totalLoss / processedBatches
    
    print(f'Validation Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    logger.info(f'Validation Loss: {avgTotalLoss:.4f} (Recon: {avgReconstructionLoss:.4f}, KL: {avgKLLoss:.4f})')
    
    # Step 2: Evaluate topic metrics on validation data
    valData, valTexts = getDataSamples(location, filename, model, device)
    
    # Step 3: Calculate and print topic metrics
    topicMetrics = None
    if valData is not None and valTexts:
        logger.info('Calculating topic metrics on validation data...')
        topicMetrics = evaluateTopicMetrics(model, valData, valTexts, vocabulary)
    else:
        print('Could not calculate topic metrics: insufficient validation data')
        logger.warning('Could not calculate topic metrics: insufficient validation data')
    
    # Final memory cleanup
    checkMemory('Post-validation', forceGc = True)

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