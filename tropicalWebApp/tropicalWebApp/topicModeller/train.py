import torch
from .utils import *  


def trainNewModel(location, filename, model, optimiser, numEpochs, numBatches, vocabulary = None, evalFrequency = 5, device = None, saveDir = None):
    ''' Train model for specified number of epochs. '''

    timestamp = getRunTimestamp()

    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stepSize = 10
    bestMetrics = {'coherence': -1.0, 'diversity': -1.0}

    # Lists to store epoch losses
    epochLosses = []
    epochReconLosses = []
    epochKLLosses = []
    
    print(f'\n === Beginning TRAINING for {numEpochs} epochs on {numBatches} batches... ===')

    # Training loop
    for epoch in range(numEpochs):
        model.train()  # Set model to training mode
        epochLoss = 0
        epochReconLoss = 0
        epochKLLoss = 0
        
        # Process x batches at a time before memory cleanup
        for stepStart in range(0, numBatches, stepSize):
            stepEnd = min(stepStart + stepSize, numBatches)
            
            for batchId in range(stepStart, stepEnd):
                print(f'\nProcessing batch {batchId+1}/{numBatches}...')
                
                # Load batch with sentence-level sentiment
                tfIdfBatch, sentimentScores = loadBatchWithSentimentScores(location, filename, batchId, device)
                
                print('Feeding forward...')
                # Forward pass
                reconstruction, allMus, allLogVars, allZs = model.feedforward(tfIdfBatch, sentimentScores)

                # Gradually increase KL Weight, from 0.1 to 1.0
                klWeight = min(1.0, (epoch + 1) / 10)
                
                # Calculate loss
                reconLoss, klLoss, totalLoss = model.calculateLoss(tfIdfBatch, reconstruction, allMus, allLogVars, allZs, klWeight = klWeight)
                
                print('Propagating backwards...')
                # Backward pass
                optimiser.zero_grad()
                totalLoss.backward()
                optimiser.step()
                
                # Memory cleanup
                del reconstruction, allMus, allLogVars, allZs, tfIdfBatch, sentimentScores
                
                # Accumulate losses
                epochLoss += totalLoss.item()
                epochReconLoss += reconLoss.item()
                epochKLLoss += klLoss.item()
                
                # Print progress
                print(f'\n  Batch {batchId+1}/{numBatches} - Loss: {totalLoss.item():.4f}')
                
                # Memory cleanup
                del reconLoss, klLoss, totalLoss
            
            # Force memory checks after each batch chunk
            checkMemory(f'\nAfter batch chunk {stepStart}-{stepEnd}', forceGc = True)
        
        # Calculate average losses
        avgLoss = epochLoss / numBatches
        avgReconLoss = epochReconLoss / numBatches
        avgKLLoss = epochKLLoss / numBatches

        # Store losses for this epoch
        epochLosses.append(avgLoss)
        epochReconLosses.append(avgReconLoss)
        epochKLLosses.append(avgKLLoss)
        
        print(f'\nEpoch {epoch+1}/{numEpochs} - Loss: {avgLoss:.4f} (Recon: {avgReconLoss:.4f}, KL: {avgKLLoss:.4f})')
        
        # Save model every epoch
        saveModel(model, filename, saveDir, epoch = epoch + 1)
        
        # Evaluate topic metrics every evalFrequency epochs, or at last epoch
        if vocabulary and ((epoch + 1) % evalFrequency == 0 or epoch == numEpochs - 1):
            model.eval()  # Set model to evaluation mode
            print('\nEvaluating topic metrics...')
            
            # Sample data for evaluation
            sampleData, sampleTexts = getDataSamples(location, filename, model, device)
            
            if sampleData is not None and sampleTexts:
                # Evaluate topic metrics
                metrics = evaluateTopicMetrics(model, sampleData, sampleTexts, vocabulary)
                
                # Print and log metrics
                printAndLogMetrics(metrics)
                
                # Extract main metrics
                coherence = metrics['coherence'].get('averageCoherence', -1.0)
                diversity = metrics['diversity'].get('averageDiversity', -1.0)
                
                # Save best models based on coherence and diversity
                if coherence > bestMetrics['coherence']:
                    bestMetrics['coherence'] = coherence
                    saveModel(model, filename, saveDir, epoch = epoch + 1, metric = 'coherence')
                    print('\nNew best coherence! Model saved.')
                
                if diversity > bestMetrics['diversity']:
                    bestMetrics['diversity'] = diversity
                    saveModel(model, filename, saveDir, epoch = epoch + 1, metric = 'diversity')
                    print('\nNew best diversity! Model saved.')
            else:
                print('No sample data available for metric evaluation.')
            
            # Visualise topics for most recent batch
            visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = True)
            
            # Force cleanup
            checkMemory('After evaluation', forceGc = True)

    # Save final model after training complete
    saveModel(model, filename, saveDir, isFinal = True)
    print('\nTraining complete!')

    # Output all epoch losses
    print('\nEpoch Loss Summary:')
    print('Epoch\tTotal Loss\tRecon Loss\tKL Loss')
    for i, (tot, rec, kl) in enumerate(zip(epochLosses, epochReconLosses, epochKLLosses), 1):
        print(f'{i}\t{tot:.4f}\t\t{rec:.4f}\t\t{kl:.4f}')

    # Save training results
    results = {
        "runTimestamp": timestamp,
        "dataset": filename,
        "numEpochs": numEpochs,
        "epochLosses": epochLosses,
        "epochReconLosses": epochReconLosses,
        "epochKLLosses": epochKLLosses,
        "finalCoherence": bestMetrics.get('coherence', None),
        "finalDiversity": bestMetrics.get('diversity', None)
    }

    return model, results



def resumeTrainingModel(location, filename, model, optimiser, startEpoch, numEpochs, numBatches, vocabulary = None, evalFrequency = 5, device = None, saveDir = None):
    ''' Resume training from a specific epoch. '''

    timestamp = getRunTimestamp()

    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stepSize = 10
    bestMetrics = {'coherence': -1.0, 'diversity': -1.0}

    # Lists to store epoch losses
    epochLosses = []
    epochReconLosses = []
    epochKLLosses = []
    
    print(f'\n === Resuming TRAINING from epoch {startEpoch} to {numEpochs} on {numBatches} batches... ===')

    # Training loop
    for epoch in range(startEpoch, numEpochs):
        model.train()  # Set model to training mode
        epochLoss = 0
        epochReconLoss = 0
        epochKLLoss = 0
        
        # Process x batches at a time before memory cleanup
        for stepStart in range(0, numBatches, stepSize):
            stepEnd = min(stepStart + stepSize, numBatches)
            
            for batchId in range(stepStart, stepEnd):
                print(f'\n\nProcessing batch {batchId+1}/{numBatches}...')
                
                # Load batch with sentence-level sentiment
                tfIdfBatch, sentimentScores = loadBatchWithSentimentScores(location, filename, batchId, device)
                
                print('Feeding forward...')
                # Forward pass
                reconstruction, allMus, allLogVars, allZs = model.feedforward(tfIdfBatch, sentimentScores)

                # Gradually increase KL Weight, from 0.1 to 1.0
                klWeight = min(1.0, (epoch + 1) / 10)
                
                # Calculate loss
                reconLoss, klLoss, totalLoss = model.calculateLoss(tfIdfBatch, reconstruction, allMus, allLogVars, allZs, klWeight = klWeight)
                
                print('Propagating backwards...')
                # Backward pass
                optimiser.zero_grad()
                totalLoss.backward()
                optimiser.step()
                
                # Memory cleanup
                del reconstruction, allMus, allLogVars, allZs, tfIdfBatch, sentimentScores
                
                # Accumulate losses
                epochLoss += totalLoss.item()
                epochReconLoss += reconLoss.item()
                epochKLLoss += klLoss.item()
                
                # Print progress
                print(f'\n  ==> Batch {batchId+1}/{numBatches} - Loss: {totalLoss.item():.4f}')
                
                # Memory cleanup
                del reconLoss, klLoss, totalLoss
            
            # Force memory checks after each batch chunk
            checkMemory(f'\nAfter batch chunk {stepStart}-{stepEnd}', forceGc = True)
        
        # Calculate average losses
        avgLoss = epochLoss / numBatches
        avgReconLoss = epochReconLoss / numBatches
        avgKLLoss = epochKLLoss / numBatches

        # Store losses for this epoch
        epochLosses.append(avgLoss)
        epochReconLosses.append(avgReconLoss)
        epochKLLosses.append(avgKLLoss)
        
        print(f'\nEpoch {epoch+1}/{numEpochs} - Loss: {avgLoss:.4f} (Recon: {avgReconLoss:.4f}, KL: {avgKLLoss:.4f})')
        
        # Save model every epoch
        saveModel(model, filename, saveDir, epoch = epoch + 1)
        
        # Evaluate topic metrics every evalFrequency epochs, or at last epoch
        if vocabulary and ((epoch + 1) % evalFrequency == 0 or epoch == numEpochs - 1):
            model.eval()  # Set model to evaluation mode
            print('\nEvaluating topic metrics...')
            
            # Sample data for evaluation
            sampleData, sampleTexts = getDataSamples(location, filename, model, device)
            
            if sampleData is not None and sampleTexts:
                # Evaluate topic metrics
                metrics = evaluateTopicMetrics(model, sampleData, sampleTexts, vocabulary)
                
                # Print and log metrics
                printAndLogMetrics(metrics)
                
                # Extract main metrics
                coherence = metrics['coherence'].get('averageCoherence', -1.0)
                diversity = metrics['diversity'].get('averageDiversity', -1.0)
                
                # Save best models based on coherence and diversity
                if coherence > bestMetrics['coherence']:
                    bestMetrics['coherence'] = coherence
                    saveModel(model, filename, saveDir, epoch = epoch + 1, metric = 'coherence')
                    print('\nNew best coherence! Model saved.')
                
                if diversity > bestMetrics['diversity']:
                    bestMetrics['diversity'] = diversity
                    saveModel(model, filename, saveDir, epoch = epoch + 1, metric = 'diversity')
                    print('\nNew best diversity! Model saved.')
            else:
                print('No sample data available for metric evaluation.')
            
            # Visualise topics for most recent batch
            visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = True)
            
            # Force cleanup
            checkMemory('After evaluation', forceGc = True)
        
    # Save final model after training complete
    saveModel(model, filename, saveDir, isFinal = True)
    print('\nTraining complete!')

    # Output all epoch losses
    print('\nEpoch Loss Summary:')
    print('Epoch\tTotal Loss\tRecon Loss\tKL Loss')
    for i, (tot, rec, kl) in enumerate(zip(epochLosses, epochReconLosses, epochKLLosses), 1):
        print(f'{i}\t{tot:.4f}\t\t{rec:.4f}\t\t{kl:.4f}')

    # Save training results
    results = {
        "runTimestamp": timestamp,
        "dataset": filename,
        "numEpochs": numEpochs,
        "epochLosses": epochLosses,
        "epochReconLosses": epochReconLosses,
        "epochKLLosses": epochKLLosses,
        "finalCoherence": bestMetrics.get('coherence', None),
        "finalDiversity": bestMetrics.get('diversity', None)
    }

    return model, results