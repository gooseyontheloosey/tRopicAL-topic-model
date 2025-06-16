import json
import os
import nltk
import pandas as pd

import preprocess
import utils
import train
import validate
import test
import modelTopics
import logger
from logger import logger as log  # Import the logger object separately



def promptModelParams():
    hiddenDim1Size = int(input('Enter hiddenDim1 size (e.g., 384): '))
    hiddenDim2Size = int(input('Enter hiddenDim2 size (e.g., 192): '))
    latentDimSize = int(input('Enter latentDim size (e.g., 75): '))
    return hiddenDim1Size, hiddenDim2Size, latentDimSize


def promptLocationAndFilename():
    location = input('Enter dataset location: ')
    filename = input('Enter dataset filename: ')
    return location, filename


def main():
    running = True

    while running:
        print('''\nOPTIONS
        0. Preprocess data only
        1. Train model from scratch
        2. Load model from checkpoint
        OR PRESS ANY KEY TO QUIT
        ''')
        option = input('Enter option: ')
        vocabulary = None

        try:
            if option == '0':
                location, filename = promptLocationAndFilename()
                preprocess.preprocess(location, filename)
                print('Preprocessing complete.')


            elif option == '1':
                location, filename = promptLocationAndFilename()
                hiddenDim1Size, hiddenDim2Size, latentDimSize = promptModelParams()

                try:
                    # Check if dataset has been preprocessed
                    path = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                except FileNotFoundError:
                    print(f'\nPreprocessed data not found at {path}. Please preprocess data first or supply correct path.')
                    continue


                timestamp = utils.getRunTimestamp()
                saveDir = utils.getModelSaveDir(filename, timestamp)
                os.makedirs(saveDir, exist_ok=True)

                vocabSize = preprocess.getVocabSize(location, filename)
                numBatches = preprocess.getNumBatches(location, filename)

                model, optimiser, device = utils.initialiseModel(vocabSize, hiddenDim1Size, hiddenDim2Size, latentDimSize)

                path = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
                df = pd.read_parquet(path)
                vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]

                numEpochs = int(input('Enter total number of epochs to train for: '))

                model, results = train.trainNewModel(location, filename, model, optimiser, numEpochs, numBatches, vocabulary, evalFrequency = 5, device = device, saveDir = saveDir)
                utils.visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = False)

                # Add model parameters to results
                results['hiddenDim1Size'] = hiddenDim1Size
                results['hiddenDim2Size'] = hiddenDim2Size
                results['latentDimSize'] = latentDimSize

                # Save results to JSON file
                resultsPath = os.path.join(saveDir, 'results.json')
                os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
                with open(resultsPath, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {resultsPath}")


            elif option == '2':
                location, filename = promptLocationAndFilename()
                hiddenDim1Size, hiddenDim2Size, latentDimSize = promptModelParams()
    
                try:
                    # Check if dataset has been preprocessed
                    path = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
                    df = pd.read_parquet(path)
                except FileNotFoundError:
                    print(f'\nPreprocessed data not found at {path}. Please preprocess data first or supply correct path.')
                    continue

                timestamp = utils.getRunTimestamp()
                saveDir = utils.getModelSaveDir(filename, timestamp)
                os.makedirs(saveDir, exist_ok=True)

                vocabSize = preprocess.getVocabSize(location, filename)
                numBatches = preprocess.getNumBatches(location, filename)

                model, optimiser, device = utils.initialiseModel(vocabSize, hiddenDim1Size, hiddenDim2Size, latentDimSize)

                modelSubfolder = input('Paste the subfolder name (e.g., "bbcnews - train-00000-of-00001-7a59686b1f65c165-20250517_223535"): ')
                modelName = input('Paste the model filename to load (e.g., "tRopicAL-model-epoch-5.pt"): ')
                modelFolder = os.path.join('Saved Models', modelSubfolder)

                model = utils.loadModel(modelFolder, modelName, model, device)

                path = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
                df = pd.read_parquet(path)
                vocabulary = [col for col in df.columns if col not in ['rowId', 'batchId']]


                print('''\nAFTER LOADING CHECKPOINT, CHOOSE:
                1. Continue training
                2. Validate
                3. Test
                4. Run on unseen data
                ''')

                subOption = input('Enter sub-option: ')

                if subOption == '1':
                    epoch = int(input('Enter epoch to training from: '))
                    numEpochs = int(input('Enter total number of epochs to train for: '))

                    model, results = train.resumeTrainingModel(location, filename, model, optimiser, epoch, numEpochs, numBatches, vocabulary, evalFrequency = 5, device = device, saveDir = saveDir)
                    utils.visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = False)

                    # Add model parameters to results
                    results['hiddenDim1Size'] = hiddenDim1Size
                    results['hiddenDim2Size'] = hiddenDim2Size
                    results['latentDimSize'] = latentDimSize

                    # Save results to JSON file
                    resultsPath = os.path.join(saveDir, 'results.json')
                    os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
                    with open(resultsPath, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {resultsPath}")


                elif subOption == '2':
                    model, results = validate.validateModel(location, filename, model, vocabulary, device, numBatches)
                    utils.visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = False)

                    # Add model parameters to results
                    results['hiddenDim1Size'] = hiddenDim1Size
                    results['hiddenDim2Size'] = hiddenDim2Size
                    results['latentDimSize'] = latentDimSize

                    # Save results to JSON file
                    resultsPath = os.path.join(saveDir, 'results.json')
                    os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
                    with open(resultsPath, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {resultsPath}")


                elif subOption == '3':
                    model, results = test.testModel(location, filename, model, vocabulary, device, numBatches)
                    utils.visualiseTopics(filename, model, vocabulary, saveDir, maxLevels = 3, batchedMode = False)

                    # Add model parameters to results
                    results['hiddenDim1Size'] = hiddenDim1Size
                    results['hiddenDim2Size'] = hiddenDim2Size
                    results['latentDimSize'] = latentDimSize

                    # Save results to JSON file
                    resultsPath = os.path.join(saveDir, 'results.json')
                    os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
                    with open(resultsPath, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {resultsPath}")                    
                

                elif subOption == '4':
                    # Run inference on unseen data
                    filename = input('Enter filename: ')
                    
                    # Run topic inference
                    results = modelTopics.modelTopics(
                        location = location,
                        filename = filename,
                        model = model,
                        vocabulary = vocabulary,
                        device = device,
                        saveDir = saveDir
                    )
                    
                    if results:
                        print(f'\nInference complete! Results saved to: {results["saveDir"]}')
                        log.info(f'Model inference on unseen corpus {filename} complete!')
                    else:
                        print('Inference failed. Please check logs for details.')
                        log.error(f'Model inference on {filename} failed')


                else:
                    print('Invalid sub-option.')


            else:
                print('Exiting program...')
                running = False

        except Exception as e:
            print('Something went wrong. Please check your inputs and try again.')
            log.error(f'An error occurred: {e}')  # Fix: Use log object for logging
            continue


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('vader_lexicon')
    logger.setDebugMode(debug = True)

    main()