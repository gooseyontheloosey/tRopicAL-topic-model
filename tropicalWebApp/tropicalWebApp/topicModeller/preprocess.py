import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import string
import re
import glob

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

from .logger import logger

# Initialise NLTK resources
lemmatiser = WordNetLemmatizer()
stopWords = set(stopwords.words('english')) # set faster than list
punctuation = set(string.punctuation)



def saveAsPARQUET(location, batch, targetFolder, i):
    ''' Saves data in PARQUET file format. '''
    
    # Create path to save to
    targetFolder = os.path.join(location, targetFolder)
    os.makedirs(targetFolder, exist_ok=True) # Ensure target folder exists (create if not)

    # Save sentiment/cleaned batches in separate files // TF-IDF dataframe in one file
    if i == -1:
        df = batch # Simply rename
        path = os.path.join(targetFolder, 'tfidf.parquet')
    else:
        # Convert 'batch' into Dataframe obj
        df = pd.DataFrame(batch)
        path = os.path.join(targetFolder, f'part-{i}.parquet')
    
    # Check if file exists (remove if does)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f'Removed existing file: {path}')
        except Exception as e:
            logger.error(f'Error removing file: {e}')

    # Create Arrow table and save to PARQUET directory
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

    logger.info(f'{path} saved successfully.')
    
    return df # Returning for testing purposes



def sentimentAnalysis(batch, batchId):
    ''' Conducts sentiment analysis on sentences using VADER. '''

    logger.debug(f'SENTIMENT ANALYSIS - processing batch {batchId}...')
    
    analyser = SentimentIntensityAnalyzer()
    sentimentScores = []

    for rowId in range(len(batch)):
        try:
            rowSentimentScores = []
            
            # Iterate through each row of column 'text' in 'batch'
            text = batch['text'].iloc[rowId]

            # Tokenise row into sentences
            sentences = sent_tokenize(text)

            # If no sentences detected but text exists, use whole text as one sentence
            if not sentences and text and not text.isspace():
                sentences = [text]
            
            # Calculate sentiment analysis score for each sentence in row and add to rowSentimentScores
            for sentenceId, sentence in enumerate(sentences):
                if sentence and not sentence.isspace(): # Only process non-empty sentences
                    rowSentimentScores.append({
                    'batchId': batchId,
                    'rowId': rowId,
                    'sentenceId': sentenceId,
                    'sentenceSentimentScore': analyser.polarity_scores(sentence)['compound']})

            # Flatten rowSentimentScores into 1D list
            for score in rowSentimentScores:
                sentimentScores.append(score)
                
        except Exception as e:
            logger.error(f'Error processing row {rowId}: {e}')

    return sentimentScores



def getWordNetTag(tag):
    ''' Helper function which maps NLTK 'pos' tags to WordNet. '''
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    


def clean(batch, batchId):
    ''' Cleans data by removing punctuation, stop words and lemmatising. '''

    logger.debug(f'CLEANING - processing batch {batchId}...')
    
    cleanedRows = []

    moreStopWords = {'one', 'would', 'like', 'use', 'get', 'go', 'say', 'make', 
                        'think', 'know', 'find', 'see', 'look', 'come', 'take',
                        'good', 'time', 'day', 'way', 'thing', 'year'}
    
    extendedStopWords = stopWords.union(moreStopWords)

    for rowId in range(len(batch)):
        try:
            # Iterate through each row of column 'text' in 'batch'
            text = batch['text'].iloc[rowId]

            # Remove punctuation from text
            punctToRemove = ''.join(char for char in string.punctuation)
            text = re.sub(f'[{re.escape(punctToRemove)}]', '', text)

            # Tokenise row of text into individual words
            words = word_tokenize(text)
                
            # Remove stop words and non-alphabet words
            words = [w for w in words if w
                          not in extendedStopWords
                          and w.isalpha()]

            posTags = pos_tag(words) # Tag words as adjectives, nouns, verbs etc. with part-of-speech tagger

            # Apply lemmatisation
            words = [lemmatiser.lemmatize(w, getWordNetTag(t)) for w, t in posTags]

            # Join individual words back together
            cleanedText = ' '.join(words)

            # Only process non-empty rows
            if cleanedText:        
                cleanedRows.append({
                    'batchId': batchId,
                    'rowId': rowId,
                    'text': cleanedText})
        
        except Exception as e:
            logger.error(f'Error processing row {rowId}: {e}')

    return cleanedRows



def tfIDF(dataframe):
    ''' Converts corpus data into numerical representation using TF-IDF. '''

    logger.debug('TF-IDF - processing dataset...')
    
    # Initialise TF-IDF tool with L2 normalisation (ignoring terms that appear in <7 and in >70% of docs)
    vectorizer = TfidfVectorizer(norm = 'l2', min_df = 7, max_df = 0.7, stop_words = 'english')

    # Get TF-IDF matrix of DataFrame obj
    tfIDFMatrix = vectorizer.fit_transform(dataframe['text'])

    # Convert matrix back into DataFrame obj, ensuring batchId and rowId columns preserved
    df = pd.DataFrame(tfIDFMatrix.toarray(), columns = vectorizer.get_feature_names_out())
    df['batchId'] = dataframe['batchId']
    df['rowId'] = dataframe['rowId']

    # Print statistics about the result
    print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')
    print(f'Matrix shape: {tfIDFMatrix.shape}')
    print(f'Matrix sparsity: {100.0 * (1.0 - tfIDFMatrix.count_nonzero() / (tfIDFMatrix.shape[0] * tfIDFMatrix.shape[1]))}%')
    
    # Check if matrix is all zeros
    if tfIDFMatrix.count_nonzero() == 0:
        print('WARNING: TF-IDF matrix contains all zeros!')
        # Show first few tokens in vocabulary to debug
        print('First 10 tokens in vocabulary:', list(vectorizer.get_feature_names_out())[:10])
    else:
        # Show top terms for first few documents
        feature_names = vectorizer.get_feature_names_out()
        print('\nTop 5 terms for first 50 documents:')
        for i in range(min(50, tfIDFMatrix.shape[0])):
            row = tfIDFMatrix[i].toarray()[0]
            top_indices = row.argsort()[-5:][::-1]  # Top 5 terms
            top_terms = [(feature_names[idx], row[idx]) for idx in top_indices if row[idx] > 0]
            print(f'  Doc {i+1}: {top_terms}')
    
    return df, tfIDFMatrix, vectorizer



def getVocabSize(location, filename):
    ''' Gets vocabSize (aka num. of unique words) TF-IDF PARQUET file. '''
    
    path = os.path.join(location, f'TF-IDF_SCORES - {filename} - DATASET', 'tfidf.parquet')
    file = pq.ParquetFile(path)
    columns = file.schema.names
    features = [col for col in columns if col not in ['batchId', 'rowId']]
    
    return len(features)



def getNumBatches(location, filename):
    path = os.path.join(location, f'CLEANED - {filename} - DATASET')
    allFiles = glob.glob(os.path.join(path, '*.parquet'))

    return len(allFiles)



def display_tfidf_results(matrix, vectorizer, dataframe):
    ''' Displays entire TF-IDF matrix. '''

    print('\n', '\n', '\n')
    
    feature_names = vectorizer.get_feature_names_out()
    
    print('\n--- Complete TF-IDF Matrix ---')
    print(f'Matrix shape: {matrix.shape} (documents × terms)')
    print(f'Total features: {len(feature_names)}')
    
    # Convert sparse matrix to dense if needed for full display
    # Warning: This can use a lot of memory for large matrices
    dense_matrix = matrix.toarray()
    
    # Display each document with all its non-zero terms
    for i in range(matrix.shape[0]):
        # Get the document text
        doc_text = dataframe['text'].iloc[i]
        
        # Get non-zero terms for this document
        row = dense_matrix[i]
        non_zero_indices = row.nonzero()[0]
        
        # Get all terms with their weights
        weights = [(feature_names[idx], row[idx]) for idx in non_zero_indices]
        weights.sort(key=lambda x: x[1], reverse=True)
        
        # print(f'\nDocument {i+1}: \'{doc_text}\'')
        # print(f'Number of non-zero terms: {len(weights)}')
        
        if len(weights) > 0:
            print('All terms with non-zero weights:')
            for term, weight in weights:
                print(f'  - {term}: {weight:.4f}')
        else:
            print('  No non-zero terms found for this document.')

        print('\n', '\n', '\n')



def preprocess(location, filename):
    ''' Performs pre-req steps of chosen dataset for topic modelling.
        Processes file in batches and obtains sentiment analysis scores and TF-IDF representation.
    '''
    
    parquetFile = pq.ParquetFile(f'{location}\\{filename}.parquet')
    
    #count = 0

    # Process PARQUET files in batches
    for batchId, thisBatch in enumerate(parquetFile.iter_batches(batch_size = 128)):
        thisBatch = thisBatch.to_pandas()
        thisBatch = thisBatch[['text']] # Only keep text column

        # if count >= 3:
        #      break
        
        ### 1. Apply sentiment analysis
        batchSentimentScores = sentimentAnalysis(thisBatch, batchId)
        saveAsPARQUET(location, batchSentimentScores, f'SENTIMENT_SCORES - {filename} - DATASET', batchId)
    
        ### 2. Clean batch
        cleanedBatch = clean(thisBatch, batchId)
        saveAsPARQUET(location, cleanedBatch, f'CLEANED - {filename} - DATASET', batchId)
        #count += 1
        

    ### 3. Read in all cleaned PARQUET files and convert into DataFrame obj
    path = os.path.join(location, f'CLEANED - {filename} - DATASET')
    allPARQUETFiles = glob.glob(os.path.join(path, '*.parquet'))
    dataframeList = [pd.read_parquet(file) for file in allPARQUETFiles]
    dataframe = pd.concat(dataframeList, ignore_index = True)
    
    
    ### 4. Get TF-IDF of DataFrame obj (aka of entire dataset)
    df, matrix, vectorizer = tfIDF(dataframe)
    saveAsPARQUET(location, df, f'TF-IDF_SCORES - {filename} - DATASET', -1)
    #display_tfidf_results(matrix, vectorizer, dataframe)

    
    vocabSize = len(vectorizer.get_feature_names_out()) # Get num. of unique words in TF-IDF matrix
    numBatches = batchId + 1

    return vocabSize, numBatches