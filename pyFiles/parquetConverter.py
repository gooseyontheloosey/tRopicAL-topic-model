import pandas as pd
import os
import nltk
import re

from docx import Document
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet = True)



def readDocxFile(location, filename):
    ''' Reads a .docx file and returns a list of cleaned sentences. '''
    
    try:
        # Create Document obj
        doc = Document(os.path.join(location, filename))
        
        # Join non-empty paragraphs into single string with spaces
        text = ' '.join([p.text for p in doc.paragraphs if p.text.strip()])

        # Check if document empty
        if not text:
            print(f'Error: {filename} is empty!')
            return False
        
        # Tokenise into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        return sentences
    
    except Exception as e:
        print(f'Error processing file: {e}')
        return False


def convertToPARQUET(location, filename, chunkSize = 3):
    ''' Converts file to PARQUET format, to be used in topic model. 
        Each row contains some sentences with proper spacing. '''
    
    try:
        sentences = readDocxFile(location, filename)
        if not sentences:
            return False

        rows = []
        for i in range(0, len(sentences), chunkSize):
            chunk = sentences[i:i + chunkSize]
            if chunk:  # Only add non-empty chunks
                rows.append(' '.join(chunk))
        
        # Create DataFrame obj and save as PARQUET
        df = pd.DataFrame(rows, columns = ['text'])
        output_filename = filename.replace('.docx', '.parquet')
        df.to_parquet(os.path.join(location, output_filename), index = False)
        print(f'File saved as {output_filename} with {len(rows)} chunks.')
        
        return True
    
    except Exception as e:
        print(f'Error processing file: {e}')
        return False


def convertMultipleToPARQUET(location, filenames, chunkSize = 3, outputName = 'combined.parquet'):
    ''' Converts multiple files into one combined PARQUET file. '''
    
    try:
        all_rows = []
        
        for filename in filenames:
            print(f'\nProcessing {filename}...')
            sentences = readDocxFile(location, filename)
            if not sentences:
                continue
            
            for i in range(0, len(sentences), chunkSize):
                chunk = sentences[i:i + chunkSize]
                if chunk:
                    all_rows.append(' '.join(chunk))
        
        if not all_rows:
            print('No content found in any files!')
            return False
        
        # Create combined DataFrame and save
        df = pd.DataFrame(all_rows, columns = ['text'])
        df.to_parquet(os.path.join(location, outputName), index = False)
        print(f'\nCombined file saved as {outputName} with {len(all_rows)} total chunks.')
        
        return True
    
    except Exception as e:
        print(f'Error processing files: {e}')
        return False


# Example usage when running this script directly
if __name__ == '__main__':
    
    print('DOCX to PARQUET Converter')
    print('-' * 30)
    
    location = input('\nEnter location of file(s) to convert: ')
    
    # Check if location exists
    if not os.path.exists(location):
        print(f'Error: Location {location} does not exist!')
        exit()
    
    # Ask for processing mode
    print('\nChoose mode:')
    print('1. Single file')
    print('2. Multiple files (combined into one)\n')
    mode = input('Enter option: ')
    
    # Get chunk size
    try:
        chunkSize = int(input('\nEnter sentences per chunk (default 3): ') or '3')
    except:
        chunkSize = 3
    
    if mode == '1':
        filename = input('Enter filename (with file extension): ')
        convertToPARQUET(location, filename, chunkSize)
        
    elif mode == '2':
        # Let user specify filenames
        filenames = []
        print('Enter filenames one by one (press Enter with empty input to finish):')
        while True:
            filename = input('Filename: ').strip()
            if not filename:
                break
            if not filename.endswith('.docx'):
                filename += '.docx'
            filenames.append(filename)
        
        if filenames:
            outputName = input('Output filename (default: combined.parquet): ') or 'combined.parquet'
            if not outputName.endswith('.parquet'):
                outputName += '.parquet'
            convertMultipleToPARQUET(location, filenames, chunkSize, outputName)
        else:
            print('No files specified!')
    
    else:
        print('Invalid choice!')