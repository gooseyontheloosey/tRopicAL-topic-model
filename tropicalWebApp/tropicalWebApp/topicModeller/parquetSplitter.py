import os
import pandas as pd



def splitPARQUET(location, filename):
    ''' Splits Parquet file into 80% training and 20% validation sets. '''
    
    try:
        print(f'[splitPARQUET] Starting split for {filename} in {location}')
        
        # Create new filename for validation set
        valFile = filename.replace('train', 'validate')
        print(f'[splitPARQUET] Validation file will be: {valFile}')
        
        # Full paths
        trainPath = os.path.join(location, filename)
        valPath = os.path.join(location, valFile)
        
        print(f'[splitPARQUET] Training path: {trainPath}')
        print(f'[splitPARQUET] Validation path: {valPath}')
        
        # Check if file exists
        if not os.path.exists(trainPath):
            print(f'[splitPARQUET] ERROR: File does not exist: {trainPath}')
            return False
        
        # Read 'train' PARQUET file into DataFrame obj
        print(f'[splitPARQUET] Reading file...')
        df = pd.read_parquet(trainPath)
        print(f'[splitPARQUET] File loaded, shape: {df.shape}')

        if df.empty:
            print(f'[splitPARQUET] ERROR: {filename} is empty!')
            return False
        
        # Shuffle DataFrame obj
        print(f'[splitPARQUET] Shuffling data...')
        shuffledDf = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split point
        split = int(len(shuffledDf) * 0.8)
        print(f'[splitPARQUET] Split point: {split} (80% of {len(shuffledDf)})')
        
        # Split DataFrame obj into training and validation sets
        trainDf = shuffledDf.iloc[:split]
        valDf = shuffledDf.iloc[split:]
        
        print(f'[splitPARQUET] Training set: {len(trainDf)} rows')
        print(f'[splitPARQUET] Validation set: {len(valDf)} rows')
        
        # Write to PARQUET files (OVERWRITES ORIGINAL TRAIN FILE)
        print(f'[splitPARQUET] Saving training file...')
        trainDf.to_parquet(trainPath, index=False)
        
        print(f'[splitPARQUET] Saving validation file...')
        valDf.to_parquet(valPath, index=False)
        
        # Verify files were created
        if os.path.exists(trainPath) and os.path.exists(valPath):
            print(f'[splitPARQUET] SUCCESS: Both files created successfully')
            print(f'[splitPARQUET] Training set saved successfully in {location}, w/ {len(trainDf)} rows')
            print(f'[splitPARQUET] Validation set saved successfully in {location}, w/ {len(valDf)} rows')
            return True
        else:
            print(f'[splitPARQUET] ERROR: Files not found after writing')
            return False
    
    except Exception as e:
        print(f'[splitPARQUET] ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False


# Example usage when running this script directly
if __name__ == '__main__':
    location = input('Enter location of PARQUET file to split: ')
    filename = input('Enter filename PARQUET file to split: ')
    splitPARQUET(location, filename)