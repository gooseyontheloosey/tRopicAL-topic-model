import os
import pandas as pd



def splitPARQUET(location, filename):
    ''' Splits Parquet file into 80% training and 20% validation sets. '''
    
    try:
        # Create new filename for validation set
        valFile = filename.replace('train', 'validate')
        
        # Read 'train' PARQUET file into DataFrame obj
        df = pd.read_parquet(os.path.join(location, filename))

        if df.empty:
            print(f'Error: {filename} is empty!')
            return False
        
        # Shuffle DataFrame obj
        shuffledDf = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split point
        split = int(len(shuffledDf) * 0.8)
        
        # Split DataFrame obj into training and validation sets
        trainDf = shuffledDf.iloc[:split]
        valDf = shuffledDf.iloc[split:]
        
        # Write to PARQUET files (OVERWRITES ORIGINAL TRAIN FILE)
        trainDf.to_parquet(os.path.join(location, filename), index=False)
        valDf.to_parquet(os.path.join(location, valFile), index=False)
        
        print(f'Training set saved succesfully in {location}, w/ {len(trainDf)} rows')
        print(f'Validation set saved succesfully in {location}, w/ {len(valDf)} rows')
    
    except Exception as e:
        print(f'Error processing file: {e}')
        return False


# Example usage when running this script directly
if __name__ == '__main__':

    location = input('Enter location of PARQUET file to split: ')
    filename = input('Enter filename PARQUET file to split: ')
    splitPARQUET(location, filename)