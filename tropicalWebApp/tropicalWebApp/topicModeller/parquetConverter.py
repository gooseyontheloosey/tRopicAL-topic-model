import os
from docx import Document
import pandas as pd


def readDocxFile(filePath):
    ''' Reads a .docx file and returns a list of paragraphs. '''
    
    try:
        # Create Document obj
        doc = Document(filePath)
        
        # Extract paragraphs from Document obj
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != '']
        
        # Check if document empty
        if not paragraphs:
            print(f'Warning: {os.path.basename(filePath)} is empty!')
            return []
        
        return paragraphs
    
    except Exception as e:
        print(f'Error processing file {os.path.basename(filePath)}: {e}')
        return []


def convertDocxToParquet(inputPath, outputPath):
    ''' Converts .docx file to PARQUET format, to be used in topic model. '''
    
    try:
        paragraphs = readDocxFile(inputPath)
        
        if not paragraphs:
            return False
        
        # Create DataFrame with text column
        df = pd.DataFrame(paragraphs, columns=['text'])
        
        # Save as parquet
        df.to_parquet(outputPath, index=False)
        print(f'File saved as PARQUET successfully: {outputPath}')
        return True
    
    except Exception as e:
        print(f'Error processing file: {e}')
        return False


def convertUploadedFile(uploadedFile, outputDir):
    ''' Converts uploaded .docx file to parquet format in specified directory. '''
    
    try:
        # Ensure output directory exists
        os.makedirs(outputDir, exist_ok=True)
        
        # Generate output filename
        baseName = os.path.splitext(uploadedFile.name)[0]
        outputPath = os.path.join(outputDir, f'{baseName}.parquet')
        
        # Read the uploaded file
        if hasattr(uploadedFile, 'temporary_file_path'):
            # File is stored temporarily on disk
            inputPath = uploadedFile.temporary_file_path()
        else:
            # File is in memory, save it temporarily
            tempPath = os.path.join(outputDir, f'temp_{uploadedFile.name}')
            with open(tempPath, 'wb') as f:
                for chunk in uploadedFile.chunks():
                    f.write(chunk)
            inputPath = tempPath
        
        # Convert to parquet
        success = convertDocxToParquet(inputPath, outputPath)
        
        # Clean up temporary file if created
        if not hasattr(uploadedFile, 'temporary_file_path') and os.path.exists(inputPath):
            os.remove(inputPath)
        
        if success:
            return outputPath
        else:
            return None
            
    except Exception as e:
        print(f'Error converting uploaded file: {e}')
        return None


def convertMultipleDocxFiles(uploadedFiles, outputDir, datasetName):
    ''' Converts multiple uploaded .docx files to a single combined parquet file. '''
    
    try:
        # Ensure output directory exists
        os.makedirs(outputDir, exist_ok=True)
        
        allParagraphs = []
        tempFiles = []
        
        # Process each uploaded file
        for uploadedFile in uploadedFiles:
            try:
                # Save uploaded file temporarily if needed
                if hasattr(uploadedFile, 'temporary_file_path'):
                    inputPath = uploadedFile.temporary_file_path()
                else:
                    tempPath = os.path.join(outputDir, f'temp_{uploadedFile.name}')
                    with open(tempPath, 'wb') as f:
                        for chunk in uploadedFile.chunks():
                            f.write(chunk)
                    inputPath = tempPath
                    tempFiles.append(tempPath)
                
                # Read paragraphs from this file
                paragraphs = readDocxFile(inputPath)
                allParagraphs.extend(paragraphs)
                
                print(f'Processed {uploadedFile.name}: {len(paragraphs)} paragraphs')
                
            except Exception as e:
                print(f'Error processing file {uploadedFile.name}: {e}')
                continue
        
        # Clean up temporary files
        for tempFile in tempFiles:
            if os.path.exists(tempFile):
                os.remove(tempFile)
        
        if not allParagraphs:
            print('No valid paragraphs found in any of the uploaded files.')
            return None
        
        # Create combined DataFrame
        df = pd.DataFrame(allParagraphs, columns=['text'])
        
        # Generate output filename
        outputPath = os.path.join(outputDir, f'{datasetName}.parquet')
        
        # Save combined parquet file
        df.to_parquet(outputPath, index=False)
        print(f'Combined {len(uploadedFiles)} DOCX files into {outputPath} with {len(allParagraphs)} total paragraphs')
        
        return outputPath
        
    except Exception as e:
        print(f'Error converting multiple DOCX files: {e}')
        return None