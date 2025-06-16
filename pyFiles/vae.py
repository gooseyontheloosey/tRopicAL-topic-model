import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import logger  # Fix: Import logger object directly
from memory import checkMemory




class VAE(nn.Module):
    def __init__(self, vocabSize, hiddenDimSize1, hiddenDimSize2, latentDimSize):

        '''
        VAE class - conducts the topic modelling.
        
        vocabSize = no. of unique words (AKA no. of column values from tfIDF matrix)
        hiddenDimSize = no. of layers in hidden dimension
        latentDimSize = no. of layers in latent dimension (aka z, size of compressed representation)

        encoder output --> mu and sigma
        mu and sigma sampled to get z, latent vector

        Each encoder is a 3-layer MLP:
        Layer 1 (Linear -> ReLU [1024])
        Layer 2 (Linear -> ReLU [512])
        Layer 3 (mu & logVar -> sampled -> latentVector [200])

        TF-IDF matrix batch [128] -> baseEncoder -> recursiveEncoder -> sentiment + latentVector [201] -> finalEncoder -> decoder                             
        '''
        
        super().__init__()

        # FOR LOGGING PURPOSES
        self.decoderCallCount = 0

        self.latentDimSize = latentDimSize

        self.mostRecentZs = None
        self.allGeneratedZs = []

        self.generatedTopicHierarchies = []

        # 3-layer MLP for sentiment weights
        self.sentimentWeights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hiddenDimSize1),
                nn.ReLU(),
                nn.Linear(hiddenDimSize1, hiddenDimSize2),
                nn.ReLU(),
                nn.Linear(hiddenDimSize2, latentDimSize),
                nn.Sigmoid()
            ) for i in range(3)  # One for each hierarchy level
        ])

        # 3-layer base encoder
        self.baseEncoder = nn.Sequential(
            nn.Linear(vocabSize, hiddenDimSize1), # Layer 1 - Transforms layer linearly, then applies ReLU
            nn.ReLU(),
            nn.Linear(hiddenDimSize1, hiddenDimSize2), # Layer 2 - As above, but input size is now size of previous layer's output
            nn.ReLU()
        )
        self.baseMu, self.baseLogVar = nn.Linear(hiddenDimSize2, latentDimSize), nn.Linear(hiddenDimSize2, latentDimSize) # Layer 3

        # 3-layer recursive encoder
        self.recursiveEncoder = nn.Sequential(
            nn.Linear(latentDimSize, hiddenDimSize1),
            nn.ReLU(),
            nn.Linear(hiddenDimSize1, hiddenDimSize2),
            nn.ReLU()
        )
        self.recursiveMu, self.recursiveLogVar = nn.Linear(hiddenDimSize2, latentDimSize), nn.Linear(hiddenDimSize2, latentDimSize) # Layer 3

        # 3-layer final encoder
        self.finalEncoder = nn.Sequential(
            nn.Linear(latentDimSize + 1, hiddenDimSize1),
            nn.ReLU(),
            nn.Linear(hiddenDimSize1, hiddenDimSize2),
            nn.ReLU()
        )
        self.finalMu, self.finalLogVar = nn.Linear(hiddenDimSize2, latentDimSize), nn.Linear(hiddenDimSize2, latentDimSize) # Layer 3

        # 3-layer decoder
        self.decoder = nn.Sequential(
            nn.Linear(latentDimSize, hiddenDimSize2),
            nn.ReLU(),
            nn.Linear(hiddenDimSize2, hiddenDimSize1),
            nn.ReLU(),
            nn.Linear(hiddenDimSize1, vocabSize),
            nn.Sigmoid()
        )
            
        
    def baseEncoderTfIdf(self, tfIdf):
        logger.debug('In baseEncoderTfIdf')


        x = self.baseEncoder(tfIdf)
        mu = self.baseMu(x)
        logVar = self.baseLogVar(x)
        z = self.reparameterise(mu, logVar) 

        return mu, logVar, z
        

    def recursiveEncoderTfIdfWithSentiment(self, latentVector, sentimentScoresByRow):
        logger.debug('In recursiveEncoderTfIdfWithSentiment')

        muList = []
        logVarList = []
        zList = [latentVector]
        currentZ = latentVector

        for rLevel in range(0, 3): # Recursion depth 3
            sentimentWeightsByRow = []

            # Loop through each row of sentiment snalysis scores per batch
            for row, scores in enumerate(sentimentScoresByRow):
                sentimentWeightsBySentence = []

                # For empty rows, use neutral sentiment weight
                if len(scores) == 0:
                    sentimentWeightsByRow.append(torch.ones_like(currentZ[row]))
                    continue
                    
                # Otherwise, loop through each sentence per row and assign it a weight
                for s in range(len(scores)):
                    sentiment = scores[s].view(1, 1)
                    weight = self.sentimentWeights[rLevel](sentiment)
                    sentimentWeightsBySentence.append(weight)

                # Stack all sentence weights up
                stackedWeights = torch.cat(sentimentWeightsBySentence, dim = 0)

                # Calculate overall row weight (deeper recursion = larger sentiment weights)
                if rLevel == 0:
                    rowSentimentWeight = torch.mean(stackedWeights, dim = 0)
                elif rLevel == 1:
                    sentimentStrength = torch.abs(scores)
                    sentimentStrengthWeight = F.softmax(sentimentStrength, dim = 0).unsqueeze(1)
                    rowSentimentWeight = torch.sum(stackedWeights * sentimentStrengthWeight, dim = 0)
                elif rLevel == 2:
                    sentimentStrength = torch.abs(scores - 0.5) * 2
                    sentimentStrengthWeight = F.softmax(sentimentStrength, dim = 0).unsqueeze(1)
                    rowSentimentWeight = torch.sum(stackedWeights * sentimentStrengthWeight, dim = 0)

                sentimentWeightsByRow.append(rowSentimentWeight)

            # Stack all row weights up
            batchWeights = torch.stack(sentimentWeightsByRow)

            sF = 1.0 + (rLevel * 0.5) # Scale factor
            batchWeights = 1.0 + (batchWeights - 0.5) * sF # Scale batch weights

            currentZ *= batchWeights # Obtain latent vector weighted by batch sentiment
            
            x = self.recursiveEncoder(currentZ)
            mu = self.recursiveMu(x)
            logVar = self.recursiveLogVar(x)
            newLatentVector = self.reparameterise(mu, logVar)

            muList.append(mu)
            logVarList.append(logVar)
            zList.append(newLatentVector)
            currentZ = newLatentVector # Update current latent vector for next recursion level

        return muList, logVarList, zList

        
    def finalEncoderSentiment(self, latentVector, sentimentTensor):
        logger.debug('In finalEncoderSentiment')

        sentimentScore = sentimentTensor.view(-1, 1)
        combined = torch.cat([latentVector, sentimentScore], dim = 1)

        x = self.finalEncoder(combined)
        mu  = self.finalMu(x)
        logVar = self.finalLogVar(x)
        z = self.reparameterise(mu, logVar)

        return mu, logVar, z

    
    def reparameterise(self, mu, logVar):
        '''
        Applies reparamterisation trick.
        
               z      =  mu  +       sigma        *   epsilon
        latent vector = mean + standard deviation * random noise
        '''
        
        sigma = torch.exp(0.5 * logVar)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon

        return z

    
    def finalDecoder(self, latentVector):
        # Only log occasionally to avoid flooding
        self.decoderCallCount += 1
        if self.decoderCallCount % 100 == 0:  # Log every 100th call
            logger.debug(f'In finalDecoder (call #{self.decoderCallCount})')
            gc.collect()  # Force garbage collection to free up memory
            torch.cuda.empty_cache()
            logger.debug('Garbage collection and CUDA cache cleared.')
            
        return self.decoder(latentVector)

    
    def feedforward(self, tfIdf, sentimentScoresByRow):
        checkMemory('feedforward start')
        
        # Run encoders
        baseMu, baseLogVar, baseZ = self.baseEncoderTfIdf(tfIdf)
        
        recursiveMus, recursiveLogVars, recursiveZs = self.recursiveEncoderTfIdfWithSentiment(baseZ, sentimentScoresByRow)

        # Calculate average sentiment score per batch
        batchSentiment = torch.tensor([
            torch.mean(sentences).item() if len(sentences) > 0
            else 0.5 # Use 0.5 if no sentences in batch
            for sentences in sentimentScoresByRow
        ], device = tfIdf.device) # Ensure tensors all stored in same location
        
        finalMu, finalLogVar, finalZ = self.finalEncoderSentiment(recursiveZs[-1], batchSentiment)

        # Run decoder
        reconstruction = self.decoder(finalZ)

        # Collect latent representations
        allMus = [baseMu] + recursiveMus + [finalMu]
        allLogVars = [baseLogVar] + recursiveLogVars + [finalLogVar]
        allZs = [baseZ] + recursiveZs + [finalZ]

        self.mostRecentZs = allZs # Save generated latent vectors 
        self.allGeneratedZs.append(allZs) # Save all generated latent vectors
        
        # Clean up any intermediate tensors
        del baseZ
        checkMemory('feedforward end')

        return reconstruction, allMus, allLogVars, allZs

    
    #def backpropagate():
        #pass
        # pyTorch has its own backpropagation function


    def calculateLoss(self, originalInput, reconstruction, allMus, allLogVars, allZs, klWeight):
        '''
        Calculates model loss.
        
        VAE loss = reconstruction loss + KL divergence
        '''
        
        reconstructionLoss = F.binary_cross_entropy(reconstruction, originalInput, reduction = 'sum')

        klLoss = 0
        for mu, logVar in zip(allMus, allLogVars):
            klDivergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            klLoss += klDivergence

        totalLoss = reconstructionLoss + klLoss * klWeight

        return reconstructionLoss, klLoss, totalLoss


    def recordTopicHierarchies(self, topK = 5, vocab = None, batchedMode = True):
        ''' Returns a list of topic hierarchies produced, one per row in batch.
            Each hierarchy is a 2D list: [level0_words, level1_words, ...].
            Each level contains topK words.

            ---

            Zs = [baseTensor, recursive1Tensor, recursive2Tensor, recursive3Tensor, ..., finalTensor]
            --Tensor = [latentVecList0, latentVecList1, latentVecList2, ...] (lists of latent vectors per batch)
            latentVecListN = [z_N_0, z_N_1, ..., z_N_{latentDimSize-1}]] (latent vectors per row)
            Each latent vector must be decoded to a word distribution.
        '''
        logger.debug('In recordTopicHierarchies')


        # Process either all generated Zs or just the most recent Zs
        if not batchedMode:
            
            # Need to flatten self.allGeneratedZs to process all batches together
            numLevels = len(self.allGeneratedZs[0]) # aka num. of levels in each topic hierarchy
            Zs = []

            for level in range(numLevels):
                # Concatenate all batches for this level
                levelTensors = [batch[level] for batch in self.allGeneratedZs]
                Zs.append(torch.cat(levelTensors, dim = 0))
        else:
            Zs = self.mostRecentZs       

        # Collect top words for each level in hierarchy
        topicsByLevel = []

        # Process each level in latent space
        for level, latent in enumerate(Zs):
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)

            # Decode to word distribution
            wordDist = self.finalDecoder(latent)

            # Get probabilities and positions of top-k words per row
            if wordDist.dim() == 1:
                wordDist = wordDist.unsqueeze(0)
            _, positions = torch.topk(wordDist, k=topK, dim=1)


            topicsInLevel = positions.detach().cpu().tolist()

            if vocab is not None:
                topicsInLevel = [[vocab[idx] for idx in row] for row in topicsInLevel]
            topicsByLevel.append(topicsInLevel)

        # Organise by rows
        numRows = len(topicsByLevel[0]) # aka batch size
        numLevels = len(topicsByLevel) # aka num. of levels in each topic hierarchy

        individualTopicHierarchies = []

        for row in range(numRows):
            topicHierarchy = []
            for level in range(numLevels):
                topicHierarchy.append(topicsByLevel[level][row])

            individualTopicHierarchies.append(topicHierarchy)

        self.generatedTopicHierarchies.append(individualTopicHierarchies)
        return individualTopicHierarchies


    def generateHierarchicalLatents(self, sentimentScoresByRow):

        device = next(self.parameters()).device
        
        # Initialise with a random latent vector
        baseZ = torch.randn(1, self.latentDimSize).to(device)
        
        # Process through recursive encoder with sentiment scores
        recursiveMus, recursiveLogVars, recursiveZs = self.recursiveEncoderTfIdfWithSentiment(baseZ, sentimentScoresByRow)
        
        # Calculate batch sentiment (average of sentiments)
        batchSentiment = torch.tensor([
            torch.mean(sentences).item() if len(sentences) > 0 
            else 0.5 # Use 0.5 if no sentences
            for sentences in sentimentScoresByRow
        ], device=device)
        
        # Process through final encoder
        finalMu, finalLogVar, finalZ = self.finalEncoderSentiment(recursiveZs[-1], batchSentiment)
        
        # Collect all latent representations
        allZs = [baseZ] + recursiveZs + [finalZ]
        
        return allZs