import gc
import torch
import psutil
from .logger import logger  # Fix: Import logger object directly



def checkMemory(location = 'Unknown', forceGc = False):
    '''Tracks memory usage and forces cleanup if needed.'''
    
    process = psutil.Process()
    
    if torch.cuda.is_available():
        gpuAllocated = torch.cuda.memory_allocated() / (1024**3)
        gpuCached = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f'[{location}] GPU Memory: {gpuAllocated:.2f}GB allocated, {gpuCached:.2f}GB cached')
        
        # Force cleanup if above threshold or requested
        if gpuAllocated > 3.0 or forceGc:  # Adjust threshold as needed
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda and not obj._base:
                        del obj
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            newAllocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f'[{location}] Memory cleanup: {gpuAllocated:.2f}GB → {newAllocated:.2f}GB')
    
    # CPU memory
    ramUsed = process.memory_info().rss / (1024**3)
    logger.info(f'[{location}] RAM usage: {ramUsed:.2f}GB')
    
    # System memory
    systemRam = psutil.virtual_memory()
    logger.info(f'[{location}] System RAM: {systemRam.percent}% used, {systemRam.available/(1024**3):.2f}GB available')
    
    # If system memory is critically low, take emergency measures
    if systemRam.available < 2 * (1024**3):  # Less than 2GB available
        logger.warning('CRITICAL MEMORY PRESSURE - EMERGENCY CLEANUP')
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            # Move model to CPU temporarily if needed
            return True
    
    print('\n')
    return False