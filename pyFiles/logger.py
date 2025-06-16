import logging



# Create logger
logger = logging.getLogger('tRopicAL')
logger.setLevel(logging.INFO)

# Create console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create simple formatter
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console)


def setDebugMode(debug = False):
    ''' Toggle debug mode on/off '''
    
    if debug:
        logger.setLevel(logging.DEBUG)
        console.setLevel(logging.DEBUG)
        logger.debug('Debug logging enabled')
    else:
        logger.setLevel(logging.INFO)
        console.setLevel(logging.INFO)


# Export logger for direct access in other modules
__all__ = ['logger', 'setDebugMode']