import os 

def getResourcePath(filename): 
    '''
    Returns an absolute path to the resource specified by filename. 
    '''
    return os.path.join(os.path.dirname(__file__), filename)
