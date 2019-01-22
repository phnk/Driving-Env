'''
Provides system independent module path for resources.  
'''
import os 

def getResourcePath(filename): 
    '''
    Returns an absolute path to the resource specified by filename. 

    Parameters
    ----------
    filename : str 
        The filename under the resources module (directory). 

    Returns
    -------
    str
        A system independent absolute path to the resource.
    '''
    return os.path.join(os.path.dirname(__file__), filename)
