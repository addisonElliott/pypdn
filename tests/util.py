import os

dataRoot = os.path.join(os.path.dirname(__file__), 'data')

def getDataPath(path):
    return os.path.join(dataRoot, path)