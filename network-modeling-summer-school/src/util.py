import pandas as pd
import urllib.request

# Linear pathway data
BASE_URL = "https://github.com/vporubsky/network-modeling-summer-school/raw/main/"
BASE_DATA_URL = "%sdata/" % BASE_URL
BASE_MODULE_URL = "%ssrc/" % BASE_URL
BASE_MODEL_URL = "%smodels/" % BASE_URL
LOCAL_FILE = "local_file.txt"


def getData(csvFilename):
    """
    Creates a dataframe from a CSV structured URL file.

    Parameters
    ----------
    csvFilename: str
        Name of the CSV file (w/o ".csv" extension)

    Returns
    -------
    pd.DataFrame
    """
    url = "%s%s.csv" % (BASE_DATA_URL, csvFilename)
    filename, _ = urllib.request.urlretrieve(url, filename=LOCAL_FILE)
    return pd.read_csv(LOCAL_FILE)

def getModule(moduleName):
    """
    Obtains common codes from the github repository.
  
    Parameters
    ----------
    moduleName: str
        name of the python module in the src directory
    """
    url = "%s%s.py" % (BASE_MODULE_URL, moduleName)
    _, _ = urllib.request.urlretrieve(url, filename=LOCAL_FILE)
    with open(LOCAL_FILE, "r") as fd:
        codeStr = "".join(fd.readlines())
    return codeStr

def getModel(modelName):
    """
    Creates returns the string for the antimony model.

    Parameters
    ----------
    modelName: str
        Name of the model w/o ".ant"

    Returns
    -------
    str
    """
    url = "%s%s.ant" % (BASE_MODEL_URL, modelName)
    filename, _ = urllib.request.urlretrieve(url, filename=LOCAL_FILE)
    with open(LOCAL_FILE, "r") as fd:
        result = "".join(fd.readlines())
    return result

# Set models
WOLF_MODEL = getModel("wolf")
WOLF_DF = getData("wolf")
WOLF_ARR = WOLF_DF.to_numpy()
LINEAR_PATHWAY_DF = getData("linear_pathway")
LINEAR_PATHWAY_ARR = LINEAR_PATHWAY_DF.to_numpy()
LINEAR_PATHWAY_MODEL = getModel("linear_pathway")
