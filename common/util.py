"""General Utlities"""

import codecs
import os
import pandas as pd
import string
import tellurium as te
from scipy.special import comb
from SBMLLint.tools.sbmllint import lint


DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DIR)
LAB_DIR = os.path.join(PROJECT_DIR, "labs")
MODEL_DIR = os.path.join(LAB_DIR, "models")
# Column names
C_XML_FILE = "xml_file"
C_NUM_REACTION = "num_reaction"
C_NUM_SPECIES = "num_species"
#
XML_FILES = [m for m in os.listdir(MODEL_DIR) if m[-3:] == "xml"]


def makeAsciiFile(in_path, out_path):
    """
    Removes non-ascii characters from the file.
    
    Parameters
    ----------
    in_path: str
        path to input file
    out_path: str
        path to output file
    """
    with codecs.open(in_path, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    printable = set(string.printable)
    newLines = []
    for line in lines:
        newLines.append("".join([c for c in line if c in printable]))
    with open(out_path, "w") as fd:
        fd.writelines(newLines)

def calcNumberExperiment(numFactor, numLevel, numWay):
    """
    Calculates the number of experiments for a design.

    Parameters
    ----------
    numFactor: int
        Number of factors in the design
    numLevel: int
        number of levels for each factor (without baseline)
    numWay: int
        n-way design
    
    Returns
    -------
    int
    """
    if numWay == 0:
        return 1
    result = calcNumberExperiment(numFactor, numLevel, numWay-1)
    result += int(comb(numFactor, numWay, exact=False)*(numLevel**numWay))
    return result

def getModel(modelId, modelDir=MODEL_DIR):
    """
    Parameters
    ----------
    modelId: str/int
        name of SBML xml file in models directory
    
    Returns
    -------
    str: Antimony Model
    ExtendedRoadrunner: roadrunner model
    """
    if isinstance(modelId, int):
        modelId = "BIOMD0000000%03d.xml" % modelId
    path = os.path.join(MODEL_DIR, modelId)
    rr = te.loadSBMLModel(path)
    modelStr = rr.getAntimony()
    return modelStr, rr

def reportModelStatistics(modelIds):
    """
    Reports various model statistics.

    Parameters
    ----------
    modelIds: list-str/int
        names of SBML xml file in models directory

    Returns
    -------
    pd.DataFrame
    """
    dct = {C_XML_FILE: [], C_NUM_REACTION: [], C_NUM_SPECIES: []}
    for modelId in modelIds:
        _, rr = getModel(modelId)
        dct[C_XML_FILE].append(modelId)
        dct[C_NUM_REACTION].append(rr.getNumReactions())
        dct[C_NUM_SPECIES].append(rr.getNumFloatingSpecies())
    df = pd.DataFrame(dct)
    df = df.sort_values(C_NUM_REACTION)
    df = df.set_index(C_XML_FILE)
    return df

def checkMassBalance(xmlFile):
    try:
        modelStr, _ = getModel(xmlFile)
        _ = lint(modelStr, mass_balance_check="games")
    except Exception as err:
        print("***Error in file %s" % xmlFile)

   
if __name__ == '__main__':
    # calcNumberExperiment
    assert(calcNumberExperiment(5, 10, 1) == 51)
    assert(calcNumberExperiment(3,  2, 2) == 19)
    # getModel
    for modelId in [52, 452]:
        modelStr, rr = getModel(modelId)
        assert(isinstance(modelStr, str))
        assert(isinstance(rr,
              te.roadrunner.extended_roadrunner.ExtendedRoadRunner))
    # Test reportModelStatistics
    size = 5
    df = reportModelStatistics(XML_FILES[0:size])
    assert(len(df) == size)
    # Tests
    checkMassBalance(XML_FILES[0])
    #
    print ("OK!")
