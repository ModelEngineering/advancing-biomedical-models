"""General Utlities"""

import codecs
import string
from scipy.special import comb


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
   


if __name__ == '__main__':
    assert(calcNumberExperiment(5, 10, 1) == 51)
    assert(calcNumberExperiment(3,  2, 2) == 19)
    print ("OK!")
