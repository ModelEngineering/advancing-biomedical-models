import tesbml
import requests

FILENUM = 64
URL_BASE = "https://www.ebi.ac.uk/biomodels-main/download?mid=BIOMD"

def makeURL(filenum=FILENUM):
    """
    Creates the URL for the BioModel number.
    :param int filenum:
    :return url:
    """
    return "%s%s" % (URL_BASE, format(filenum, '010d'))

def makeReactionStrings(filenum=FILENUM, model=None):
  if model is None:
      model = makeModel(filenum)
  stg = " "
  for i in range(model.getNumReactions()):
    reaction = model.getReaction(i)
    for j in range(reaction.getNumReactants()):
      reactant = reaction.getReactant(j)
      stg = stg + " " + reactant.getSpecies()
    stg = stg + " -> "
    for k in range(reaction.getNumProducts()):
      product = reaction.getProduct(k)
      stg = stg + " " + product.getSpecies()
    stg = stg + ";\n"
  return stg

def makeModel(filenum=FILENUM):
    """
    :param str file_num:
    :return libsbml.Model:
    """
    url = makeURL(filenum)
    response = requests.get(url).content
    document = tesbml.readSBMLFromString(response.decode("utf-8"))
    return document.getModel()

if __name__ == "__main__":
  makeReactionStrings()
