import tesbml
import requests

FILENUM = 64
URL_BASE = "https://www.ebi.ac.uk/biomodels-main/download?mid=BIOMD"

def run(filenum):
  url = "%s%s" % (URL_BASE, format(filenum, '010d'))
  response = requests.get(url).content
  document = tesbml.readSBMLFromString(response.decode("utf-8"))
  model = document.getModel()
  stg = " "
  for i in range(model.getNumReactions()):
    reaction = model.getReaction(i)
    for j in range(reaction.getNumReactants()):
      reactant = reaction.getReactant(j)
      stg = stg + " + " + reactant.getSpecies()
    stg = stg + " -> "
    for k in range(reaction.getNumProducts()):
      product = reaction.getProduct(k)
      stg = stg + " + " + product.getSpecies()
    stg = stg + ";\n"
  return stg

for filenum in range(550, 580):
  print("\n\n***%d" % filenum)
  try:
    stg = run(filenum)
    print (run(filenum))
  except:
    pass
