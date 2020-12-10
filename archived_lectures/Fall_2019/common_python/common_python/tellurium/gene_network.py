"""
Constructs a gene network for the modeling game.

A gene is described by a descriptor string. There are
three cases.
1. 0 TF: g
2. 1 TF: gsp
3. 2 TF: gspisp

where: 
  g is the gene number
  i indicates the kind of integration:
      A for AND with non-competitive binding (impossible to have AND if competitive)
      O for OR with non-competitive binding
      P for OR with competitive binding
  s is either "+" or "-" to indicate that the protein activates
    or inhibits the gene product
  p is a protein number

The module is implemented as follows:

  GeneDescriptor implements a representation of the
  transcription factors (proteins) that activate/inhibit a gene.

  GeneReaction constructions the mRNA reaction for a gene,
  along with providing the constants in the reaction.

  GeneNetwork generates the entire gene network for the game.
"""

import constants as cn
import model_fitting as mf
import modeling_game as mg
import util

from collections import namedtuple
import copy
import numpy as np
import os
import pandas as pd

NUM_GENE = 8  # Number of genes
PLUS = "+"
# Initial network from the modeling game
INITIAL_NETWORK = [
    "1+0O+4", "2+4", "3+6", "4-2", 5, "6+7A-1", 7, "8-1"]
# Structure of gene string descriptor
POS_GENE = 0
POS_SIGN_1 = 1
POS_PROTEIN_1 = 2

# Initial arcs in reaction network
P0 = "INPUT"  # Protein 0 is the input
IDX_L = 0
IDX_DMRNA = 1

# Indexed by (is_competitive, is_or_integration)
CONJUNCTION_DICT = {
  (True, True): "P",
  (False, True): "O",
  (False, False): "A",
  }


################ Helper Functions ###################
def _setList(values):
  if values is None:
    return []
  else:
    return values

def _equals(list1, list2):
  diff = set(list1).symmetric_difference(list2)
  return len(diff) == 0

def _extendUnique(list1, list2):
  list1.extend(list2)
  return list(set(list1))


################ Classes ###################
class GeneDescriptor(object):
  # Describes the transcription factors (protein) for a gene
  # and how they affect the gene's activation

  def __init__(self, ngene, nprots=None, 
      is_activates=None, is_or_integration=True,
      is_competitive=False):
    """
    :param int ngene: number of the gene
    :param list-int nprots: list of protein TFs
    :param list-bool is_activates: list of how corresponding prot impacts gene
    :param bool is_or_integration: logic used to combine terms
    :param bool is_competitive binding:
    """
    self.ngene = int(ngene)
    self.nprots = [int(p) for p in _setList(nprots)]
    self.is_activates = _setList(is_activates)
    self.is_or_integration = is_or_integration
    self.is_competitive = is_competitive

  def equals(self, other):
    result = True
    result = result and (self.ngene == other.ngene)
    result = result and _equals(self.nprots, other.nprots)
    result = result and _equals(self.is_activates, other.is_activates)
    result = result and (
        self.is_or_integration == other.is_or_integration)
    return result

  def __repr__(self):
    def makeTerm(is_activate, nprot):
      if is_activate:
        sign = "+"
      else:
        sign = "-"
      stg = "%s%d" % (sign, nprot)
      return stg
    #
    stg = str(self.ngene)
    if len(self.nprots) > 0:
      stg += makeTerm(self.is_activates[0], self.nprots[0])
    if len(self.nprots) == 2:
      conjunction = CONJUNCTION_DICT[
          (self.is_competitive, self.is_or_integration)]
      stg += conjunction
      stg += makeTerm(self.is_activates[1], self.nprots[1])
    return stg
      

  @classmethod
  def parse(cls, string):
    """
    Parses a descriptor string (as described in the module
    comments).
    :param str string: gene descriptor string
    :return GeneDescriptor:
    """
    if isinstance(string, int):
      string = str(string)
    if not len(string) in [1, 3, 6]:
      raise ValueError("Invalid string descriptor: %s" % string)
    # Initializations
    string = str(string)  # With 0 TFs, may have an int
    nprots = []
    is_activates = []
    is_or_integration = True
    is_competitive = True
    #
    def extractTF(stg):
      if stg[0] == "+":
        is_activate = True
      else:
        is_activate = False
      nprots.append(int(stg[1]))
      is_activates.append(is_activate)     
    # Extract gene
    ngene = int(string[0])
    #
    if len(string) >= 3:
      extractTF(string[1:3])
    if len(string) == 6:
      conjunction_term = string[3]
      if not conjunction_term in CONJUNCTION_DICT.values():
        raise ValueError("Invalid integration term in descriptor: %s" % string)
      if (conjunction_term == "O") or (conjunction_term == "P"):
        is_or_integration = True
      else:
        is_or_integration = False
      if (conjunction_term == "P"):
        is_competitive = True
      else:
        is_competitive = False
      extractTF(string[4:6])
    #
    return GeneDescriptor(
        ngene,
        is_or_integration=is_or_integration,
        nprots=nprots,
        is_activates=is_activates,
        is_competitive=is_competitive)


######################################################
class GeneReaction(object):
  """Creates the reaction for gene production of mRNA."""
    
  def __init__(self, descriptor, is_or_integration=True):
    """
    :param int/GeneDescriptor ngene: Gene descriptor or int (if default)
    :param bool is_or_integration: logic used to combine terms
    """
    if not isinstance(descriptor, GeneDescriptor):
      ngene = descriptor
      descriptor = GeneDescriptor(ngene,
          is_or_integration=is_or_integration)
    # Public properties
    self.descriptor = descriptor
    self.constants = [
        self._makeVar("L"),          # IDX_L
        self._makeVar("d_mRNA"),     # IDX_DMRNA
        ]
    self.mrna_reaction = None
    self.mrna_kinetics = None
    self.protein_kinetics = None
    # Private
    self._k_index = 0  # Index of the K constants
    self._H = self._makeVar("H")  # H constant
    self._Vm = self._makeVar("Vm")  # H constant
    self._mrna = self.__class__.makeMrna(self.descriptor.ngene)
      
  def add(self, nprot, is_activate=True):
    """
    Adds a protein to the mRNA generation reaction for this gene.
    :param int nprot: numbers of the protein that is TF for the gene
    :param list-bool is_activation: whether the gene is activation (True)
    """
    nprots = list(set(self.descriptor.nprots).union([nprot]))
    is_activates = list(set(self.descriptor.is_activates).union(
        [is_activate]))
    self.descriptor = GeneDescriptor(
      self.descriptor.ngene,
      nprots=nprots,
      is_activates=is_activates,
      is_or_integration=self.descriptor.is_or_integration,
      )
      
  def _makeVar(self, name):
    return self.__class__.makeVar(name, self.descriptor.ngene)
     
  @classmethod
  def makeVar(cls, name, ngene):
    return "%s%d" % (name, ngene)
     
  @classmethod 
  def makeMrna(cls, ngene):
    return cls.makeVar("mRNA", ngene)
     
  @classmethod
  def makeProtein(cls, ngene):
    return cls.makeVar("P", ngene)

  def _makeKVar(self):
    self._k_index += 1
    var = "K%d_%d" % (self._k_index, self.descriptor.ngene)
    self.constants.append(var)
    return var

  @staticmethod
  def _makePVar(nprot):
    if nprot == 0:
      stg = "INPUT"
    else:
      stg = "P%d" % nprot
    return stg

  def _makeMrnaBasicKinetics(self):
    return "%s - %s*%s" % (
        self.constants[IDX_L], 
        self.constants[IDX_DMRNA],
        self._mrna)

  def _makeProteinKinetics(self):
    return "%s*%s - %s*%s" % (
        self._makeVar("a_protein"),
        self._mrna,
        self._makeVar("d_protein"),
        self._makeVar("P")
        )

  def _makeTerm(self, nprots):
    """
    Creates the term Km_n*(Pi*Pj)^Hm
    :param list-int nprots:
    """
    term = "%s" % self._makeKVar()
    for nprot in nprots:
      term = term + "*%s^%s" % (GeneReaction._makePVar(nprot),
          self._H)
    return term

  def _makeNumerator(self, terms):
    """
    Constructs the numerator of a kinetics expression.
    :param list-str terms:
    :return str: numerator
    """
    # Numerator patters. The key is is_activate values for
    # terms and the corresponding numerator
    # The dictionaries are keyed by the tuple of values
    # of is_activate[0] for T0 and is_activates[1] for T1.
    # [T0, T1, T2] = terms
    AND_INTEGRATION = {
        (True, True): "T2",
        (True, False): "T0",
        (False, True): "T1",
        (False, False): "1",
        }
    # OR integration for competitive binding
    OR_INTEGRATION_COMPETE = {
        (True, True): "T0 + T1",
        (True, False): "1 + T0",
        (False, True): "1 + T1",
        (False, False): "1 + T0 + T1",  # Note - always True
        }
    # OR integration for non-competitive binding
    OR_INTEGRATION_NOCOMPETE = {
        (True, True): "T0 + T1 + T2",
        (True, False): "1 + T0 + T2",
        (False, True): "1 + T1 + T2",
        (False, False): "1 + T0 + T1",
        }
    # No terms
    if len(terms) == 0:
      raise RuntimeError("No terms present.")
    # 1 term
    elif len(terms) == 1:
      if self.descriptor.is_activates[0]:
        numerator = terms[0]
      else:
        numerator = "1"
    # 2 or 3 terms
    elif len(terms) <= 3:
      # Initialize values
      key = (self.descriptor.is_activates[0],
          self.descriptor.is_activates[1])
      if len(terms) == 2:
        T0, T1 = terms
      else:
        T0, T1, T2 = terms
      # Obtain the numerator pattern
      if self.descriptor.is_or_integration:
        if self.descriptor.is_competitive:
          pattern = OR_INTEGRATION_COMPETE[key]
        else:
          pattern = OR_INTEGRATION_NOCOMPETE[key]
      else:
        pattern = AND_INTEGRATION[key]
      # Create the numerator
      numerator = pattern.replace("T0", T0)
      numerator = numerator.replace("T1", T1)
      if len(terms) == 3:
        numerator = numerator.replace("T2", T2)
    return numerator

  def _makeTerms(self):
    terms = [self._makeTerm([p]) for p in self.descriptor.nprots]
    if (len(self.descriptor.nprots) > 1) and (
        not self.descriptor.is_competitive):
      new_term = self._makeTerm(self.descriptor.nprots)
      terms.append(new_term)
    return terms

  def _makeTFKinetics(self):
    """
    Creates the kinetics for the transcription factors
    """
    if len(self.descriptor.nprots) == 0:
      return ""
    terms = self._makeTerms()
    # Make the denominator
    denominator = "1"
    for term in terms:
      denominator += " + %s" % term
    numerator = self._makeNumerator(terms)
    # Clean up the numerator by removing leading "+"
    splits = numerator.split(" ")
    if splits[0] == PLUS:
      numerator = " ".join(splits[1:])
    result = "%s * ( %s ) / ( %s )" % (
        self._Vm, numerator, denominator)
    return result
  
  def _makeMrnaKinetics(self):
    """
    Constructs the kinetics expression for the mRNA.
    :return str:
    Updates
      self.constants
      self.mrna_kinetics
    """
    if len(self.descriptor.nprots) == 0:
      stg = self._makeMrnaBasicKinetics()
    else:
      stg = "%s + %s" % (self._makeMrnaBasicKinetics(),
          self._makeTFKinetics())
      self.constants.extend([
          self._Vm,
          self._H,
          ])
    self.mrna_kinetics = stg

  def generate(self):
    """
    Generates the reaction string.
    Updates:
      self.mrna_reaction
      self.protein_kinetics
    """
    label = self._makeVar("J")
    self._makeMrnaKinetics()
    self.protein_kinetics = self._makeProteinKinetics()
    self.mrna_reaction = "%s: => %s; %s" % (label, 
        self._mrna, self.mrna_kinetics)
    
  def __repr__(self):
    if self.mrna_reaction is None:
      self.generate()
    return self.mrna_reaction

  @classmethod
  def do(cls, descriptor):
    """
    Constructs the reaction and constants for the gene.
    :param str/GeneDescriptor descriptor: gene descriptor
    :return GeneReaction:
    """
    if not isinstance(descriptor, GeneDescriptor):
      descriptor = GeneDescriptor.parse(descriptor)
    reaction = GeneReaction(descriptor)
    reaction.generate()
    return reaction
      

######################################################
class GeneNetwork(object):
  """
  Create a full model.
  Usage:
    network = GeneNetwork()
    network.update(<list-gene-descriptions>)
    network.update(<list-gene-descriptions>)
    network.do()
  Can then use network.model and network.parameters
  """

  def __init__(self, initial_network=INITIAL_NETWORK):
    """
    :param list-str gene_descriptors: gene descriptor strings
    """
    self._ngene = NUM_GENE
    self._network = {}  # Key is gene; value is GeneDescriptor
    self._constants = []  # All constants in model
    self._uninitialize_constants = []  # Constants not initialized to 0
    self.update(initial_network, is_initialize=False)
    # Parameters for which initializations are constructed
    self._parameter_initializations = []
    # Generated outputs
    # All parameters in the model
    self.parameters = None
    # Parameters initally set to 0
    self.new_parameters = None
    self.model = None  # Model string

  def update(self, strings, is_initialize=True):
    """
    Updates the gene network.
    :param list-str strings: list of gene descriptor strings
    :param bool is_intialize: constants should be initialized
    Updates
      self._constants
      self._initialize_constants
      self._network
    """
    for string in strings:
      new_reaction = GeneReaction.do(string)
      # Remove constants from the old descriptor
      if new_reaction.descriptor.ngene in self._network.keys():
        old_reaction = self._network[new_reaction.descriptor.ngene]
        difference = set(self._constants).difference(
            old_reaction.constants)
        self._constants = list(difference)
      # Add the new reaction to the network
      self._network[new_reaction.descriptor.ngene] = new_reaction
      self._constants = _extendUnique(
          self._constants, new_reaction.constants)
      if not is_initialize:
        self._uninitialize_constants = _extendUnique(
            self._uninitialize_constants, new_reaction.constants)
    # Verify that this is a complete network
    if len(self._network.keys()) != self._ngene:
      raise RuntimeError("Some key is not initialized: %s:"
          % str(self._network.keys()))

  def addInitialization(self, parameters):
    """
    Adds a constant to initialize.
    :param lmfit.Parameters parameters:
    Updates:
      self._parameter_initializations
    """
    self._parameter_initializations.append(parameters)

  def generate(self):
    """
    Generates an antimony model for the gene network.
    Updates
      self.model
      self.parameters
      self.new_parameters
    """
    # 1: Append the head of the file
    self.model = util.readFile(cn.PATH_DICT[cn.HEAD])
    # 2: Append gene and protein reactions
    self.model += str(self)
    self.model += util.readFile(cn.PATH_DICT[cn.PROTEIN])
    # 3a: Append constant initializations
    comment = "\n\n// Initializations for new constants\n"
    self.model += comment
    constants = [v for v in self._constants
        if not v in self._uninitialize_constants]
    self.model += self._makeInitializationStatements(constants,
        np.repeat(0, len(constants)))
    # 3b: Append initializations from parameters
    initialized_constants = []
    self.model += "\n"
    for parameters in self._parameter_initializations:
      constants = parameters.valuesdict().keys()
      values = parameters.valuesdict().values()
      self.model += self._makeInitializationStatements(
          constants, values)
      self.model += "\n"
      initialized_constants.extend(constants)
    # 4: Append the tail of the file
    self.model += "\n" + util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    self.model += "\n" + util.readFile(cn.PATH_DICT[cn.CONSTANTS])
    # 5: Construct the lmfit.parameters for constants in the model
    self.parameters = mg.makeParameters(self._constants)
    new_constants = set(self._constants).difference(
        self._uninitialize_constants)
    new_constants = new_constants.difference(initialized_constants)
    self.new_parameters = mg.makeParameters(
        list(set(new_constants)))

  def _makeInitializationStatements(self, constants, values):
    """
    Creates statements that initialize constants.
    :param list-str constants:
    :param list-float values:
    :return str:
    """
    statements = "\n".join(["%s = %f;" % (n, v)
        for n, v in zip(constants, values)])
    return statements

  def copy(self):
    """
    Copies the GeneReaction.
    :return GeneReaction:
    """
    return copy.deepcopy(self)

  def __repr__(self):
    return "\n".join([str(r) for r in self._network.values()])

  @classmethod
  def makeParameterInitializations(cls, df):
    """
    Creates parameter initialization statements for parameter
    values.
    :param pd.DataFrame df:
        name: name of the parameter
        value: value of the parameter
    :return str:
    """
    statements = ""
    for idx in df.index:
      row = df.loc[idx, :]
      statement = "%s = %f;\n" % (row[cn.NAME], row[cn.VALUE])
      statements += statement
    return statements
    
