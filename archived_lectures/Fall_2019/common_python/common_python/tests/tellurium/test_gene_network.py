from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

import common_python.tellurium.constants as cn
from common_python.tellurium import modeling_game as mg
from common_python.tellurium import gene_network as gn
from common_python.tellurium.gene_network import  \
    GeneDescriptor, GeneReaction, GeneNetwork

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
NGENE = 1
NPROTS = [2, 3]
IS_ACTIVATES = [True, False]
NCONST_0_TF = 2  # L, d_mRNA
NCONST_1_TF = NCONST_0_TF + 3  # Vm, K1, H
NCONST_2_TF = NCONST_1_TF + 2  # K2, K3
# GeneDescriptor
ACTIVATE_1 = "+"
PROTEIN_1 = "3"
ACTIVATE_2 = "-"
PROTEIN_2 = "4"
INTEGRATE = "A"
GENE_0_TF = str(NGENE)
GENE_1_TF = "%d%s%s" % (NGENE, ACTIVATE_1, PROTEIN_1)
GENE_2_TF = "%d%s%s%s%s%s" % (
    NGENE, ACTIVATE_1, PROTEIN_1, INTEGRATE, ACTIVATE_2, PROTEIN_2)
NEW_NETWORK = [s.replace(str(1), str(v)) for s,v 
    in zip(np.repeat(gn.INITIAL_NETWORK[0], 8),
    range(1, gn.NUM_GENE+1))]


###########################################################
class TestGeneDescriptor(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    if IGNORE_TEST:
      return
    self.descriptor = GeneDescriptor.parse(GENE_2_TF)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    descriptor = GeneDescriptor(NGENE,
        nprots=[PROTEIN_1, PROTEIN_2],
        is_activates=[True, False],
        is_or_integration=False)
    self.assertTrue(self.descriptor.equals(descriptor))

  def testParse(self):
    if IGNORE_TEST:
      return
    desc = GeneDescriptor.parse("2")
    self.assertEqual(desc.ngene, 2)
    desc = GeneDescriptor.parse(2)
    self.assertEqual(desc.ngene, 2)
    desc = GeneDescriptor.parse("2+3")
    self.assertEqual(desc.ngene, 2)
    self.assertEqual(desc.is_activates[0], True)
    self.assertEqual(desc.nprots[0], 3)
    desc = GeneDescriptor.parse("2-3")
    self.assertEqual(desc.is_activates[0], False)
    self.assertEqual(desc.nprots[0], 3)
    desc = GeneDescriptor.parse("2-3O+4")
    self.assertEqual(len(desc.is_activates), 2)
    self.assertEqual(desc.is_activates[0], False)
    self.assertEqual(desc.nprots[0], 3)
    self.assertEqual(desc.is_activates[1], True)
    self.assertEqual(desc.nprots[1], 4)

  def testParse2(self):
    if IGNORE_TEST:
      return
    # Test by checking the we can recover the original string
    def test(descriptor_stg):
      descriptor = GeneDescriptor.parse(descriptor_stg)
      stg = str(descriptor)
      self.assertEqual(descriptor_stg, stg)
    #
    test("6+7P-1")
    test("1+2O-1")
    test("1+2P-1")
    test("1+2A-1")
    test("1")
    test("1+2")
    

###########################################################
class TestGeneReaction(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self, descriptor=NGENE):
    if isinstance(descriptor, str):
      descriptor = GeneDescriptor.parse(descriptor)
    self.reaction = GeneReaction(descriptor)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.reaction.descriptor.nprots), 0)

  def _adds(self, reaction, nprots, is_activates):
    if IGNORE_TEST:
      return
    is_activates = [True, False]
    [reaction.add(n, is_activate=b)
        for b, n in zip(is_activates, NPROTS)]

  def testAddProtein(self):
    if IGNORE_TEST:
      return
    self._adds(self.reaction, NPROTS, IS_ACTIVATES)
    self.assertFalse(all(self.reaction.descriptor.is_activates))
    self.assertEqual(len(self.reaction.descriptor.is_activates),
      len(NPROTS))
    self.assertEqual(len(self.reaction.descriptor.nprots),
      len(NPROTS))
   
  def testMakePVar(self):
    if IGNORE_TEST:
      return
    nprot = 24
    self.assertEqual("P%d" % nprot, self.reaction._makePVar(nprot))

  def testMakeMrnaBasicKinetics(self):
    if IGNORE_TEST:
      return
    stg = self.reaction._makeMrnaBasicKinetics()
    self.assertTrue("L" in stg)
    self.assertTrue("d_mRNA" in stg)
    self.assertTrue("*mRNA" in stg)

  def testMakeTerm(self):
    if IGNORE_TEST:
      return
    stg = self.reaction._makeTerm(NPROTS)
    self.assertTrue("K1" in stg)
    self.assertEqual(stg.count("P"), 2)
    stg = self.reaction._makeTerm(NPROTS)
    self.assertTrue("K2" in stg)

  def testMakeNumerator(self):
    if IGNORE_TEST:
      return
    def test(stg, is_term0):
      descriptor = GeneDescriptor.parse(stg)
      reaction = GeneReaction(descriptor)
      terms = reaction._makeTerms()
      numerator = reaction._makeNumerator(terms)
      if is_term0:
        expecteds = ["H", "1", "K"]
      else:
        expecteds = "1"
      for expected in expecteds:
        self.assertTrue(expected in numerator)
    #
    test("4+2A-1", True)
    test("4+2", True)
    test("4-2", False)

  def testMakeTFKinetics(self):
    if IGNORE_TEST:
      return
    def test(is_or_integration):
      reaction = GeneReaction(NGENE,
          is_or_integration=is_or_integration)
      self._adds(reaction, NPROTS, IS_ACTIVATES)
      stg = reaction._makeTFKinetics()
      for n in range(1, 4):
        substg = "K%d" % n
        if not is_or_integration:
          break
        self.assertTrue(substg in stg)
      self.assertTrue("/" in stg)
    #
    test(True)
    test(False)

  def testMakeTFKinetics1(self):
    if IGNORE_TEST:
      return
    FACTOR = "K3_6"
    def test(stg, is_factor):
      descriptor = GeneDescriptor.parse(stg)
      self._init(descriptor)
      kinetics = self.reaction._makeTFKinetics()
      if is_factor:
        self.assertTrue(FACTOR in kinetics)
      else:
        self.assertFalse(FACTOR in kinetics)
    #
    test("6+7P-1", False)
    test("6+7O-1", True)

  def testMakeTFKinetics2(self):
    if IGNORE_TEST:
      return
    def makeInit(var):
      return "%s = 0\n" % var
    # 
    def test(desc_stg):
      # Verifies correct construction of kinetics expression
      ngene = int(desc_stg[0])
      statements = makeInit("INPUT")
      for constant in ["Vm", "H"]:
        statements += makeInit("%s%d" % (constant, ngene))
      # Initialize possible constants
      for idx in range(1, 4):
        statements += makeInit("K%d_%d" % (idx, ngene))
      # Initialize proteins
      for protein in range(1, gn.NUM_GENE + 1):
        statements += "P%d = 0\n" % protein
      # Add the kinetics expression
      self._init(desc_stg)
      kinetics = self.reaction._makeTFKinetics()
      statements += "_ = %s\n" % kinetics
      try:
        exec(statements)
      except:
        self.assertTrue("Bad statements: %s" % statements)
    #
    test("1+0O+4")
    test("6+7P-1")
    test("6+7O-1")
    test("1+2O-1")
    test("1+2P-1")
    test("1+2A-1")
    test("1")
    test("1+2")

  def testGenerate(self):
    if IGNORE_TEST:
      return
    self._adds(self.reaction, NPROTS, IS_ACTIVATES)
    self.reaction.generate()
    self.assertEqual(len(self.reaction.constants), NCONST_2_TF)
    self.assertTrue("=>" in self.reaction.mrna_reaction)
    for constant in self.reaction.constants:
      self.assertTrue(constant in self.reaction.mrna_reaction)
    self.assertTrue("P%d" % NGENE in self.reaction.protein_kinetics)
 
  def testDo(self):
    if IGNORE_TEST:
      return
    desc_stg = "8-1"
    spec = GeneReaction.do(desc_stg)
    desc_stg = "1+0O+4"
    reaction = GeneReaction.do(desc_stg)
    self.assertEqual(len(reaction.constants),  NCONST_2_TF)
    desc_stg = "4-2"
    reaction = GeneReaction.do(desc_stg)
    self.assertEqual(len(reaction.constants),  NCONST_1_TF)

  def testStr(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self._adds(self.reaction, NPROTS, IS_ACTIVATES)
    _ = str(self.reaction)
    

###########################################################
class TestGeneNetwork(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.network = GeneNetwork()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.network._network.keys()),
        gn.NUM_GENE)

  def testUpdate(self):
    if IGNORE_TEST:
      return
    def strip(stg):
      p = stg.index("K3_1")
      return stg[p:]
    #
    self.network.update(NEW_NETWORK)
    length1 = len(str(self.network._network[1]))
    for reaction in self.network._network.values():
      self.assertEqual(length1, len(str(reaction)))

  def testUpdate2(self):
    if IGNORE_TEST:
      return
    # Tests for constant initialization
    network = GeneNetwork(range(1,
        gn.NUM_GENE + 1))
    network.update(gn.INITIAL_NETWORK)
    difference = set(self.network._constants).difference(
        network._constants)
    self.assertEqual(len(difference), 0)
    difference = set(self.network._uninitialize_constants).difference(
        network._uninitialize_constants)
    difference = list(difference)
    for constant_type in ["L", "d_mRNA"]:
      self.assertFalse(constant_type in difference)

  def testGenerate1(self):
    # Check parameters
    if IGNORE_TEST:
      return
    self._init()
    network = self.network.copy()
    network.generate()
    constants = list(network.parameters.valuesdict().keys())
    new_constants = list(network.new_parameters.valuesdict().keys())
    self.assertEqual(len(set(new_constants).difference(constants)), 0)
    #
    def test(prefix, suffixes):
      for sfx in suffixes:
        stg = "%s%s" %  (prefix, str(sfx))
        self.assertTrue(stg in constants, "%s not in constants" % stg)
    #
    for pfx in ["L", "d_mRNA"]:
      test(pfx, range(1, gn.NUM_GENE + 1))
    for pfx in ["Vm", "H", "K1_"]:
      test(pfx, [1, 2, 3, 4, 6, 8])
    for pfx in ["K2_", "K3_"]:
      test(pfx, [1, 6])

  def testGenerate2(self):
    # Checks model
    if IGNORE_TEST:
      return
    # Check parameters
    network = self.network.copy()
    network.generate()
    constants = network.parameters.valuesdict().keys()
    # Other checks
    self.network.update(NEW_NETWORK)
    self.network.generate()
    self.assertGreater(len(self.network.model), 4000)
    self.assertGreater(len(self.network.parameters.valuesdict()), 10)

  def testAddInitialization(self):
    if IGNORE_TEST:
      return
    self._init()
    VALUES = [1.5, 4.8]
    CONSTANTS = ["Vm6", "H6"]
    for value, constant in zip(VALUES, CONSTANTS):
      parameters = mg.makeParameters([constant], [value])
      self.network.addInitialization(parameters)
    self.network.generate()
    for value, constant in zip(VALUES, CONSTANTS):
      statement = "%s = %f" % (constant, value)
      self.assertTrue(statement in self.network.model)

  def testMakeParameterInitializations(self):
    COLA = "a"
    VALUEA = 1
    COLB = "b"
    VALUEB = 2
    COLUMNS = [COLA, COLB]
    df = pd.DataFrame({
      cn.NAME: [COLA, COLB],
      cn.VALUE: [VALUEA, VALUEB],
      })
    result = gn.GeneNetwork.makeParameterInitializations(df)
    trues = [c in result for c in COLUMNS]
    self.assertTrue(all(trues))


if __name__ == '__main__':
  unittest.main()
