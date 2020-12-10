"""Tests for KeggExtractor."""

import common_python.constants as cn
from common_python.bioinformatics import kegg_extractor
from common_python.testing import helpers

import unittest

IGNORE_TEST = False
MTV = "mtv"  # TB organism


class TestKeggExtractor(unittest.TestCase):

  def setUp(self):
    self.extractor = kegg_extractor.KeggExtractor(MTV)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.extractor._organism, MTV)

  def testMakeBaseURL(self):
    if IGNORE_TEST:
      return
    self.extractor._makeBaseURL(cn.KEGG_CMD_LIST)
    self.assertEqual(self.extractor._url,
        "http://rest.kegg.jp/list")

  def testAddArguments(self):
    if IGNORE_TEST:
      return
    self.extractor._makeBaseURL(cn.KEGG_CMD_LIST)
    self.extractor._addArguments(["pathway", MTV])
    self.assertEqual(self.extractor._url,
        "http://rest.kegg.jp/list/pathway/mtv")

  def testIssueRequest(self):
    if IGNORE_TEST:
      return
    self.extractor._makeBaseURL(cn.KEGG_CMD_LIST)
    self.extractor._addArguments(["pathway", MTV])
    text = self.extractor._issueRequest()
    self.assertGreater(len(text), 0)

  def testListPathway(self):
    if IGNORE_TEST:
      return
    df = self.extractor.listPathway()
    self.assertTrue(helpers.isValidDataFrame(df, 
        [cn.KEGG_PATHWAY, cn.KEGG_DESCRIPTION]))

  def testGetPathwayGenes(self):
    if IGNORE_TEST:
      return
    df = self.extractor.getPathwayGenes("path:mtv00010")
    self.assertTrue(helpers.isValidDataFrame(df, 
        [cn.KEGG_PATHWAY, cn.KEGG_GENE, cn.KEGG_DESCRIPTION,
        cn.KEGG_KO, cn.KEGG_EC]))

  def testGetAllPathwayGenes(self):
    if IGNORE_TEST:
      return
    df = self.extractor.getAllPathwayGenes(max_count=5)
    self.assertTrue(helpers.isValidDataFrame(df, 
        [cn.KEGG_PATHWAY, cn.KEGG_GENE, cn.KEGG_DESCRIPTION,
        cn.KEGG_KO, cn.KEGG_EC]))
    df2 = self.extractor.getAllPathwayGenes(max_count=6)
    self.assertGreater(len(df2), len(df))

  def testExtractIdentifiedString(self):
    if IGNORE_TEST:
      return
    ec_stg = "1.2.1.-"
    full_stg = "aldehyde dehydrogenase  [KO:K00138] [EC:%s]"  \
        % ec_stg
    stg = kegg_extractor.KeggExtractor._extractIdentifiedString(
        "EC", full_stg)
    self.assertEqual(stg, ec_stg)
    #
    stg = kegg_extractor.KeggExtractor._extractIdentifiedString(
        "KO", full_stg)
    self.assertGreater(len(stg), 0)
    #
    stg = kegg_extractor.KeggExtractor._extractIdentifiedString(
        "KK", full_stg)
    self.assertIsNone(stg)


if __name__ == '__main__':
  unittest.main()
