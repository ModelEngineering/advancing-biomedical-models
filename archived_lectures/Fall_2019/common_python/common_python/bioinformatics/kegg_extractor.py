"""Extracts information from the KEGG repositories."""

import common_python.constants as cn

import pandas as pd
import numpy as np
import requests

SEPARATOR = "/"
STATUS_GOOD = 200
GENE = "GENE"


class KeggExtractor(object):

  def __init__(self, organism):
    """
    :param str organism: KEGG organism string
    """
    self._organism = organism
    self._url = ""  # URL for request

  def _makeBaseURL(self, command):
    """ 
    Creates a base URL with the KEGG REST command.
    :param str command:
    """ 
    if not command in cn.KEGG_CMDS:
      raise ValueError("Invalid KEGG Command.")
    self._url = "http://rest.kegg.jp/%s" % command

  def _addArguments(self, arguments):
    """
    Adds REST arguments to the URL.
    """
    for argument in arguments:
      self._url = "%s%s%s" % (self._url, SEPARATOR, argument)

  def _issueRequest(self):
    """
    Sends the URL and checks the status.
    :return str:
    """
    response = requests.get(self._url, headers={})
    if response.status_code != STATUS_GOOD:
      raise ValueError("Bad response to URL: %s" % self._url)
    return response.text

  def listPathway(self):
    """
    Gets a list of pathways for the organism.
    The URL should be "http://rest.kegg.jp/list/pathway/<organism>"
    :return pd.DataFrame:
        KEGG_PATHWAY
        KEGG_DESCRIPTION
    """
    self._makeBaseURL(cn.KEGG_CMD_LIST)
    self._addArguments(["pathway", self._organism])
    text = self._issueRequest()
    lines =text.split("\n")
    result = self.__class__._initializeDict([
        cn.KEGG_PATHWAY, cn.KEGG_DESCRIPTION])
    for line in lines:
        splits = line.split("\t")
        if len(splits) > 1:
            result[cn.KEGG_PATHWAY].append(splits[0])
            result[cn.KEGG_DESCRIPTION].append(splits[1])
    return pd.DataFrame(result)

  @staticmethod
  def _initializeDict(names):
    """
    Initializes dictionary to empty lists for each name.
    :return dict:
    """
    result = {}
    for name in names:
      result[name] = []
    return result

  @staticmethod
  def _extractIdentifiedString(identifier, full_stg):
    """
    Extracts an structured string of the form
    [<identifier>:<stg>]
    :param str identifier:
    :param str stg:
    :return str: <stg> or None (if not present)
    """
    start_locator = "[%s:" % identifier
    end_locator = "]"
    start = full_stg.find(start_locator)
    if start < 0:
      return None
    end = full_stg[start:].find(end_locator) + start
    if end < 0:
      return None
    start = start + 1 + len(identifier) + 1
    stg = full_stg[start:end]
    return stg

  def getPathwayGenes(self, pathway):
    """
    Obtains the genes for a pathway.
    :param str pathway:
    :return pd.DataFrame:
        KEGG_PATHWAY - always pathway
        KEGG_GENE
        KEGG_DESCRIPTION
        KEGG_EC
        KEGG_KO
    Note:
      1. Key is KEGG_GENE, KEGG_EC
      2. Renames gene to be consisent with Rv* convention
    """
    self._makeBaseURL(cn.KEGG_CMD_GET)
    self._addArguments([pathway])
    lines = self._issueRequest().split("\n")
    is_gene = False
    result = self.__class__._initializeDict([
        cn.KEGG_PATHWAY, cn.KEGG_GENE, cn.KEGG_DESCRIPTION,
        cn.KEGG_EC, cn.KEGG_KO])
    for idx, line in enumerate(lines):
      splits = line.split(" ")
      # Is start of gene?
      if splits[0] == GENE:
        is_gene = True
        line = line.replace(GENE, "")
      # Process a gene line, handling multiple EC numbers
      if is_gene:
        # Check for end
        if line[0] != " ":
          break
        # Still have genes
        splits = [t for t in line.split(" ") if len(t) > 0]
        if len(splits) < 1:
          import pdb; pdb.set_trace()
          continue
        gene = splits[0]
        gene = gene.replace("VBD_", "v")
        description = " ".join(splits[1:])
        #
        ec_stg = self.__class__._extractIdentifiedString(
          "EC", description)
        if ec_stg is None:
          ecs = [None]
        else:
          ecs = ec_stg.split(" ")
        for ec in ecs:
          ko_stg = self.__class__._extractIdentifiedString(
            "KO", description)
          if ko_stg is None:
            kos = [None]
          else:
            kos = ko_stg.split(" ")
          for ko in kos:
            result[cn.KEGG_PATHWAY].append(pathway)
            result[cn.KEGG_GENE].append(gene)
            result[cn.KEGG_DESCRIPTION].append(description)
            result[cn.KEGG_EC].append(ec)
            result[cn.KEGG_KO].append(ko)
    return pd.DataFrame(result)

  def getAllPathwayGenes(self, max_count=-1):
    """
    Gets the genes in each pathway.
    :param int max_count: Number of pathways to process
    :return pd.DataFrame:
        KEGG_PATHWAY - always pathway
        KEGG_GENE
        KEGG_DESCRIPTION
        KEGG_EC
        KEGG_KO
    """
    df_pathway = self.listPathway()
    dfs = []
    for count, pathway in enumerate(df_pathway[cn.KEGG_PATHWAY]):
      if (count >= max_count) and (max_count > 0):
        break
      dfs.append(self.getPathwayGenes(pathway))
    return pd.concat(dfs)
