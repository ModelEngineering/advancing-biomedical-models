'''Extends list.'''

class ExtendedList(list):

  def removeAll(self, element):
    while self.count(element) > 0:
      self.remove(element)

  def unique(self):
    """
    Returns a list of unique elements.
    Does not preserve order.
    """
    new_list = ExtendedList(list(set(self)))
    [self.pop() for _ in range(len(self))]
    [self.append(e) for e in new_list]

  def isSubset(self, other):
    """
    Determines if elements of this object are in other.
    :param iterable other:
    :return bool:
    """
    trues = [x in other for x in self]
    return all(trues)

  def isSame(self, other):
    """
    Determines if elements of this object are the same as other.
    :param iterable other:
    :return bool:
    """
    extended_other = ExtendedList([x for x in other])
    return self.isSubset(extended_other)  \
        and extended_other.isSubset(self)
