
class Counter:
    """Basic counter class. Increments by one on calls to `iterate`

    :param _counter: The current value of the counter.
    :type _counter: int
    """

    def __init__(self):
        """Constructor method."""
        self._counter = 0

    @property
    def counter(self):
        """Returns the current value of the counter.

        :return: The current value of the counter.
        :rtype: int
        """
        return self._counter

    def iterate(self):
        """Increments the counter """
        self._counter += 1

    def reset(self):
        self.__init__()


class AggregateCounter(Counter):
    """Counter that keeps the sum off all inputs as well as a count of all inputs. Can be used to compute an average
    of all items ingested.

    :param _total: Sum of all inputs ingested.
    :type _total: float
    """

    def __init__(self):
        """Constructor method."""
        super().__init__()
        self._total = 0

    @property
    def average(self):
        """Returns the average of all the values ingested by the counter. Returns 0 if no values have been ingested.

        :return: The average of all the values ingested by the counter.
        :rtype: float
        """
        return self.total/self.counter if self.counter > 0 else 0

    def iterate(self, value):
        """Increases the running total by the most recent value and increases the count by one.

        :param value: A new value to add to the counter.
        :type value: float
        """
        self._total += value
        super().iterate()

    def reset(self):
        """Resets the total and counter back to zero."""
        self.__init__()

    @property
    def total(self):
        """Returns the total sum of all values ingested.

        :return: The total sum of all values ingested.
        :rtype: float
        """
        return self._total


class RecordCounter(Counter):
    """Counter that keeps a list of all items ingested.

    :param _record: List of all items ingested.
    :type _record: List
    """

    def __init__(self):
        """Constructor method."""
        super().__init__()
        self._record = []

    def iterate(self, value):
        """Adds new value to the list and increases the count by one.

        :param value: A new value to add to the counter
        """
        self._record.append(value)
        super().iterate()

    @property
    def record(self):
        """Returns the current list of values ingested.

        :return: The current list of values ingested.
        :rtype: List
        """
        return self._record

    def reset(self):
        """Resets the Counter. List is returned to empty and counter is returned to zero."""
        self.__init__()
