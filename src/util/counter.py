
class Counter:

    def __init__(self):
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def iterate(self):
        self._counter += 1

    def reset(self):
        self.__init__()


class AverageCounter(Counter):

    def __init__(self):
        super().__init__()
        self._total = 0

    @property
    def average(self):
        return self.total/self.counter

    def iterate(self, value):
        self._total += value
        super().iterate()

    def reset(self):
        self.__init__()

    @property
    def total(self):
        return self._total


class RecordCounter(Counter):

    def __init__(self):
        super().__init__()
        self._record = []

    def iterate(self, value):
        self._record.append(value)
        super().iterate()

    @property
    def record(self):
        return self._record

    def reset(self):
        self.__init__()