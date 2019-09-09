from unittest import TestCase

from src.util.counter import AverageCounter, Counter


class TestCounter(TestCase):
    def test_counter(self):
        counter = Counter()
        counter.iterate()
        assert counter.counter == 1
        for _ in range(18):
            counter.iterate()
        assert counter.counter == 19
        counter.reset()
        assert counter.counter == 0

    def test_average_counter(self):
        avg_counter = AverageCounter()
        nums = [9.15, 3.8, 9.79, 4.09, 7.36, 1.46, 9.52, 4.04, 5.36, 0.8]
        for n in nums:
            avg_counter.iterate(n)
        assert avg_counter.total == 55.37
        assert avg_counter.counter == 10
        assert avg_counter.average == 5.537
        avg_counter.reset()
        assert avg_counter.total == 0
        assert avg_counter.counter == 0
        self.assertRaises(ZeroDivisionError, lambda: avg_counter.average)
