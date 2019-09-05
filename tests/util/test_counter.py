from src.util.counter import AverageCounter, Counter


def test_counter():
    counter = Counter()
    counter.iterate()
    assert counter.counter == 1
    for _ in range(10):
        counter.iterate()
    assert counter.counter == 11


def test_average_counter():
    avg_counter = AverageCounter()
    nums = [1, 1, 2, 2, 2, 1]
    for n in nums:
        avg_counter.iterate(n)
    assert avg_counter.total == 9
    assert avg_counter.counter == 6
    assert avg_counter.average == 1.5
