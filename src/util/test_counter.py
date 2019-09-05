from .counter import Counter


def test_counter():
    counter = Counter()
    counter.iterate()
    assert counter.counter == 1