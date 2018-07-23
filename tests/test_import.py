import sys


class A:
    pass


class B:
    pass


def test_name():

    a = getattr(sys.modules[__name__], "A")
    assert a == A

    c = getattr(sys.modules[__name__], "C")
