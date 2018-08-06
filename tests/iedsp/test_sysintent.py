from iedsp.core import SysIntent


def test_systemintent():

    # Define variables
    i1 = SysIntent([], [2], [], [4])
    i2 = SysIntent([1], [], [3], [])

    assert not i1.empty()

    # __eq__
    assert i1 == i1 and i1 is i1

    # __add__
    i3 = i1 + i2
    assert i3 == SysIntent([1], [2], [3], [4])

    # __radd__
    i4 = i1 + i2
    assert i4 == SysIntent([1], [2], [3], [4])

    # __iadd__
    i1 += i2
    assert i1 == SysIntent([1], [2], [3], [4])
    i1 += i2
    assert i1 == SysIntent([1, 1], [2], [3, 3], [4])

    assert i2 == SysIntent([1], [], [3], [])

    # empty
    empty_i = SysIntent()
    assert empty_i.empty()

    # clear
    i1.clear()
    assert i1 == SysIntent()
    assert i1.empty()

    # copy
    i5 = i1.copy()
    assert i5 == i1 and i5 is not i1
