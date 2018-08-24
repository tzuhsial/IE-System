from iedsp.core import SysIntent


def test_systemintent():

    # Define variables
    i1 = SysIntent([], [{'slot': 2}], [{'slot': 3}])
    i2 = SysIntent([{'slot': 1}], [], [])

    assert not i1.empty()

    # __eq__
    assert i1 == i1 and i1 is i1

    # copy
    i5 = i1.copy()
    assert i5 == i1 and i5 is not i1

    # __add__
    i3 = i1 + i2
    assert i3 == SysIntent([{'slot': 1}], [{'slot': 2}], [{'slot': 3}])

    # __radd__
    i4 = i1.__radd__(i2)
    assert i4 == SysIntent([{'slot': 1}], [{'slot': 2}], [{'slot': 3}])

    # __iadd__
    i1 += i2
    assert i1 == SysIntent([{'slot': 1}], [{'slot': 2}], [{'slot': 3}])
    i1 += i2
    assert i1 == SysIntent([{'slot': 1}, {'slot': 1}], [
                           {'slot': 2}], [{'slot': 3}])

    assert i2 == SysIntent([{'slot': 1}], [], [])

    # empty
    empty_i = SysIntent()
    assert empty_i.empty()

    # clear
    i1.clear()
    assert i1 == SysIntent()
    assert i1.empty()