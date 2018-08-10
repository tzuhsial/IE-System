from iedsp.core import Slot, SysIntent


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


def test_slot():

    s1 = Slot('a')
    s2 = Slot('b')

    assert sorted([s2, s1]) == [s1, s2]

    assert s1 < s2
    assert s2 > s1
    assert s1 <= s1
    assert s2 >= s2

    s3 = Slot('a')
    assert s1 == s1 and s1 is s1
    assert s1 == s3 and s1 is not s3

    assert s1.to_json() == {'slot': 'a', 'value': None, 'conf': None}
