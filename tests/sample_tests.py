from nose import tools
from digidemod.sample import decimate


@tools.raises(ValueError)
def test_decimate_valueError1():
    # Attempts to decimate by a *negative* value should fail
    decimate(1, -1)


@tools.raises(ValueError)
def test_decimate_valueError2():
    # Attempts to decimate by anything other than an int should fail
    decimate(1, 1.5)
