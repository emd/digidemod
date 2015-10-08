from nose import tools
from digidemod.sample import decimate


def test_decimate_valueErrors():
    # Attempts to decimate by a *negative* value should fail
    tools.assert_raises(ValueError, decimate, 1, -1)

    # Attempts to decimate by anything other than an int should fail
    tools.assert_raises(ValueError, decimate, 1, 1.5)

    # The transition band of the low pass filter cannot occupy
    # a negative fraction of the filtered signal's spectral bandwidth
    tools.assert_raises(ValueError, decimate,
                        1, 10, {'trans': -0.1})

    # The transition band of the low pass filter cannot occupy
    # a bandwidth larger than that of the filtered signal's bandwidth
    tools.assert_raises(ValueError, decimate,
                        1, 10, {'trans': 1.1})
