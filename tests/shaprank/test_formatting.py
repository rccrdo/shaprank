import pytest

from shaprank import formatting


@pytest.mark.parametrize(
    "x,decimals,exp",
    [
        (0, 4, "0"),
        (0.1, 4, "0.1"),
        (0.0001, 4, "0.0001"),
        (0.00001, 4, "0"),
    ],
)
def test_number(x, decimals, exp):
    res = formatting.number(x, decimals=decimals)

    assert res == exp
