import pytest
from templatelib.template import Template


@pytest.mark.parametrize(
    "value1, value2, expected",
    [
        (1, 1, 2),
        (2, 2, 4),
        (3, 2.0, 5),
        (4, 2, 6),
        (5, 7, 12),
        (3, 4, 7),
        ("3", "4", 7),
        (3.0, 4.0, 7),
    ],
)
def test_sum(value1, value2, expected):
    template = Template()
    assert template.sum(value1, value2) == expected
