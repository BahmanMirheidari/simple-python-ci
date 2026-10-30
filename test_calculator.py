
import pytest

from calculator import add, divide, subtract


def test_add():
    assert add(2, 3) == 5


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
        

def test_subtract():
    from calculator import subtract
    assert subtract(8, 3) == 5