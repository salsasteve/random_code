# tests/test_module.py
from sound_spec import greet


def test_greet():
    assert greet("World") == "Hello, World!"
