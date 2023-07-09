'''

Testing common util functions

'''

from uniteai.common import insert_text_at, find_pattern_in_document
import pytest


def test_insert_text_at():
    # Test case: Starting empty
    doc = ""
    assert insert_text_at(doc, "Beautiful", 0, 0) == "Beautiful"
    assert insert_text_at(doc, "Beautiful", 1, 0) == "\nBeautiful"
    assert insert_text_at(doc, "Beautiful", -1, 0) == "\nBeautiful"

    # Test case: Simple
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "Beautiful ", 1, 0) == "Hello\nBeautiful World"
    assert insert_text_at(doc, " Beautiful", 0, 5) == "Hello Beautiful\nWorld"
    assert insert_text_at(doc, "Beautiful", 2, 0) == "Hello\nWorld\nBeautiful"

    doc = "Hello\nWorld\n012345"
    assert insert_text_at(doc, " XYZ ", 2, 2) == "Hello\nWorld\n01 XYZ 2345"

    # Negatives, and end line
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "XYZ", 0, -1) == "HelloXYZ\nWorld"
    assert insert_text_at(doc, "XYZ", -1, 0) == "Hello\nWorld\nXYZ"
    assert insert_text_at(doc, "XYZ", -1, -1) == "Hello\nWorld\nXYZ"
    assert insert_text_at(doc, "XYZ", 1, 0) == "Hello\nXYZWorld"
    assert insert_text_at(doc, "XYZ", 1, 5) == "Hello\nWorldXYZ"

    # Test case: Errors
    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, " Beautiful", 0, 7)
    assert str(e.value) == "Column number out of range"

    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, "XYZ", -2, 0)
    assert str(e.value) == "Line number out of range"

    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, "XYZ", 0, -3)
    assert str(e.value) == "Column number out of range"


def test_find_pattern_in_document():
    document = '''
Hello, world!
Regex is fun.
I like programming in Python.
'''

    assert find_pattern_in_document(document, "o") == [(1, 4, 5), (1, 8, 9),
                                                       (3, 9, 10), (3, 26, 27)]
    assert find_pattern_in_document(document, "P...on") == [(3, 22, 28)]
    assert find_pattern_in_document(document, "Java") == []
    assert find_pattern_in_document('', "o") == []
