from llmpal.common import insert_text_at
import pytest

def test_insert_text_at():
    # Test case 0
    doc = ""
    assert insert_text_at(doc, "Beautiful", 0, 0) == "Beautiful"
    assert insert_text_at(doc, "Beautiful", 1, 0) == "\nBeautiful"
    assert insert_text_at(doc, "Beautiful", -1, 0) == "\nBeautiful"

    # Test case 1
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "Beautiful ", 1, 0) == "Hello\nBeautiful World"

    # Test case 2
    doc = "Hello\nWorld"
    assert insert_text_at(doc, " Beautiful", 0, 5) == "Hello Beautiful\nWorld"

    # Test case 3
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "Beautiful", 2, 0) == "Hello\nWorld\nBeautiful"

    # Test case 4
    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, " Beautiful", 0, 7)
    assert str(e.value)  == "Column number out of range"

    # Test case 5
    doc = "Hello\nWorld\n012345"
    assert insert_text_at(doc, " XYZ ", 2, 2) == "Hello\nWorld\n01 XYZ 2345"

    # Test case 6
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "XYZ", -1, -1) == "Hello\nWorld\nXYZ"

    # Test case 7
    doc = "Hello\nWorld"
    assert insert_text_at(doc, "XYZ", 0, -1) == "HelloXYZ\nWorld"
    assert insert_text_at(doc, "XYZ", -1, 0) == "Hello\nWorld\nXYZ"
    assert insert_text_at(doc, "XYZ", 1, 0) == "Hello\nXYZWorld"

    # Test case 8
    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, "XYZ", -2, 0)
    assert str(e.value)  == "Line number out of range"

    # Test case 9
    with pytest.raises(ValueError) as e:
        doc = "Hello\nWorld"
        insert_text_at(doc, "XYZ", 0, -3)
    assert str(e.value)  == "Column number out of range"
