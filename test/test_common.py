'''

Testing common util functions

'''

from uniteai.common import insert_text_at, find_block, extract_block
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


def test_extract_block():
    doc = "This is the first document line.\nThe second line is start_tag and also contains some more text.\nThis is the third line between the tags.\nThis is the fourth line.\nThe fifth line is end_tag and also contains some more text.\nThis is the last document line."
    start_tag = "start_tag"
    end_tag = "end_tag"

    # Getting the start and end line and column tuples
    start, end = find_block(start_tag, end_tag, doc)

    # Expecting three lines as output - the line containing start tag, the line between the tags and the line containing the end tag
    expected_output = " and also contains some more text.\nThis is the third line between the tags.\nThis is the fourth line.\nThe fifth line is "

    assert extract_block(start, end, doc) == expected_output, "Test case 1 failed!"

    # Test case when no tag is there in document
    start_tag = "no_tag"
    end_tag = "no_tag"
    start, end = find_block(start_tag, end_tag, doc)

    # Expecting None since no tag is there in document
    expected_output = None

    assert extract_block(start, end, doc) == expected_output, "Test case 2 failed!"

    print('All test cases passed!')
