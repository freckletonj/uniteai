from uniteai.lsp_server import extract_range
from lsprotocol.types import (
    Position,
    Range,
)
def test_extract_range():
    # Dummy document with some source text
    doc = (
'''Line0
Line1
Line2
Line3
Line4
'''
)

    # Dummy range (selects "Line 2" and "Line 3")
    start = Position(line=1, character=0)  # Line numbers are 0-indexed
    end = Position(line=2, character=3)
    range = Range(start=start, end=end)

    # Extract the text from the range
    extracted_text = extract_range(doc, range)

    # The extracted text should be "Line 2\nLine 3"
    assert extracted_text == "Line1\nLin"

    # Another range (selects "ine 2" from "Line 2")
    start = Position(line=2, character=3)  # Line numbers are 0-indexed
    end = Position(line=4, character=2)
    range = Range(start=start, end=end)

    # Extract the text from the range
    extracted_text = extract_range(doc, range)

    # The extracted text should be "ine 2"
    expected_text = (
'''e2
Line3
Li'''
)
    assert extracted_text == expected_text

test_extract_range()
