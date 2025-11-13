"""Utility functions for parsing structured text and tags."""

import re
from typing import Dict


def parse_tags(text: str) -> Dict[str, str]:
    """
    Parse XML-style tags from text.

    Extracts content from tags like <tag_name>content</tag_name>

    Args:
        text: Text containing XML-style tags

    Returns:
        Dictionary mapping tag names to their content

    Example:
        >>> text = "Some text <url>http://example.com</url> more text"
        >>> parse_tags(text)
        {'url': 'http://example.com'}
    """
    tags = {}

    # Pattern to match <tag>content</tag>
    pattern = r'<(\w+)>(.*?)</\1>'

    matches = re.findall(pattern, text, re.DOTALL)

    for tag_name, content in matches:
        tags[tag_name] = content.strip()

    return tags
