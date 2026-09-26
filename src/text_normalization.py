"""
Text normalization module for Business Entity Resolution.
Handles multilingual characters, accents, legal entity suffixes, and address abbreviations.
"""

import re
import unicodedata
from typing import Set, List, Tuple

# Common legal and corporate suffixes across US, India, and France
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(pvt|private|ltd|limited|llc|inc|incorporated|corp|corporation|co|company|"
    r"gmbh|sa|sarl|sas|sasu|llp|plc|holdings|enterprises|services|group)\b",
    re.IGNORECASE
)

# Address abbreviations for standardizing addresses
ADDRESS_REPLACEMENTS = [
    (re.compile(r"\b(rd|rd\.)\b", re.IGNORECASE), "road"),
    (re.compile(r"\b(st|st\.)\b", re.IGNORECASE), "street"),
    (re.compile(r"\b(ave|ave\.)\b", re.IGNORECASE), "avenue"),
    (re.compile(r"\b(blvd|blvd\.)\b", re.IGNORECASE), "boulevard"),
    (re.compile(r"\b(dr|dr\.)\b", re.IGNORECASE), "drive"),
    (re.compile(r"\b(ln|ln\.)\b", re.IGNORECASE), "lane"),
    (re.compile(r"\b(hwy|hwy\.)\b", re.IGNORECASE), "highway"),
    (re.compile(r"\b(apt|apt\.)\b", re.IGNORECASE), "apartment"),
    (re.compile(r"\b(ste|ste\.)\b", re.IGNORECASE), "suite"),
    (re.compile(r"\b(fl|flr|fl\.)\b", re.IGNORECASE), "floor"),
    (re.compile(r"\b(bldg|bldg\.)\b", re.IGNORECASE), "building"),
    (re.compile(r"\b(sq|sq\.)\b", re.IGNORECASE), "square"),
    (re.compile(r"\b(opp|opp\.)\b", re.IGNORECASE), "opposite"),
    (re.compile(r"\b(nr|nr\.)\b", re.IGNORECASE), "near"),
    (re.compile(r"\b(no|no\.)\b", re.IGNORECASE), "number"),
]

# Common non-distinctive words in entity names and addresses
NAME_STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "by", "from", "to", "with",
    "et", "de", "des", "du", "la", "le", "en", "pour", "sur",
    "co", "inc", "ltd", "pvt", "llc", "corp", "sa", "sarl"
}

PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_REGEX = re.compile(r"\s+")
DIGIT_REGEX = re.compile(r"\d+")

def normalize_unicode(text: str) -> str:
    """Normalize unicode and remove accents/diacritics."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def clean_general(text: str) -> str:
    """Basic cleaning: unicode, lowercase, remove punctuation, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = normalize_unicode(text).lower()
    cleaned = PUNCT_REGEX.sub(" ", cleaned)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
    return cleaned

def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Returns (cleaned_name, core_name_without_legal_suffix).
    Preserves both representations for feature extraction.
    """
    cleaned = clean_general(name)
    core = LEGAL_SUFFIX_PATTERN.sub(" ", cleaned)
    core = WHITESPACE_REGEX.sub(" ", core).strip()
    if not core:
        core = cleaned
    return cleaned, core

def clean_address(addr: str) -> str:
    """Cleans and standardizes address abbreviations."""
    if not addr or not isinstance(addr, str):
        return ""
    cleaned = clean_general(addr)
    for pattern, replacement in ADDRESS_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return WHITESPACE_REGEX.sub(" ", cleaned).strip()

def extract_tokens(text: str, min_len: int = 2) -> Set[str]:
    """Extract set of tokens meeting minimum length threshold."""
    if not text:
        return set()
    return {w for w in text.split() if len(w) >= min_len}

def extract_distinctive_name_tokens(core_name: str) -> Set[str]:
    """Extract distinctive name tokens excluding common stopwords."""
    tokens = extract_tokens(core_name, min_len=2)
    distinctive = {t for t in tokens if t not in NAME_STOPWORDS}
    return distinctive if distinctive else tokens

def extract_numbers(text: str) -> Set[str]:
    """Extract digits/numbers (PIN codes, street numbers, building numbers)."""
    if not text:
        return set()
    return set(DIGIT_REGEX.findall(text))
