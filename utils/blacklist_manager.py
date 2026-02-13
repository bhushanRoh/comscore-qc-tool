import pandas as pd
import os

BLACKLIST_FILE = "data/blacklist.csv"

def load_blacklist():
    """Load blacklist from CSV."""
    if not os.path.exists(BLACKLIST_FILE):
        return []
    try:
        df = pd.read_csv(BLACKLIST_FILE)
        if "term" in df.columns:
            return df["term"].tolist()
        return []
    except Exception:
        return []

def save_blacklist(terms):
    """Save blacklist to CSV."""
    df = pd.DataFrame({"term": terms})
    df.to_csv(BLACKLIST_FILE, index=False)

def add_term(term):
    """Add a term to the blacklist."""
    terms = load_blacklist()
    if term and term.lower() not in [t.lower() for t in terms]:
        terms.append(term)
        save_blacklist(terms)
        return True
    return False

def remove_term(term):
    """Remove a term from the blacklist."""
    terms = load_blacklist()
    original_len = len(terms)
    terms = [t for t in terms if t.lower() != term.lower()]
    if len(terms) < original_len:
        save_blacklist(terms)
        return True
    return False

def check_compliance(detected_text, blacklist):
    """
    Check if detected text contains any blacklist terms (exact substring match only).
    Returns:
        List of matching forbidden terms found in the text.
    """
    violations = []
    detected_lower = detected_text.strip().lower()

    if not detected_lower:
        return violations

    for term in blacklist:
        term_lower = term.lower()
        if term_lower in detected_lower:
            violations.append(term)

    return violations


def find_term_positions(detected_text, blacklist):
    """
    Find the character positions of each blacklist term within the detected text.
    Returns:
        List of (term, start_index, end_index) for each occurrence.
    """
    positions = []
    detected_lower = detected_text.strip().lower()

    if not detected_lower:
        return positions

    for term in blacklist:
        term_lower = term.lower()
        start = 0
        # Find ALL occurrences of the term in the text
        while True:
            idx = detected_lower.find(term_lower, start)
            if idx == -1:
                break
            positions.append((term, idx, idx + len(term_lower)))
            start = idx + 1

    return positions

