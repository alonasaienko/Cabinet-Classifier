LABELS = [
    "lc:bcabo",
    "lc:wcabo",
    "lc:muscabinso",
    "lc:wcabcub",
    "lc:bcabocub",
]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
