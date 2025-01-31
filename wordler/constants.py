from pathlib import Path

#: Path to the assets directory
ASSETS: Path = Path(__file__).parents[1] / "assets"

#: Path to the word list file
WORDS_FILE: Path = ASSETS / "word-list.txt"
