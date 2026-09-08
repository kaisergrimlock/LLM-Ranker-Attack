"""Insert precomputed query keywords at random passage word boundaries."""

import csv
import json
import random
import re
from pathlib import Path

DEFAULT_KEYWORDS = (
    Path(__file__).resolve().parents[1] / "content_attack/unique_queries.tsv"
)


class KeywordInjection:
    """Keep query keywords and insertion randomness separate from target selection."""

    def __init__(self, path, seed):
        """Load and validate the precomputed query-keyword mapping."""
        self.rng = random.Random(seed)
        self.keywords = {}
        with Path(path).open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                query = row["query"]
                raw = row.get("keywords", "").strip()
                if not raw:
                    continue
                words = json.loads(raw)
                if (
                    not isinstance(words, list)
                    or not words
                    or any(
                        not isinstance(word, str) or not word.strip() for word in words
                    )
                ):
                    raise ValueError(f"Invalid keywords for query: {query!r}")
                words = list(dict.fromkeys(word.strip() for word in words))
                if query in self.keywords and self.keywords[query] != words:
                    raise ValueError(f"Conflicting keywords for query: {query!r}")
                self.keywords[query] = words

    def validate_queries(self, instances):
        """Reject missing keyword entries before the clean ranking calls."""
        missing = {row[0] for row in instances if row[0] not in self.keywords}
        if missing:
            raise ValueError(
                f"Missing keywords for {len(missing)} queries: {sorted(missing)[:3]}"
            )

    def __call__(self, query, text):
        """Insert every unique keyword once without changing original passage text."""
        boundaries = [0] + [match.end() for match in re.finditer(r"\S+", text)]
        insertions = {}
        for keyword in self.keywords[query]:
            offset = self.rng.choice(boundaries)
            insertions.setdefault(offset, []).append(keyword)
        for offset in sorted(insertions, reverse=True):
            payload = " " + " ".join(insertions[offset]) + " "
            text = text[:offset] + payload + text[offset:]
        return text


def configure_attack(parser, args, prompts):
    """Resolve attack text and validate placement before preparing evaluations."""
    position = "random" if args.attack_type == "key_injection" else "back"
    if args.attack_position is None:
        args.attack_position = position
    if args.attack_type == "key_injection":
        if args.attack_position != "random":
            parser.error("key_injection requires --attack_position random")
        try:
            return KeywordInjection(args.keywords_path, args.seed)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    if args.attack_position == "random":
        parser.error("--attack_position random requires --attack_type key_injection")
    return prompts[args.attack_type]


def render_attack_text(attack, query, text, position):
    """Render either keyword insertions or an existing prefix/suffix attack."""
    if isinstance(attack, KeywordInjection):
        return attack(query, text)
    payload = attack.format(query=query)
    return payload + text if position == "front" else text + payload
