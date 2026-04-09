import argparse
import json
import random
import re
from collections import Counter

import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def count_blanks(text: str) -> int:
    return len(re.findall(r"_{3,}", text or ""))


def make_replicate_rng(round_id: int | str, replicate: int) -> random.Random:
    """
    Deterministic RNG per (round_id, replicate).
    Ensures different slates across replicates, fixed across models.
    """
    seed = hash((str(round_id), replicate)) % (2**32)
    return random.Random(seed)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV with columns: player_id, black_card_text, white_card_text, "
            "round_id, won, winning_index, white_card_is_dirty, white_card_reaction"
        ),
    )
    parser.add_argument(
        "--output",
        default="slates.jsonl",
        help="Output JSONL file with frozen slates",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Number of replicates (default: 2)",
    )
    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Number of white cards per slate (default: 10)",
    )

    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(args.input)

    required = [
        "player_id",
        "black_card_text",
        "white_card_text",
        "round_id",
        "won",
        "winning_index",
        "white_card_is_dirty",
        "white_card_reaction",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize `won`
    if df["won"].dtype != bool:
        df["won"] = df["won"].astype(str).str.lower().isin(
            ["1", "true", "t", "yes", "y"]
        )

    # Normalize winning_index (CRITICAL)
    df["winning_index"] = pd.to_numeric(
        df["winning_index"], errors="coerce"
    )

    # Normalize dirty flag
    if df["white_card_is_dirty"].dtype != bool:
        df["white_card_is_dirty"] = (
            df["white_card_is_dirty"]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes", "y"])
        )

    # -----------------------------
    # Generate slates
    # -----------------------------
    total_written = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for replicate in range(1, args.replicates + 1):
            print(f"Generating replicate {replicate}/{args.replicates}")

            for round_id, group in df.groupby("round_id", sort=False):
                rng = make_replicate_rng(round_id, replicate)

                player_id = str(group.iloc[0]["player_id"])
                black_card = str(group.iloc[0]["black_card_text"])

                # Build full white-card objects
                white_cards = [
                    {
                        "text": str(row["white_card_text"]),
                        "is_dirty": bool(row["white_card_is_dirty"]),
                        "reaction": str(row["white_card_reaction"]),
                    }
                    for _, row in group.iterrows()
                ]

                if not white_cards:
                    continue

                # Sample slate
                if len(white_cards) >= args.slate_size:
                    slate = rng.sample(white_cards, args.slate_size)
                else:
                    slate = list(white_cards)

                rng.shuffle(slate)

                # Winners (text only, canonical)
                winners_all = (
                    group.loc[group["won"] == True, "white_card_text"]
                    .astype(str)
                    .tolist()
                )

                # Determine target slot (multi-blank cards)
                blanks = count_blanks(black_card)
                target_slot = None

                if blanks >= 2:
                    winning_slots = (
                        group.loc[group["won"] == True, "winning_index"]
                        .dropna()
                        .astype(int)
                        .tolist()
                    )

                    if winning_slots:
                        target_slot = Counter(winning_slots).most_common(1)[0][0]
                    else:
                        target_slot = 0

                # Filter winners if target slot is defined
                if target_slot in (0, 1):
                    winners = (
                        group.loc[
                            (group["won"] == True)
                            & (group["winning_index"] == target_slot),
                            "white_card_text",
                        ]
                        .astype(str)
                        .tolist()
                    )
                else:
                    winners = winners_all

                entry = {
                    "round_id": round_id,
                    "replicate": replicate,
                    "player_id": player_id,
                    "black_card": black_card,
                    "white_cards_slate": slate,  # LIST OF OBJECTS
                    "winners": sorted(set(winners)),
                    "target_slot": target_slot,
                    "slate_size": len(slate),
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_written += 1

    print("\n✅ Done.")
    print(f"Written entries: {total_written}")
    print(f"Output file: {args.output}")
    print("White-card metadata preserved per card.")


if __name__ == "__main__":
    main()
