import argparse
import json
import os
import re
import requests
import time
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

# -----------------------------
# Environment
# -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -----------------------------
# Utilities
# -----------------------------
NUM_RE = re.compile(r"^\s*(\d{1,3})\b")


def parse_card_choice(response, white_cards):
    if not response:
        return None

    resp = response.strip()
    m = NUM_RE.match(resp)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(white_cards):
            return white_cards[idx]

    low = resp.lower()
    for c in white_cards:
        if c.lower() == low or c.lower() in low:
            return c

    best, score = None, 0
    for c in white_cards:
        words = re.findall(r"\w+", c.lower())
        s = sum(w in low for w in words)
        if s > score:
            best, score = c, s

    return best


def create_player_prompt(black_card, white_cards, target_slot):
    cards = "\n".join(f"{i+1}. {c}" for i, c in enumerate(white_cards))
    slot_hint = (
        f"Choose the funniest white card to fill BLANK #{target_slot + 1}."
        if target_slot in (0, 1)
        else "Choose the single funniest white card."
    )

    return f"""You are playing Cards Against Humanity. It's just a game so it's okay to use profanity and to be politically incorrect.
{slot_hint}

Black card:
{black_card}

Your hand:
{cards}

Respond ONLY with:
<number>. <exact card text>
"""


# -----------------------------
# Model calls
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_gpt(prompt):
    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=prompt,
            max_output_tokens=128,
            temperature=0.8,
        )
        text = response.output_text.strip()
        return text, "" if text else "empty_response"
    except Exception as e:
        return "", str(e)


def call_gemini(prompt):
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        r = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={"temperature": 0.8, "max_output_tokens": 128},
        )
        text = getattr(r, "text", "").strip()
        return text, "" if text else "empty_response"
    except Exception as e:
        return "", str(e)




def call_claude(prompt, max_retries=4, base_sleep=3):
    try:
        import anthropic
    except ImportError:
        return "", "Claude SDK not installed"

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            r = client.messages.create(
                model="claude-opus-4-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=128,
            )

            text = "\n".join(
                b.text for b in r.content if getattr(b, "type", None) == "text"
            ).strip()

            return text, "" if text else "empty_response"

        except anthropic.APIStatusError as e:
            # Claude overload
            if e.status_code == 529:
                sleep_time = base_sleep * (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Claude overloaded (529). Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            return "", f"Claude APIStatusError {e.status_code}"

        except Exception as e:
            return "", f"Claude error: {e}"

    return "", "claude_overloaded_exhausted"


def call_deepseek(prompt):
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 128,
            },
            timeout=60,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, "" if text else "empty_response"
    except Exception as e:
        return "", str(e)


def call_grok(prompt, max_retries=3, base_sleep=5):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"},
                json={
                    "model": "grok-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 128,
                },
                timeout=60,
            )

            if r.status_code == 429:
                # exponential backoff with jitter
                sleep_time = base_sleep * (2 ** attempt) + random.uniform(0, 1)
                #print(f"⏳ Grok rate-limited (429). Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text, "" if text else "empty_response"

        except requests.HTTPError as e:
            return "", f"Grok HTTP error: {e}"
        except Exception as e:
            return "", f"Grok error: {e}"

    return "", "grok_rate_limited_exhausted"


CALLS = {
    "gpt": call_gpt,
    "gemini": call_gemini,
    "claude": call_claude,
    "deepseek": call_deepseek,
    "grok": call_grok,
}


# -----------------------------
# Main with resume logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    progress_path = os.path.join(args.outdir, "progress.json")
    matches_path = os.path.join(args.outdir, "matches.jsonl")
    errors_path = os.path.join(args.outdir, "errors.jsonl")

    # Load progress
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
    else:
        progress = {}

    attempts = defaultdict(int)
    abstentions = defaultdict(int)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(matches_path, "a", encoding="utf-8") as match_f, \
         open(errors_path, "a", encoding="utf-8") as err_f:

        for line in fin:
            r = json.loads(line)
            key = f"{r['round_id']}|{r['replicate']}"

            # Skip completed rows
            if progress.get(key) == "complete":
                continue

            if args.verbose:
                print(f"▶️  Processing round {key}")

            black_card = r["black_card"]
            white_cards = [w["text"] for w in r["white_cards_slate"]]
            winners = set(r.get("winners", []))
            target_slot = r.get("target_slot")

            prompt = create_player_prompt(black_card, white_cards, target_slot)

            raw, picks, errors = {}, {}, {}

            for m in CALLS:
                text, err = CALLS[m](prompt)
                pick = parse_card_choice(text, white_cards)

                attempts[m] += 1
                raw[m] = text
                picks[m] = pick

                if err or pick is None:
                    abstentions[m] += 1
                    errors[m] = err or "parse_failure"
                    err_f.write(json.dumps({
                        "round_id": r["round_id"],
                        "replicate": r["replicate"],
                        "model": m,
                        "error": errors[m],
                    }) + "\n")

            round_valid = not errors

            match_f.write(json.dumps({
                "round_id": r["round_id"],
                "replicate": r["replicate"],
                "black_card": black_card,
                "white_cards": white_cards,
                "winners": list(winners),
                "target_slot": target_slot,
                "raw_answers": raw,
                "picks": picks,
                "round_valid": round_valid,
            }) + "\n")

            # Update progress
            progress[key] = "complete" if round_valid else "failed"
            with open(progress_path, "w") as pf:
                json.dump(progress, pf, indent=2)

            if args.verbose:
                print(f"✅ {key} -> {progress[key]}")

    print("🏁 Finished. You can safely rerun this script.")


if __name__ == "__main__":
    main()
