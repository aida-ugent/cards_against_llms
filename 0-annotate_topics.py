#!/usr/bin/env python3
"""
Cards Against Humanity White Card Annotation Script
Model: Mixtral 8x7B | Processing: Single-card (no batching)
"""

import pandas as pd
import json
import ast
import os
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ollama import Client, chat
from tqdm import tqdm
from collections import Counter

load_dotenv()

MODEL_NAME = "mixtral:8x7b"
TEMPERATURE = 0.0
MAX_RETRIES = 3
NUM_THREADS = 6
CONTEXT_WINDOW = 8192

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# ================= PATHS (RELATIVE TO CURRENT DIRECTORY) =================
root_path = Path(os.getenv('HUMOR_DATA_ROOT', Path.cwd()))
research_path = root_path / 'cah_lab_v2_data_for_research_2025_06/'
gameplay_path = research_path / 'cah_lab_v2_data_for_research_2025_06_GAMEPLAY.csv'
save_path = root_path / "annot_data/"
save_path.mkdir(parents=True, exist_ok=True)

VALID_TOPICS = [
    "bodily_functions_gross_out", "sexual_themes", "violence_crime_death_threat",
    "politics_ideology_society_culture", "drugs_alcohol_risky_behavior",
    "pop_culture_media_consumerism", "food_eating_consumables", "animals_nature_creatures",
    "absurdism_surreal_nonsensical", "identity_demographics_traits",
    "family_relationships_everyday", "emotional_states_mental_health",
    "supernatural_cosmic_paranormal", "money_work_technology_modern",
    "random_objects_miscellaneous"
]

REQUIRED_KEYS = ["topics"]

SYSTEM_PROMPT = '''
You are an expert annotator labeling "white cards" from Cards Against Humanity for humor research.

### TASK
Analyze the input card and return a JSON object matching the schema below.

### OUTPUT SCHEMA (STRICT)
{
  "topics": ["slug1", "slug2"],           // 1-4 items from TOPICS list
}

### TOPICS (use slugs exactly; select 1-4)
1. bodily_functions_gross_out         // Anatomy, bodily fluids, gross-out physical humor
2. sexual_themes                      // Sexual content: innuendo, explicit acts, relationships
3. violence_crime_death_threat        // Physical harm, mortality, criminal acts, threats
4. politics_ideology_society_culture  // Government, activism, social norms, cultural commentary
5. drugs_alcohol_risky_behavior       // Substance use, addiction, reckless actions
6. pop_culture_media_consumerism      // Celebrities, movies, memes, brands, viral trends
7. food_eating_consumables            // Meals, ingredients, dining, consumption
8. animals_nature_creatures           // Wildlife, pets, ecosystems, biological refs
9. absurdism_surreal_nonsensical      // Illogical juxtapositions, nonsense, anti-humor
10. identity_demographics_traits      // Race, gender, age, disability, sexuality, nationality
11. family_relationships_everyday     // Parenting, friendships, domestic life, mundane interactions
12. emotional_states_mental_health    // Anxiety, joy, depression, coping, psychological framing
13. supernatural_cosmic_paranormal    // Ghosts, aliens, magic, existential cosmic themes
14. money_work_technology_modern      // Jobs, finance, digital life, institutional critique
15. random_objects_miscellaneous      // Concrete items/concepts not captured above

### RULES
1. Use slugs exactly as written (underscores, lowercase, no spaces).
2. Output ONLY valid JSON. No markdown, no preamble, no extra text.
'''

USER_PROMPT_TEMPLATE = '''
### INPUT CARD
"{card_text}"

### OUTPUT
'''

def get_card_hash(text: str) -> str:
    """Generate hash for caching/deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


def safe_parse_list(val):
    """
    Safely parse a list from CSV string or return as-is if already a list.
    Handles both JSON format (double quotes) and Python repr format (single quotes).
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            # Try JSON first (proper double-quote format)
            return json.loads(val)
        except json.JSONDecodeError:
            try:
                # Fallback: Python literal eval (handles single quotes)
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
    return []


def validate_annotation(ann: dict) -> tuple[bool, str]:
    """Validate annotation against schema. Returns (is_valid, error_message)."""
    # Check required keys
    missing = set(REQUIRED_KEYS) - set(ann.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    
    # Validate topics
    if not isinstance(ann["topics"], list):
        return False, "topics must be a list"
    if not (1 <= len(ann["topics"]) <= 4):
        return False, f"topics must have 1-4 items, got {len(ann['topics'])}"
    if not all(t in VALID_TOPICS for t in ann["topics"]):
        invalid = set(ann["topics"]) - set(VALID_TOPICS)
        return False, f"Invalid topic slugs: {invalid}"
    
    return True, ""


def default_annotation():
    """Return a minimal valid default annotation for fallback."""
    return {
        "topics": ["random_objects_miscellaneous"],
    }


def annotate_card(card_text: str, client: Client, log_raw: bool = False) -> tuple[dict, str]:
    """
    Annotate a single card with retry logic.
    Returns (annotation, raw_response) for logging.
    """
    # Sanitize input
    safe_text = card_text.strip()[:1000]
    prompt = USER_PROMPT_TEMPLATE.format(card_text=safe_text)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": TEMPERATURE,
                    "num_threads": NUM_THREADS,
                    "num_ctx": CONTEXT_WINDOW,
                },
                format="json"
            )
            
            output_text = response.message.content.strip()
            raw_response = output_text
            
            # Parse JSON
            try:
                annotation = json.loads(output_text)
            except json.JSONDecodeError as e:
                if log_raw:
                    print(f"JSON parse error (attempt {attempt+1}): {e}")
                    print(f"Raw response: {output_text[:300]}...")
                continue
            
            # Validate against schema
            is_valid, error = validate_annotation(annotation)
            if not is_valid:
                if log_raw:
                    print(f"Validation error (attempt {attempt+1}): {error}")
                    print(f"Annotation: {annotation}")
                continue
            
            return annotation, raw_response
            
        except Exception as e:
            if log_raw:
                print(f"Annotation attempt {attempt+1} failed for card '{safe_text[:50]}...': {e}")
            if attempt == MAX_RETRIES - 1:
                if log_raw:
                    print("→ Using default annotation")
                return default_annotation(), "ERROR: All retries exhausted"
    
    return default_annotation(), "ERROR: Unexpected failure"


def load_existing_annotations(save_path: Path) -> dict:
    """Load existing annotations to avoid re-processing."""
    interim_file = save_path / "white_card_annotations_topics_interim.csv"
    if interim_file.exists():
        df = pd.read_csv(interim_file)
        return {
            get_card_hash(row['white_card_text']): row.to_dict()
            for _, row in df.iterrows()
        }
    return {}


def print_summary(final_df: pd.DataFrame):
    """Print annotation summary statistics with robust list parsing."""
    print("\n" + "=" * 60)
    print("Annotation Complete!")
    print("=" * 60)
    
    print(f"\n Annotation Summary:")
    print(f"  Total cards: {len(final_df)}")

    print(f"\n  Top 5 topics:")
    all_topics = []
    for val in final_df['topics']:
        all_topics.extend(safe_parse_list(val))
    
    if all_topics:
        topic_counts = Counter(all_topics)
        for topic, count in topic_counts.most_common(5):
            print(f"    {topic}: {count}")
    else:
        print("    (no topics found)")
    

def main():
    print("=" * 60)
    print("CAH White Card Annotation Script")
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Root Path: {root_path.resolve()}")
    print(f"Save Path: {save_path.resolve()}")
    print("=" * 60)
    
    # Initialize Ollama client
    client = Client(host=OLLAMA_HOST)
    models = client.list()
    
    gameplay = pd.read_csv(gameplay_path)
    white_cards = gameplay['white_card_text'].dropna().unique().tolist()
    print(f"Found {len(white_cards)} unique white cards.")
    
    # Load existing annotations (resume capability)
    existing = load_existing_annotations(save_path)
    processed_hashes = set(existing.keys())
    print(f"Resuming from {len(existing)} previously annotated cards.")
    
    # Filter out already processed cards
    cards_to_process = [
        card for card in white_cards 
        if get_card_hash(card) not in processed_hashes
    ]
    print(f"Cards remaining to process: {len(cards_to_process)}")
    
    # Initialize results with existing annotations
    results = list(existing.values())
    
    # Process cards with progress bar
    if cards_to_process:
        print("\n Starting annotation...")
        for card_text in tqdm(cards_to_process, desc="Annotating cards"):
            annotation, raw_response = annotate_card(card_text, client, log_raw=False)
            
            results.append({
                "white_card_text": card_text,
                "card_hash": get_card_hash(card_text),
                **annotation,
                "annotated_at": datetime.now().isoformat(),
            })
            
            # Incremental save every 10 cards
            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(
                    save_path / "white_card_annotations_topics_interim.csv", 
                    index=False
                )
    
    final_df = pd.DataFrame(results)
    output_file = save_path / "white_card_annotations_topics_final.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n Results saved to: {output_file}")
    
    # Print summary
    print_summary(final_df)


if __name__ == "__main__":
    main()