import json
import time
from pathlib import Path
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.discrete.conditional_models import ConditionalLogit


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

TOPIC_SLUGS = [
    'bodily_functions_gross_out', 'sexual_themes', 'violence_crime_death_threat',
    'politics_ideology_society_culture', 'drugs_alcohol_risky_behavior',
    'pop_culture_media_consumerism', 'food_eating_consumables', 'animals_nature_creatures',
    'absurdism_surreal_nonsensical', 'identity_demographics_traits',
    'family_relationships_everyday', 'emotional_states_mental_health',
    'supernatural_cosmic_paranormal', 'money_work_technology_modern',
    'random_objects_miscellaneous'
]

MODEL_LIST = ['gpt', 'gemini', 'claude', 'deepseek', 'grok']

TEST_SIZE = 0.2
RANDOM_STATE = 42

root_path = Path.cwd()

root_path = Path.cwd()

model_answers_path = root_path / 'outputs' / 'matches_final.jsonl'
annotations_path = root_path / "annot_data" / "white_card_annotations_topics_final.csv"
annotation_path = root_path / "annot_data" / "white_card_annotations_topics_final.csv"

def safe_parse_list(val):
    """
    Safely parse a list from CSV string or return as-is if already a list.
    Handles both JSON format (double quotes) and Python repr format (single quotes).
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
    return []

with open(model_answers_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Filter valid rounds
valid = [r for r in data if r['round_valid']]

# Flat dataframe: one row per (model, round, replicate)
records = []
for c, r in enumerate(valid):
    winner = r['winners'][0] if r['winners'] else None
    for m in MODEL_LIST:
        pick = r['picks'].get(m)
        records.append({
            'round_id': r['round_id'],
            'choice_set': c,
            'replicate': r['replicate'],
            'black_card': r['black_card'],
            'model': m,
            'pick': pick,
            'winner': winner,
            'matched_winner': (pick == winner) if pick else False,
            'white_cards' : r["white_cards"],
            "uid": str(r["round_id"]) + "_" + str(r["target_slot"])
        })
initial_df = pd.DataFrame(records)

print("Loading annotations...")

annotations = pd.read_csv(annotations_path)
print(f"  Annotated cards: {len(annotations)}")

# Verify new schema columns exist
required_cols = ['white_card_text', 'topics']

# Create binary columns from list-based annotations BEFORE merge
for topic in TOPIC_SLUGS:
    col_name = f'has_topic_{topic}'
    annotations[col_name] = annotations['topics'].apply(
        lambda x: 1 if topic in safe_parse_list(x) else 0
    )

binary_cols = [c for c in annotations.columns if c.startswith('has_')]
feature_cols = binary_cols
annotations = annotations[['white_card_text'] + binary_cols]


annotation_lookup = (
    annotations
    .set_index("white_card_text")[binary_cols]
    .to_dict("index")
)


rows = []

for rownum, r in initial_df.iterrows():

    white_cards = r["white_cards"]
    chosen = r.get("pick", {})
    model = r.get('model')

    for pos, card in enumerate(white_cards):

        feats = annotation_lookup.get(card, {})

        row = {
            "choice_set": r['choice_set'],
            "entity": model,
            "chosen": int(card == chosen),
            "position": pos,
            "round_id": r["round_id"],
            "uid": r["uid"]
        }

        for f in feature_cols:
            row[f] = feats.get(f, 0)

        rows.append(row)

long_df = pd.DataFrame(rows)

print("Rows:", len(long_df))

feature_sums = long_df[feature_cols].sum(axis=0)
MIN_FEATURE_ONES = 3000
valid_feature_mask = (feature_sums >= MIN_FEATURE_ONES * 10) |  (len(long_df) - feature_sums <= MIN_FEATURE_ONES * 10)
valid_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_feature_mask.values[i]]
n_dropped = len(feature_cols) - len(valid_feature_cols)
feature_cols = valid_feature_cols
print(feature_cols)

for i in range(1, 10): # OHE of position, drop 0 as ref
    long_df['position_' + str(i)] = long_df['position'] == i
    feature_cols += ['position_' + str(i)]


choice_sets = long_df["round_id"].unique()

train_rounds, test_rounds = train_test_split(
    long_df["round_id"].unique(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


results = []

trained_models = {}
model_predictions = {}

for model in MODEL_LIST:
    
    data = long_df[long_df["entity"] == model]

    print("\nModel:", model)

    train = data[data["round_id"].isin(train_rounds)]
    test = data[data["round_id"].isin(test_rounds)]

    X_train = train[feature_cols].astype(float)
    y_train = train["chosen"]

    model_cl = ConditionalLogit(
        y_train,
        X_train,
        groups=train["choice_set"]
    )

    fit = model_cl.fit(disp=False)
    trained_models[model] = fit

    beta = fit.params.values

    X_test = test[feature_cols].values
    utilities = X_test @ beta
    
    test = test.copy()
    test["utility"] = utilities
    
    pred = (
        test
        .sort_values(["choice_set", "utility"], ascending=False)
        .groupby("choice_set")
        .head(1)
    )

    accuracy = pred["chosen"].mean()

    print(f"Accuracy: {round(accuracy, 3)}")


    results.append((model, accuracy))