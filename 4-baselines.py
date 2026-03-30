"""
XGBoost Human Baseline - 
Train: Full gameplay (excluding LLM sample rounds/players)
Features: white & black card embeddings, card topics
"""

import numpy as np
import pandas as pd
import json
import ast
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import time
    
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only
warnings.filterwarnings('ignore')

root_path = Path.cwd()
research_path = root_path / 'cah_lab_v2_data_for_research_2025_06/'
gameplay_path = research_path / 'cah_lab_v2_data_for_research_2025_06_GAMEPLAY.csv'
model_answers_path = root_path / 'outputs' / 'matches_final.jsonl'
annotation_path = root_path / "annot_data" / "white_card_annotations_topics_final.csv"

TOPIC_SLUGS = [
    'bodily_functions_gross_out', 'sexual_themes', 'violence_crime_death_threat',
    'politics_ideology_society_culture', 'drugs_alcohol_risky_behavior',
    'pop_culture_media_consumerism', 'food_eating_consumables', 'animals_nature_creatures',
    'absurdism_surreal_nonsensical', 'identity_demographics_traits',
    'family_relationships_everyday', 'emotional_states_mental_health',
    'supernatural_cosmic_paranormal', 'money_work_technology_modern',
    'random_objects_miscellaneous'
]
RANDOM_STATE = 42
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
HP_TUNING = True
N_HP_ITERATIONS = 50
CV_FOLDS = 3
SCORING = 'roc_auc'

with open(model_answers_path, 'r') as f:
    llm_data = [json.loads(line) for line in f]
valid_llm_rounds = [r for r in llm_data if r['round_valid']]

# Build test dataframe: ONE ROW PER CARD OPTION
test_records = []
for r in valid_llm_rounds:
    round_id = r['round_id']
    target_slot = r.get('target_slot', 0)
    black_card = r['black_card']
    white_cards = r.get('white_cards', [])
    winners = r['winners'] if r['winners'] else []
    winner_card = winners[0] if len(winners) > 0 else None
    
    for pos, card in enumerate(white_cards):
        test_records.append({
            'round_id': round_id,
            'target_slot': target_slot,
            'choice_set_id': f"{round_id}_{target_slot}",
            'black_card_text': black_card,
            'white_card_text': card,
            'position': pos,
            'won': 1 if card == winner_card else 0
        })

df_test = pd.DataFrame(test_records)
test_round_ids = set(df_test['round_id'].unique())
print(f"  Valid LLM rounds (test set): {len(test_round_ids)}")
print(f"  Test rows (card options): {len(df_test)}")
print(f"  Test win rate: {df_test['won'].mean():.3f}")

gameplay = pd.read_csv(gameplay_path, usecols=['player_id', 'round_id', 'white_card_text', 'black_card_text', 'won'])
print(f"  Total gameplay rows: {len(gameplay)} ({gameplay['round_id'].nunique()} rounds)")

# Identify players involved in LLM rounds (to exclude from training)
llm_players = set(
    gameplay[gameplay['round_id'].isin(test_round_ids)]['player_id'].dropna().unique()
)
print(f"  Players in LLM rounds: {len(llm_players)}")

# Create training set: Exclude LLM rounds AND players in those rounds
train_mask = (~gameplay['round_id'].isin(test_round_ids)) & \
             (~gameplay['player_id'].isin(llm_players))
df_train = gameplay[train_mask].copy()
print(f"  Training rows: {len(df_train)} ({df_train['round_id'].nunique()} rounds)")
print(f"  Training win rate: {df_train['won'].mean():.3f}")
print(df_train.player_id.isnull().mean())

print("\nProcessing Annotations...")

def safe_parse_list(val):
    if isinstance(val, list): return val
    if isinstance(val, str) and val.strip():
        try: return json.loads(val)
        except json.JSONDecodeError:
            try: return ast.literal_eval(val)
            except: return []
    return []

annotations = pd.read_csv(annotation_path)
for topic in TOPIC_SLUGS:
    col_name = f'has_topic_{topic}'
    annotations[col_name] = annotations['topics'].apply(
        lambda x: 1 if topic in safe_parse_list(x) else 0
    )

topic_cols = [f'has_topic_{t}' for t in TOPIC_SLUGS]
annotations = annotations[['white_card_text'] + topic_cols]

# Nb: player IDs get mapped to 0 here
df_train = df_train.merge(annotations, on='white_card_text', how='left').fillna(0)
df_test = df_test.merge(annotations, on='white_card_text', how='left').fillna(0)

print("\n Generating Embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
    
# Encode ALL unique cards and prompts (train + test)
unique_cards = pd.concat([df_train['white_card_text'], df_test['white_card_text']]).unique()
unique_prompts = pd.concat([df_train['black_card_text'], df_test['black_card_text']]).unique()
    
print(f"  Encoding {len(unique_cards)} unique cards...")
card_emb_dict = dict(zip(unique_cards, embedder.encode(unique_cards, show_progress_bar=False)))
print(f"  Encoding {len(unique_prompts)} unique prompts...")
prompt_emb_dict = dict(zip(unique_prompts, embedder.encode(unique_prompts, show_progress_bar=False)))
    
def add_embs(df):
    df['card_emb'] = df['white_card_text'].map(card_emb_dict)
    df['prompt_emb'] = df['black_card_text'].map(prompt_emb_dict)
    return df
    
df_train = add_embs(df_train)
df_test = add_embs(df_test)
    
emb_dim = 384
card_cols = [f'card_emb_{i}' for i in range(emb_dim)]
prompt_cols_emb = [f'prompt_emb_{i}' for i in range(emb_dim)]
    
df_train[card_cols] = pd.DataFrame(df_train['card_emb'].tolist(), index=df_train.index)
df_train[prompt_cols_emb] = pd.DataFrame(df_train['prompt_emb'].tolist(), index=df_train.index)
df_test[card_cols] = pd.DataFrame(df_test['card_emb'].tolist(), index=df_test.index)
df_test[prompt_cols_emb] = pd.DataFrame(df_test['prompt_emb'].tolist(), index=df_test.index)
    
feature_cols = card_cols + prompt_cols_emb + topic_cols

print("\nTraining XGBoost with Hyperparameter Tuning...")

X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train['won'].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)
y_test = df_test['won'].values.astype(np.float32)
choice_set_ids_test = df_test['choice_set_id'].values

print(f"  Training matrix: {X_train.shape}")
print(f"  Test matrix: {X_test.shape}")
print(f"  Class imbalance: {y_train.mean():.3f} (positive rate)")

# Parameter distribution 
param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],
        'min_child_weight': [3, 5, 7, 10],
        'scale_pos_weight': [7, 8, 9, 10, 11],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
# Base model
base_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1,
        verbosity=0
    )
    
# Cross-validation strategy (stratified to preserve class balance)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
# Randomized search
search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_HP_ITERATIONS,
        scoring=SCORING,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
start_time = time.time()
search.fit(X_train, y_train)
tuning_time = time.time() - start_time
    
print(f"\n  Hyperparameter tuning completed in {tuning_time:.1f}s")
print(f"  Best CV {SCORING}: {search.best_score_:.4f}")
print(f"  Best parameters:")
for param, value in search.best_params_.items():
    print(f"    - {param}: {value}")
    
# Use the best model (already refit on all training data by RandomizedSearchCV)
model = search.best_estimator_
best_params = search.best_params_


print("\nEvaluating on Small Sample...")

preds = model.predict_proba(X_test)[:, 1]

# Choice-set-level Top-1 Accuracy 
choice_set_accs = []
for cs_id in np.unique(choice_set_ids_test):
    mask = choice_set_ids_test == cs_id
    p = preds[mask]
    y = y_test[mask]
    if len(p) > 0 and (y == 1).sum() > 0:
        if p.argmax() == np.where(y == 1)[0][0]:
            choice_set_accs.append(1)
        else:
            choice_set_accs.append(0)

xgb_accuracy = np.mean(choice_set_accs) if choice_set_accs else 0
xgb_auc = roc_auc_score(y_test, preds)

# Popularity Baseline (for comparison)
card_pop_baseline = df_train.groupby('white_card_text')['won'].mean()
df_test['pop_baseline'] = df_test['white_card_text'].map(card_pop_baseline).fillna(0)
baseline_accs = []
for cs_id in np.unique(df_test['choice_set_id']):
    sub = df_test[df_test['choice_set_id'] == cs_id]
    if len(sub) > 0:
        pred_card = sub.loc[sub['pop_baseline'].idxmax(), 'white_card_text']
        actual = sub.loc[sub['won'] == 1, 'white_card_text'].values
        if len(actual) > 0 and pred_card == actual[0]:
            baseline_accs.append(1)
        else:
            baseline_accs.append(0)
baseline_accuracy = np.mean(baseline_accs) if baseline_accs else 0


print("\n" + "=" * 70)
print("RESULTS (Upper Bound on Humor Model Performance)")
print("=" * 70)
print(f"Test Set: ALL valid LLM rounds (built from matches_final.jsonl)")
print(f"  - Test rounds: {len(test_round_ids)}")
print(f"  - Test rows (card options): {len(df_test):,}")
print(f"\nTrain Set: Gameplay excluding LLM rounds + LLM players")
print(f"  - Training rows: {len(df_train):,}")
print(f"  - Training rounds: {df_train['round_id'].nunique():,}")
print(f"  - Excluded players: {len(llm_players):,}")
print(f"\nXGBoost Top-1 Acc (per slot):  {xgb_accuracy:.4f}")
print(f"XGBoost AUC-ROC:               {xgb_auc:.4f}")
print(f"Popularity Baseline:           {baseline_accuracy:.4f}")