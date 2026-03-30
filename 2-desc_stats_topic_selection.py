import numpy as np
import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from pathlib import Path
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Color scheme from notebook
MODEL_COLORS = {
    'gpt': '#10a37f', 
    'gemini': '#4285f4', 
    'claude': '#d97706', 
    'deepseek': '#7c3aed', 
    'grok': '#ef4444'
}
MODEL_LIST = ['gpt', 'gemini', 'deepseek', 'claude', 'grok']

TOPIC_SLUGS = [
    'bodily_functions_gross_out', 'sexual_themes', 'violence_crime_death_threat',
    'politics_ideology_society_culture', 'drugs_alcohol_risky_behavior',
    'pop_culture_media_consumerism', 'food_eating_consumables', 'animals_nature_creatures',
    'absurdism_surreal_nonsensical', 'identity_demographics_traits',
    'family_relationships_everyday', 'emotional_states_mental_health',
    'supernatural_cosmic_paranormal', 'money_work_technology_modern',
    'random_objects_miscellaneous'
]

root_path = Path.cwd()
research_path = root_path / 'cah_lab_v2_data_for_research_2025_06/'
demo_data_path = research_path / 'cah_lab_v2_data_for_research_2025_06_DEMOGRAPHIC_ANSWERS.csv'
model_answers_path = root_path / 'outputs' / 'matches_final.jsonl'
annotation_path = root_path / "annot_data" / "white_card_annotations_topics_final.csv"

output_path = root_path / 'desc_stats/'
output_path.mkdir(parents=True, exist_ok=True)


# Model answers

with open(model_answers_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Filter valid rounds
valid = [r for r in data if r['round_valid']]

# Flat dataframe: one row per (model, round, replicate)
records = []
for r in valid:
    winner = r['winners'][0] if r['winners'] else None
    for m in MODEL_LIST:
        pick = r['picks'].get(m)
        records.append({
            'round_id': r['round_id'],
            'replicate': r['replicate'],
            'black_card': r['black_card'],
            'model': m,
            'pick': pick,
            'winner': winner,
            'matched_winner': (pick == winner) if pick else False,
        })
df = pd.DataFrame(records)

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


annotations = pd.read_csv(annotation_path)
print(f"  Annotated cards: {len(annotations)}")

# Verify new schema columns exist
required_cols = ['white_card_text', 'topics']

# Create binary columns from list-based annotations BEFORE merge
for topic in TOPIC_SLUGS:
    col_name = f'has_topic_{topic}'
    annotations[col_name] = annotations['topics'].apply(
        lambda x: 1 if topic in safe_parse_list(x) else 0
    )

binary_cols = [c for c in annotations.columns 
if c.startswith('has_') or c.startswith('harm_')]
binary_cols = [c for c in binary_cols if c != 'harm_potential']
annotations = annotations[['white_card_text'] + binary_cols]


long = pd.merge(df, annotations, left_on ='pick', right_on='white_card_text')
long = pd.merge(long, annotations, left_on='winner', 
                right_on='white_card_text', 
                suffixes=('', '_winner'))

all_white_cards = []
for r in valid:
    all_white_cards.extend(r.get('white_cards', []))

# Create dataframe for all white cards
all_white_df = pd.DataFrame({'white_card_text': all_white_cards})
all_white_annotated = pd.merge(all_white_df, annotations, 
                                left_on='white_card_text', 
                                right_on='white_card_text', 
                                how='left')

heatmap_data = long.groupby('model')[binary_cols].mean()
heatmap_data.loc['Humans'] = long[[c + '_winner' for c in binary_cols]].mean().values
heatmap_data.loc['All Valid Rounds'] = all_white_annotated[binary_cols].mean().values

heatmap_data = heatmap_data.T
heatmap_data = heatmap_data[['Humans'] + MODEL_LIST + ['All Valid Rounds']]
heatmap_cols = ['Humans'] + MODEL_LIST + ['All Valid Rounds']

topic_cols = [c for c in heatmap_data.index if c.find('topic') != -1]

LABEL_MAP = {
    # TOPICS (bodily/sexual/violence)
    'has_topic_bodily_functions_gross_out': 'Bodily/Gross',
    'has_topic_sexual_themes': 'Sexual',
    'has_topic_violence_crime_death_threat': 'Violence/Crime',
    
    # TOPICS (society/culture)
    'has_topic_politics_ideology_society_culture': 'Politics/Society',
    'has_topic_drugs_alcohol_risky_behavior': 'Drugs/Risk',
    'has_topic_pop_culture_media_consumerism': 'Pop Culture',
    
    # TOPICS (everyday life)
    'has_topic_food_eating_consumables': 'Food/Eating',
    'has_topic_animals_nature_creatures': 'Animals/Nature',
    'has_topic_absurdism_surreal_nonsensical': 'Absurd/Surreal',
    'has_topic_identity_demographics_traits': 'Identity/Demographics',
    'has_topic_family_relationships_everyday': 'Family/Relationships',
    
    # TOPICS (internal/abstract)
    'has_topic_emotional_states_mental_health': 'Emotions/Mental Health',
    'has_topic_supernatural_cosmic_paranormal': 'Supernatural/Cosmic',
    'has_topic_money_work_technology_modern': 'Money/Work/Tech',
    'has_topic_random_objects_miscellaneous': 'Miscellaneous',
}

# Apply to heatmap index
heatmap_data.index = [LABEL_MAP.get(idx, idx) for idx in heatmap_data.index]

FONTSIZE = 16
# Create the heatmap
plt.figure(figsize=(16, 10))
mask = heatmap_data.isna()
ax = sns.heatmap(heatmap_data, annot=True, annot_kws={'size': FONTSIZE}, fmt='.2f', cmap='RdYlGn',
                 vmin=0, vmax=1, center=0.5, mask=mask,
                 cbar_kws={'label': 'Proportion'}, linewidths=0.5, linecolor='white')

# Set labels and title
plt.xlabel('', fontsize=FONTSIZE, fontweight='bold')

# Format x-axis labels
n_cols = len(heatmap_data.columns)
ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)

for i in [1, heatmap_data.shape[1] - 1]:
    ax.axvline(i, color='white', lw=5)

# Adjust layout to make room for side labels
plt.tight_layout()
plt.subplots_adjust(left=0.18)  # Extra space on left for group labels

# Save and display
plt.savefig(output_path / 'card_choices.png', 
            dpi=150, bbox_inches='tight')
plt.close()


