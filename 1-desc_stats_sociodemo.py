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

HUMOR_TYPE_SLUGS = [
    'incongruity_surprise', 'taboo_violation_shock', 'wordplay_linguistic',
    'absurdity_nonsequitur', 'superiority_mockery', 'self_deprecation',
    'dark_gallows', 'observational_relatable'
]

HARM_LEVELS = ['low', 'medium', 'high']


# %%
root_path = Path.cwd()
research_path = root_path / 'cah_lab_v2_data_for_research_2025_06/'
demo_data_path = research_path / 'cah_lab_v2_data_for_research_2025_06_DEMOGRAPHIC_ANSWERS.csv'
gameplay_path = research_path / 'cah_lab_v2_data_for_research_2025_06_GAMEPLAY.csv'
model_answers_path = root_path / 'outputs' / 'matches_final.jsonl'

output_path = root_path / 'desc_stats/'
output_path.mkdir(parents=True, exist_ok=True)


N_BOOTSTRAP = 500

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

### Merge with demographics

# Demographics
demo_raw = pd.read_csv(demo_data_path)
demo_pivot = demo_raw.drop_duplicates(subset=['player_id', 'section']).pivot(
    index='player_id', columns='section', values='answer'
)

# Gameplay (has player_id mapping)
gameplay = pd.read_csv(gameplay_path)
round_player_map = gameplay[['round_id', 'player_id']].drop_duplicates()

model_with_players = df.merge(round_player_map, on='round_id', how='left')
print(f"  After player merge: {model_with_players['player_id'].notna().sum():,} records")

# Merge with demographics
model_merged = model_with_players.merge(
    demo_pivot.reset_index()[['player_id'] + list(demo_pivot.columns)],
    on='player_id', how='left'
)
print(f"  After demo merge: {model_merged['country'].notna().sum():,} records with demographics")


model_merged.player_id.nunique(), model_merged.loc[model_merged.player_id.notnull(), 'round_id'].nunique()

model_merged.player_id.isnull().mean(), model_merged.gender.isnull().mean()

def recode_demographics(df):
    """Create regrouped demographic variables"""
    df = df.copy()
    
    politics_map = {
        'very liberal': 'liberal', 'liberal': 'liberal', 'moderate': 'moderate',
        'conservative': 'conservative', 'very conservative': 'conservative',
        'other': 'moderate', 'ignore': 'missing'
    }
    df['politics_regrouped'] = df['politics'].map(politics_map).fillna('missing')
    
    gender_map = {
        'man': 'man', 'woman': 'woman', 'non-binary': 'other',
        'other': 'other', 'ignore': 'missing'
    }
    df['gender_regrouped'] = df['gender'].map(gender_map).fillna('missing')
    
    sexual_map = {
        'heterosexual': 'heterosexual', 'bisexual': 'LGBTQ+',
        'homosexual': 'LGBTQ+', 'other': 'LGBTQ+', 'ignore': 'missing'
    }
    df['sexual_regrouped'] = df['sexual'].map(sexual_map).fillna('missing')
    
    df['country_regrouped'] = np.where(
        df['country'] == 'US', 'US',
        np.where(df['country'] == 'ignore', 'missing', 'non-US')
    )
    
    def recode_race(x):
        if pd.isna(x): return 'missing'
        x_lower = str(x).lower()
        if 'white' in x_lower: return 'white'
        elif 'black' in x_lower or 'african' in x_lower: return 'black'
        elif 'hispanic' in x_lower or 'latino' in x_lower: return 'hispanic'
        elif 'asian' in x_lower: return 'asian'
        elif 'ignore' in x_lower: return 'missing'
        else: return 'other'
    
    df['race_regrouped'] = df['race'].apply(recode_race)
    
    return df

model_merged = recode_demographics(model_merged)

model_merged.groupby('model')['matched_winner'].mean()

def compute_subgroup_accuracy(data, demo_col, model_list, n_bootstrap=500):
    """Compute accuracy and 95% bootstrap CIs for each subgroup level"""
    results = []
    unique_levels = data[demo_col].dropna().unique()
    
    for level in unique_levels:
        mask = data[demo_col] == level
        n_obs = mask.sum()
        if n_obs < 10:
            continue
            
        for model in model_list:

            sub_data = data.loc[mask & (data.model == model)].groupby('player_id')['matched_winner'].mean().reset_index()
            point_est = sub_data['matched_winner'].mean()
            
            boot_means = []
            for b in range(n_bootstrap):
                boot_sample = resample(sub_data, replace=True, 
                n_samples=len(sub_data), random_state=b)
                boot_means.append(boot_sample['matched_winner'].mean())
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            
            results.append({
                'demographic': demo_col,
                'level': level,
                'model': model,
                'accuracy': point_est,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_obs': n_obs
            })
    
    return pd.DataFrame(results)

demo_cols = ['country_regrouped', 'gender_regrouped', 'politics_regrouped', 
             'race_regrouped', 'sexual_regrouped']

print("\nComputing accuracy by sociodemographic groups...")
all_results = []
for col in demo_cols:
    if col in model_merged.columns:
        print(f"  Processing {col}...")
        results = compute_subgroup_accuracy(model_merged, col, MODEL_LIST, N_BOOTSTRAP)
        all_results.append(results)

def plot_faceted_accuracy(results, output_path, baseline_accuracies=None):
    """Create faceted plot showing all demographics"""
    
    demographics = results['demographic'].unique()
    n_demographics = len(demographics)
    
    fig, axes = plt.subplots(n_demographics, 1, figsize=(11, 2.2 * n_demographics), 
                             sharex=True, sharey=False)
    
    if n_demographics == 1:
        axes = [axes]
    
    for idx, demographic in enumerate(demographics):
        ax = axes[idx]

        if baseline_accuracies is not None:
            for model in MODEL_LIST:
                if model in baseline_accuracies:
                    ax.axvline(x=baseline_accuracies[model], linestyle='--', 
                            color=MODEL_COLORS[model], linewidth=0.8, alpha=0.5, zorder=0)

        sub_results = results[results['demographic'] == demographic].copy()
        
        level_order = sub_results.groupby('level')['accuracy'].mean().sort_values(ascending=True).index.tolist()
        sub_results['level'] = pd.Categorical(sub_results['level'], categories=level_order, ordered=True)
        sub_results = sub_results.sort_values('level')
        
        model_positions = np.linspace(-0.25, 0.25, len(MODEL_LIST))
        
        for midx, model in enumerate(MODEL_LIST):
            model_data = sub_results[sub_results['model'] == model].copy()
            if len(model_data) == 0:
                continue
                
            y_pos = np.arange(len(level_order)) + model_positions[midx]
            xerr = np.array([model_data['accuracy'] - model_data['ci_low'], 
                    model_data['ci_high'] - model_data['accuracy']])
            #xerr = np.array([proportions - ci_lows, ci_highs - proportions])
            
            ax.errorbar(model_data['accuracy'], y_pos, xerr=xerr, fmt='none',
                       ecolor=MODEL_COLORS[model], elinewidth=0.9, capsize=2.5, 
                       capthick=0.8, alpha=0.6, zorder=1)
            ax.scatter(model_data['accuracy'], y_pos, color=MODEL_COLORS[model],
                      s=35, edgecolors='white', linewidth=0.5, zorder=2, alpha=0.85)
        
        ax.set_yticks(np.arange(len(level_order)))
        ax.set_yticklabels(level_order, fontsize=9)
        ax.set_title(demographic.replace("_regrouped", "").replace("_", " ").title(), 
                    fontsize=10, fontweight='bold', pad=8, loc='left')
        ax.grid(True, axis='x', linestyle='--', alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 0.4)
        ax.set_ylim(-0.8, len(level_order) - 0.2)
        
        if idx == 0:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=MODEL_COLORS[m], 
                                         markersize=8, label=m.upper(),
                                         markeredgecolor='white', markeredgewidth=0.5) 
                              for m in MODEL_LIST]
            ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
                     framealpha=0.95, fontsize=8, labelspacing=0.25, handlelength=1)
    
    fig.text(0.5, 0.02, 'Accuracy (Proportion Correct)', ha='center', fontsize=11, fontweight='medium')
    fig.text(0.03, 0.5, 'Subgroup', va='center', rotation='vertical', fontsize=11, fontweight='medium')

    plt.tight_layout(rect=[0.04, 0.03, 1, 0.97])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


baseline_accuracies = (model_merged[model_merged['player_id'].notna()]
                       .groupby(['model', 'player_id'])['matched_winner']
                       .mean()
                       .groupby('model')
                       .mean()
                       .to_dict())

faceted_path = output_path / 'accuracy_all_demographics_faceted.png'
plot_faceted_accuracy(pd.concat(all_results), faceted_path, baseline_accuracies)

model_merged[model_merged.model == "gpt"].player_id.isnull().mean()

model_merged[(model_merged['player_id'].notna()) & (model_merged.model == "gpt")].player_id.nunique()


