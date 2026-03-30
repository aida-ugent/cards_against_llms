# Replication scripts for the paper "Cards Against LLMs: Benchmarking Humor Alignment in Large Language Models"

## Data collection 

This step of the analysis pipeline queries LLMs to play rounds of the game Cards Against Humanity.

Environment: requirements.txt
Script: callms_final.py
The output is expected to be saved in outputs/matches_final.jsonl 

Note that the underlying data is proprietary, but can be obtained upon request for research purposes.

## Analysis (except for human gameplay prediction baselines)

Environment: requirements_analysis.txt 

Install it by running in a terminal:
conda create -n humor_analysis
conda activate humor_analysis
conda install pip
pip install -r requirements_analysis.txt

Scripts:
0-annotate_topics.py
1-desc_stats_sociodemo.py
2-desc_stats_topic_selection.py
3-cond_logit_topics.py

CAH Lab Data is assumed to be placed in folder cah_lab_v2_data_for_research_2025_06/, containing two files:
-cah_lab_v2_data_for_research_2025_06_DEMOGRAPHIC_ANSWERS.csv
-cah_lab_v2_data_for_research_2025_06_GAMEPLAY.csv

0-annotate_topics.py annotates white cards with topics, using OLLAMA and Mixtral:8x7B.
1-desc_stats_sociodemo.py computes models' accuracies for demographic groups.
2-desc_stats_topic_selection.py computes a heatmap contrasting LLMs' white card topic selection patterns with those of human players, and the repartition of white cards in player hands.
3-cond_logit_topics.py fits conditional logistic models to predict LLM white card choice based on the card's position and topics. 

## Analysis (baselines)

Environment: environment_baselines.yml

Install it by running in a terminal
conda env create -n humor_baselines -f environment_baselines.yml
conda activate humor_baselines

Scripts: 
4-baselines.py

4-baselines.py fits two baselines to predict human behavior: a card popularity baseline and an ensemble of boosted trees predicting card choice based on white and black card embeddings, as well as white card topics.

