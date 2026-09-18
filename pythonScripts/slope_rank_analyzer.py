#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
slope_rank_analyzer.py

Re-analyzes final candidate clusters to identify the best-performing member
within each cluster based on enrichment slope (log2 fold-change per round).

Loads only the sequences present in final reports from the Step 1 data,
keeping memory usage minimal.
"""

import pandas as pd
import os
import re
import argparse
import numpy as np
import pickle
import glob

# ==============================================================================
# === HELPER FUNCTIONS ===
# ==============================================================================

def load_pickle(path, description):
    """Safely loads a pickle file."""
    print(f"--- Loading {description} from: {os.path.basename(path)} ---")
    if not os.path.exists(path):
        print(f"  [FATAL ERROR] Path not found. Aborting.")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load. Error: {e}")
        return None

def load_final_reports(path):
    """Loads all final candidate CSVs from the Step 6 directory."""
    print(f"--- Loading Step 6 Final Candidates from: {path} ---")
    files = glob.glob(os.path.join(path, "final_candidates_*.csv"))
    if not files:
        print("  [WARNING] No final_candidates files found.")
        return {}
    
    data = {}
    for f in files:
        try:
            antigen_name = re.search(r'final_candidates_(.+)\.csv', os.path.basename(f)).group(1)
            df = pd.read_csv(f)
            df['ClusterMembers_List'] = df['ClusterMembers_List'].astype(str)
            data[antigen_name] = df
        except Exception as e:
            print(f"  [WARNING] Could not load or parse {os.path.basename(f)}. Error: {e}")
            
    print(f"  - Loaded {len(data)} antigen reports.")
    return data

def calculate_enrichment_slope(row):
    """
    Calculates enrichment slope using log2(CPM+1)-transformed values.
    Because enrichment is multiplicative, the slope on a log scale (average
    log2 fold-change per round) is a more meaningful measure than the
    arithmetic difference in raw CPM.
    """
    r1 = np.log2(row.get('R1_CPM', 0) + 1)
    r2 = np.log2(row.get('R2_CPM', 0) + 1)
    r3 = np.log2(row.get('R3_CPM', 0) + 1)
    r4_raw = row.get('R4_CPM', 0)

    if r4_raw > 0:
        r4 = np.log2(r4_raw + 1)
        return ((r2 - r1) + (r3 - r2) + (r4 - r3)) / 3
    else:
        return ((r2 - r1) + (r3 - r2)) / 2

# ==============================================================================
# === MAIN ANALYSIS FUNCTION ===
# ==============================================================================

def analyze_and_rank_by_slope(run_directory, output_dir):
    """Main function to load data, find the best performer, and generate reports."""
    
    # --- 1. Load Step 6 reports and identify all sequences we care about ---
    final_reports = load_final_reports(os.path.join(run_directory, "step6_final_reports_advanced"))
    if not final_reports:
        print("No final reports found. Exiting.")
        return

    print("\n--- [1/4] Aggregating required sequences from final reports ---")
    required_sequences = set()
    for df_report in final_reports.values():
        if not df_report.empty and 'ClusterMembers_List' in df_report.columns:
            members = df_report['ClusterMembers_List'].str.split(';').explode().unique()
            required_sequences.update(members)
    
    print(f"  - Found {len(required_sequences):,} unique member sequences to analyze.")

    # --- 2. Load Step 1 data and filter it down immediately ---
    step1_data = load_pickle(os.path.join(run_directory, "step1_all_loaded_data_with_cpms.pkl"), "Step 1 Raw Data")
    if not step1_data:
        return
        
    print("\n--- [2/4] Building filtered master DataFrame for lookups ---")
    all_dfs = [df for (antigen, _), df in step1_data.items() if antigen != 'R0_Naive' and df is not None]
    if not all_dfs:
        print("  [ERROR] No valid dataframes found in Step 1 data.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    del step1_data, all_dfs

    master_df = master_df[master_df['Sequence'].isin(required_sequences)].copy()
    print(f"  - Filtered master DataFrame to {len(master_df):,} relevant rows.")

    # --- 3. Build efficient lookup tables from the small, filtered data ---
    print("\n--- [3/4] Building efficient lookup tables ---")
    
    cpm_df = master_df.pivot_table(
        index=['AntigenID', 'Sequence'],
        columns='Round',
        values='CPM',
        fill_value=0
    ).add_prefix('R').add_suffix('_CPM').reset_index()
    cpm_lookup = cpm_df.set_index(['AntigenID', 'Sequence']).to_dict('index')
    del cpm_df

    final_round_df = master_df[master_df['Round'] == FINAL_ANALYSIS_ROUND].copy()
    if final_round_df.empty:
        print("  - No R4 data found for ranking, falling back to R3.")
        final_round_df = master_df[master_df['Round'] == 3].copy()
    del master_df
        
    final_round_df['NGS_Rank'] = final_round_df.groupby('AntigenID')['Count'].rank(method='first', ascending=False)
    ngs_rank_lookup = final_round_df.set_index(['AntigenID', 'Sequence'])['NGS_Rank'].to_dict()

    # --- 4. Process each antigen's report using the fast lookups ---
    print("\n--- [4/4] Analyzing clusters to find top performers ---")
    for antigen, df_report in final_reports.items():
        print(f"  - Processing antigen: {antigen}")
        if df_report.empty: continue
        
        new_report_rows = []
        for _, cluster_row in df_report.iterrows():
            members = cluster_row['ClusterMembers_List'].split(';')
            
            member_data = []
            for seq in members:
                cpms = cpm_lookup.get((antigen, seq), {})
                slope = calculate_enrichment_slope(cpms)
                rank = ngs_rank_lookup.get((antigen, seq), np.nan)
                entry = {'Sequence': seq, 'Slope': slope, **cpms, 'Rank': rank}
                member_data.append(entry)
            
            if not member_data: continue

            top_performer = max(member_data, key=lambda x: x['Slope'])
            rep_data = next((item for item in member_data if item['Sequence'] == cluster_row['RepSequence']), None)
            
            new_row = cluster_row.to_dict()
            new_row.update({
                'Top_Sequence': top_performer['Sequence'],
                'Top_sequence_slope': top_performer['Slope'],
                'rep_sequence_slope': rep_data['Slope'] if rep_data else np.nan,
                'Top_sequence_R1_CPM': top_performer.get('R1_CPM', 0),
                'Top_sequence_R2_CPM': top_performer.get('R2_CPM', 0),
                'Top_sequence_R3_CPM': top_performer.get('R3_CPM', 0),
                'Top_sequence_R4_CPM': top_performer.get('R4_CPM', 0),
                'Top_sequence_NGS_rank': top_performer['Rank'],
                'Rep_sequence_NGS_rank': rep_data['Rank'] if rep_data else np.nan,
            })
            new_report_rows.append(new_row)
            
        if not new_report_rows:
            print(f"    - No data to report for {antigen}.")
            continue
            
        final_df = pd.DataFrame(new_report_rows)
        
        priority_cols = [
            'ClusterMembers_Count', 'ClusterMembers_List', 
            'Top_Sequence', 'Top_sequence_slope', 
            'RepSequence', 'rep_sequence_slope',
            'Top_sequence_R1_CPM', 'Top_sequence_R2_CPM', 
            'Top_sequence_R3_CPM', 'Top_sequence_R4_CPM',
            'Top_sequence_NGS_rank', 'Rep_sequence_NGS_rank'
        ]
        
        all_cols = final_df.columns.tolist()
        other_cols = [col for col in all_cols if col not in priority_cols]
        
        new_column_order = priority_cols + other_cols
        final_df = final_df[new_column_order]
        slope_sorted_df = final_df.sort_values('Top_sequence_slope', ascending=False)
        slope_output_path = os.path.join(output_dir, f"{antigen}_final_candidates_ranked_by_slope.csv")
        slope_sorted_df.to_csv(slope_output_path, index=False)
        print(f"    - Saved report ranked by slope.")

        # Save file ranked by cluster size
        size_sorted_df = final_df.sort_values('ClusterMembers_Count', ascending=False)
        size_output_path = os.path.join(output_dir, f"{antigen}_final_candidates_ranked_by_size.csv")
        size_sorted_df.to_csv(size_output_path, index=False)
        print(f"    - Saved report ranked by size.")

# ==============================================================================
# === MAIN EXECUTION BLOCK ===
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Re-rank final candidates based on the best performing member within each cluster.")
    parser.add_argument("--run_directory", required=True, help="Path to the top-level directory of a completed pipeline run.")
    parser.add_argument("--output_dir", required=True, help="Path to a new directory for the output reports.")
    args = parser.parse_args()

    global FINAL_ANALYSIS_ROUND
    FINAL_ANALYSIS_ROUND = 4

    os.makedirs(args.output_dir, exist_ok=True)
    analyze_and_rank_by_slope(args.run_directory, args.output_dir)
    print("\n--- Analysis Finished ---")

if __name__ == '__main__':
    main()