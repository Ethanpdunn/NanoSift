import pandas as pd
import os
import glob
import re
import multiprocessing
from functools import partial
from scipy.stats import fisher_exact, spearmanr
import numpy as np
import subprocess
import shutil
import time
import pickle
from statsmodels.stats.multitest import multipletests
import tempfile
import math
from collections import Counter, defaultdict
import itertools
import argparse
import ast
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import io


# --- Attempt to import optional libraries ---
try:
    from Bio import AlignIO
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:
    print("CRITICAL ERROR: BioPython library not found. Please install it: pip install biopython")
    exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
except ImportError:
    print("Warning: matplotlib or seaborn not found. Plotting will be disabled.")
    plt, sns = None, None

try:
    import logomaker
except ImportError:
    print("Warning: logomaker library not found. PSSM logo generation will be skipped.")
    logomaker = None

# ==============================================================================
# === OVERALL CONFIGURATION ===
# ==============================================================================

# --- Pipeline Input & Path Configuration ---
RANKED_SEQS_DIR = "/projects/wg-biopanning/biopanning_epdunn/clustering/ranked_seqs_dedup3"
NAIVE_LIBRARY_R0_FILE = "/projects/wg-biopanning/results_04.16.25/nextSeqRnd0/nextSeqRnd0_merged_concat_mod3_dedup1_woUMI_dedup2_fullCDRs_dedup3_counts.csv"
MMSEQS_EXECUTABLE = "mmseqs"
MAFFT_EXECUTABLE = "mafft"

# --- Step II: Sequence Filtering Parameters ---
CPM_CUTOFF = 0.0
CPM_CUTOFF_R3 = 5.0
# Raw-read floor applied alongside CPM_CUTOFF_R3 so the gatekeeper does not
# loosen at low sequencing depth. At production depth (>= ~1M reads / round)
# the CPM term is strictly tighter than this count term and the filter is
# unchanged. Below ~1M reads / round, this floor prevents singleton-level
# noise from inflating the candidate pool. Set to 0 to disable.
MIN_R3_READS = 5
FISHER_P_VALUE_THRESHOLD_FOR_DECREASE = 0.05

# --- Step III: Clustering Parameters ---
MIN_SEQ_ID = 0.8
COVERAGE = 0.9
COV_MODE = 0
CLUSTER_MODE = 0

# --- Step V: Specificity Analysis Parameters ---
Q_VALUE_CUTOFF_SPECIFICITY = 0.05
FINAL_ANALYSIS_ROUND = 4 # The final round to use for enrichment, specificity, and reporting
FOLD_CHANGE_CUTOFF_SPECIFICITY = 3.0
CROSS_REACTIVITY_EXCLUSION_GROUPS = [] 

# --- Step VI & VII: Reporting & Visualization Parameters ---
DEFAULT_AA_BACKGROUND = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137, 'Q': 0.0393, 'E': 0.0675,
    'G': 0.0707, 'H': 0.0227, 'I': 0.0596, 'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386,
    'P': 0.0470, 'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687
}
TOP_N_CLUSTERS_TO_PLOT = 20
TOP_N_CLUSTERS_FOR_LOGO = 3
ANTIGEN_SIMILARITY_ROUND = 3
MIN_CLUSTER_SIZE_FOR_METRICS = 3
ANTIGEN_SIMILARITY_JACCARD_CPM_THRESHOLD = 5.0
CDR1_LEN, CDR2_LEN = 7, 8
CDR3_START_POS = CDR1_LEN + CDR2_LEN


# --- System & Global Variables ---
NUM_PROCESSES = 8
if 'SLURM_CPUS_PER_TASK' in os.environ:
    try:
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        if num_cpus > 0: NUM_PROCESSES = num_cpus
    except ValueError:
        print(f"WARNING: Could not parse SLURM_CPUS_PER_TASK. Using default: {NUM_PROCESSES}")
else:
    detected_cpus = os.cpu_count()
    if detected_cpus: NUM_PROCESSES = min(NUM_PROCESSES, detected_cpus)

CURRENT_RUN_OUTPUT_DIR = ""
COVERAGE = 0.8

# ==============================================================================
# === HELPER & WORKER FUNCTIONS ===
# ==============================================================================

### --- Generic Helpers ---

def load_pickle(file_path, file_description="Pickle file"):
    print(f"  Loading: {os.path.basename(file_path)} ({file_description})", end="...")
    if not os.path.exists(file_path):
        print(" File not found!")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(" Done.")
        return data
    except Exception as e:
        print(f" Error loading: {e}")
        return None

def parse_cdrs(sequence):
    if not isinstance(sequence, str) or len(sequence) < CDR3_START_POS + 1:
        return "N/A", "N/A", "N/A", 0
    cdr1, cdr2, cdr3 = sequence[:CDR1_LEN], sequence[CDR1_LEN:CDR3_START_POS], sequence[CDR3_START_POS:]
    return cdr1, cdr2, cdr3, len(cdr3)

### --- Step I Helpers ---

def parse_filename_info_step1(filename):
    match = re.match(r"(.+?)_Rnd(\d+)_.*\.csv", filename, re.IGNORECASE)
    if match: return match.group(1), int(match.group(2))
    match_old1 = re.match(r"(.+?)_R(\d+)_.*\.csv", filename, re.IGNORECASE)
    if match_old1: return match_old1.group(1), int(match_old1.group(2))
    match_old2 = re.match(r"(.+?)_Round(\d+).*\.csv", filename, re.IGNORECASE)
    if match_old2: return match_old2.group(1), int(match_old2.group(2))
    return None, None

def load_ngs_data_worker_step1(filepath, is_naive_lib_arg=False):
    filename = os.path.basename(filepath)
    antigen_id_override, round_num_override = None, None
    is_naive_lib = is_naive_lib_arg
    if not is_naive_lib:
        antigen_id_override, round_num_override = parse_filename_info_step1(filename)
        if not antigen_id_override: return None
    df = None
    try:
        try:
            df_comma = pd.read_csv(filepath, sep=',')
            if 'cdrFusion_aa' in df_comma.columns and 'counts' in df_comma.columns: df = df_comma
        except Exception: pass
        if df is None:
            try:
                df_tab = pd.read_csv(filepath, sep='\t')
                if 'cdrFusion_aa' in df_tab.columns and 'counts' in df_tab.columns: df = df_tab
            except Exception: pass
        if df is None: return None
        df = df.rename(columns={'cdrFusion_aa': 'Sequence', 'counts': 'Count'})
        if 'Count' not in df.columns or 'Sequence' not in df.columns: return None
        df['Count'] = pd.to_numeric(df['Count'], errors='coerce'); df = df.dropna(subset=['Count']); df['Count'] = df['Count'].astype(int)
        
        df = df[['Sequence', 'Count']].copy()
        current_antigen_id = 'R0_Naive' if is_naive_lib else antigen_id_override
        current_round_num = 0 if is_naive_lib else round_num_override
        df['AntigenID'] = current_antigen_id; df['Round'] = current_round_num
        
        df_grouped = df.groupby('Sequence').agg(Count=('Count', 'sum'), AntigenID=('AntigenID', 'first'), Round=('Round', 'first')).reset_index()
        total_reads = df_grouped['Count'].sum()
        return ((current_antigen_id, current_round_num), df_grouped, total_reads, filename)
    except Exception: return None

### --- Step II Helpers ---

def plot_top_precluster_clones(enriched_data_s2, output_dir):
    """Plots enrichment profiles for the top 20 pre-clustering clones per antigen."""
    global plt, sns
    if plt is None or sns is None:
        print("\n--- SKIPPING Top Pre-cluster Clone Plotting: matplotlib/seaborn not found. ---")
        return

    print("\n--- Plotting Top 20 Pre-Clustering Enriched Clones ---")
    
    plots_output_dir = os.path.join(output_dir, "step2_top_clone_profiles")
    os.makedirs(plots_output_dir, exist_ok=True)
    
    for antigen, df_enriched in enriched_data_s2.items():
        if df_enriched is None or df_enriched.empty:
            print(f"  - No enriched data for antigen '{antigen}'. Skipping plot.")
            continue

        if 'CPM_R4' in df_enriched.columns:
            rank_col = 'CPM_R4'
            plot_rounds_xaxis = ['R1', 'R2', 'R3', 'R4']
        elif 'CPM_R3' in df_enriched.columns:
            rank_col = 'CPM_R3'
            plot_rounds_xaxis = ['R1', 'R2', 'R3']
        else:
            print(f"  - No R3 or R4 CPM data for antigen '{antigen}'. Skipping plot.")
            continue
        
        top_20_clones = df_enriched.nlargest(20, rank_col)
        
        if top_20_clones.empty:
            print(f"  - No top clones found for antigen '{antigen}'. Skipping plot.")
            continue
            
        print(f"  - Generating enrichment profile plot for top {len(top_20_clones)} clones for {antigen} (ranking by {rank_col})...")
        
        plt.figure(figsize=(12, 8))
        for _, row in top_20_clones.iterrows():
            
            cpm_values = []
            for r_label in plot_rounds_xaxis:
                cpm_values.append(row.get(f'CPM_{r_label}', np.nan))
            
            label = f"{row['Sequence'][:15]}..."
            plt.plot(plot_rounds_xaxis, cpm_values, marker='o', linestyle='-', label=label)
            
        plt.title(f'Top {len(top_20_clones)} Pre-Clustering Clone Profiles - Antigen: {antigen} (Ranked by {rank_col})')
        plt.xlabel('Round')
        plt.ylabel('CPM (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Sequence Start")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen))
        plot_path = os.path.join(plots_output_dir, f"plot_top20_precluster_profiles_{safe_name}.png")
        plt.savefig(plot_path)
        plt.close()

    print("  - Finished generating pre-clustering clone plots.")


def apply_cpm_filter_step2(df_with_cpm, cutoff):
    return df_with_cpm[df_with_cpm['CPM'] >= cutoff].copy()


def perform_enrichment_filtering_for_antigen_generic(df_r1_p, df_r2_p, df_r3_p, df_r4_p, t_r1_o, t_r2_o, t_r3_o, t_r4_o, p_val_thresh, id_col_name='Sequence'):
    """Filters sequences/clusters for monotonic enrichment across rounds using Fisher's exact test."""
    cols_needed = [id_col_name, 'Count']
    df_r1 = df_r1_p if df_r1_p is not None else pd.DataFrame(columns=cols_needed)
    df_r2 = df_r2_p if df_r2_p is not None else pd.DataFrame(columns=cols_needed)
    df_r3 = df_r3_p if df_r3_p is not None else pd.DataFrame(columns=cols_needed)
    df_r4 = df_r4_p if df_r4_p is not None else pd.DataFrame(columns=cols_needed)
    
    for df in [df_r1, df_r2, df_r3, df_r4]:
        for col in cols_needed:
            if col not in df.columns:
                df[col] = 0 if col == 'Count' else pd.NA

    df_m = pd.merge(df_r1[cols_needed], df_r2[cols_needed], on=id_col_name, how='outer', suffixes=('_R1', '_R2'))
    df_r3_to_merge = df_r3[cols_needed].rename(columns={'Count': 'Count_R3'})
    df_m = pd.merge(df_m, df_r3_to_merge, on=id_col_name, how='outer')
    
    if t_r4_o > 0:
        df_r4_to_merge = df_r4[cols_needed].rename(columns={'Count': 'Count_R4'})
        df_m = pd.merge(df_m, df_r4_to_merge, on=id_col_name, how='outer')

    df_m = df_m.fillna(0)
    count_cols = ['Count_R1', 'Count_R2', 'Count_R3']
    if t_r4_o > 0:
        count_cols.append('Count_R4')
    for col in count_cols: 
        df_m[col] = df_m[col].astype(int)

    df_m['CPM_R1'] = df_m['Count_R1'] / (t_r1_o + 1e-9) * 1e6
    df_m['CPM_R2'] = df_m['Count_R2'] / (t_r2_o + 1e-9) * 1e6
    df_m['CPM_R3'] = df_m['Count_R3'] / (t_r3_o + 1e-9) * 1e6
    if t_r4_o > 0:
        df_m['CPM_R4'] = df_m['Count_R4'] / (t_r4_o + 1e-9) * 1e6

    ok23 = df_m['CPM_R3'] >= df_m['CPM_R2']
    ok12 = df_m['CPM_R2'] >= df_m['CPM_R1']
    
    borderline12 = ~ok12
    borderline23 = ~ok23
    
    # Borderline cases: use Fisher's exact test to check for significant decrease
    if t_r1_o > 0 and t_r2_o > 0:
        for idx in df_m.index[borderline12]:
            c1, c2 = df_m.loc[idx, 'Count_R1'], df_m.loc[idx, 'Count_R2']
            tbl = [[c1, int(t_r1_o - c1)], [c2, int(t_r2_o - c2)]]
            tbl[0][1], tbl[1][1] = max(0, tbl[0][1]), max(0, tbl[1][1])
            try:
                _, pval = fisher_exact(tbl, alternative='greater')
                if pval > p_val_thresh: ok12.loc[idx] = True
            except ValueError:
                ok12.loc[idx] = True

    if t_r2_o > 0 and t_r3_o > 0:
        for idx in df_m.index[borderline23]:
            c2, c3 = df_m.loc[idx, 'Count_R2'], df_m.loc[idx, 'Count_R3']
            tbl = [[c2, int(t_r2_o-c2)], [c3, int(t_r3_o-c3)]]
            tbl[0][1],tbl[1][1] = max(0,tbl[0][1]),max(0,tbl[1][1])
            try:
                _, pval = fisher_exact(tbl, alternative='greater')
                if pval > p_val_thresh: ok23.loc[idx] = True
            except ValueError:
                ok23.loc[idx] = True
    
    if t_r4_o > 0:
        ok34 = df_m['CPM_R4'] >= df_m['CPM_R3']
        borderline34 = ~ok34
        if t_r3_o > 0:
            for idx in df_m.index[borderline34]:
                c3, c4 = df_m.loc[idx, 'Count_R3'], df_m.loc[idx, 'Count_R4']
                tbl = [[c3, int(t_r3_o-c3)], [c4, int(t_r4_o-c4)]]
                tbl[0][1],tbl[1][1] = max(0,tbl[0][1]),max(0,tbl[1][1])
                try:
                    _, pval = fisher_exact(tbl, alternative='greater')
                    if pval > p_val_thresh: ok34.loc[idx] = True
                except ValueError:
                    ok34.loc[idx] = True
        
        enriched_df = df_m[ok12 & ok23 & ok34].copy()
    else:
        enriched_df = df_m[ok12 & ok23].copy()
    
    final_cols = {id_col_name: id_col_name,
                  'Count_R1': 'Count_R1', 'CPM_R1': 'CPM_R1', 
                  'Count_R2': 'Count_R2', 'CPM_R2': 'CPM_R2', 
                  'Count_R3': 'Count_R3', 'CPM_R3': 'CPM_R3'}
    
    if t_r4_o > 0:
        final_cols.update({'Count_R4': 'Count_R4', 'CPM_R4': 'CPM_R4'})
    
    final_df_cols = [col for col in final_cols.keys() if col in enriched_df.columns]
    return enriched_df[final_df_cols].rename(columns=final_cols)

def worker_enrichment_filter_step2(args):
    antigen_id, df_r1_p, df_r2_p, df_r3_p, df_r4_p, t_r1_o, t_r2_o, t_r3_o, t_r4_o, p_val_thresh = args
    return antigen_id, perform_enrichment_filtering_for_antigen_generic(df_r1_p, df_r2_p, df_r3_p, df_r4_p, t_r1_o, t_r2_o, t_r3_o, t_r4_o, p_val_thresh, id_col_name='Sequence')

### --- Step III Helpers ---
def write_fasta_file_step3(sequences, fasta_path):
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{i}\n{seq}\n")
    print(f"  Wrote {len(sequences)} sequences to {fasta_path}", flush=True)

def load_id_to_sequence_map_from_fasta_generic(fasta_path):
    id_to_seq = {}
    if not os.path.exists(fasta_path):
        print(f"CRITICAL ERROR: FASTA file for ID mapping not found: {fasta_path}")
        return None
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                current_id = line[1:].strip()
                id_to_seq[current_id] = next(f).strip()
    print(f"  Loaded {len(id_to_seq)} sequence IDs from {os.path.basename(fasta_path)}.")
    return id_to_seq

### --- Step IV Helpers ---

def aggregate_cluster_counts_step4(cluster_df, id_to_seq_map, all_data_s1):
    """Aggregates original read counts for all cluster members via pandas merge."""
    print("  Aggregating original counts for cluster members...")

    id_map_df = pd.DataFrame(list(id_to_seq_map.items()), columns=['member_id', 'Sequence'])

    rep_map_df = pd.DataFrame(list(id_to_seq_map.items()), columns=['representative_id', 'RepresentativeSequence'])

    merged_map = pd.merge(cluster_df, id_map_df, on='member_id', how='left')
    full_map = pd.merge(merged_map, rep_map_df, on='representative_id', how='left')
    seq_to_rep_seq_map = full_map[['Sequence', 'RepresentativeSequence']].dropna().drop_duplicates()

    all_counts_list = []
    for (antigen, round_num), df in all_data_s1.items():
        if antigen != 'R0_Naive' and df is not None:
            all_counts_list.append(df[['Sequence', 'Count', 'AntigenID', 'Round']])
    all_counts_df = pd.concat(all_counts_list)

    merged_counts = pd.merge(all_counts_df, seq_to_rep_seq_map, on='Sequence', how='inner')
    cluster_agg_data = merged_counts.groupby(['RepresentativeSequence', 'AntigenID', 'Round'])['Count'].sum().reset_index()

    if cluster_agg_data.empty:
        print("  Warning: No cluster counts were aggregated.")
        
    return cluster_agg_data


def worker_cluster_enrichment_filter_step4(args):
    """Processes all clusters for a single antigen, filtering for enrichment."""
    antigen_id, df_agg_for_antigen, total_reads_s1, p_val_thresh = args
    
    print(f"  - Starting cluster enrichment for {antigen_id}...")
    
    # Pivot the data to get rounds as columns for easy comparison
    pivot_df = df_agg_for_antigen.pivot_table(
        index='RepresentativeSequence', 
        columns='Round', 
        values='Count', 
        fill_value=0
    )

    for round_num in [1, 2, 3, 4]:
        if round_num not in pivot_df.columns:
            pivot_df[round_num] = 0
            
    pivot_df = pivot_df.rename(columns={1: 'Count_R1', 2: 'Count_R2', 3: 'Count_R3', 4: 'Count_R4'})

    t_r1_o = total_reads_s1.get((antigen_id, 1), 0)
    t_r2_o = total_reads_s1.get((antigen_id, 2), 0)
    t_r3_o = total_reads_s1.get((antigen_id, 3), 0)
    t_r4_o = total_reads_s1.get((antigen_id, 4), 0)

    pivot_df['CPM_R1'] = (pivot_df['Count_R1'] / t_r1_o) * 1e6 if t_r1_o > 0 else 0
    pivot_df['CPM_R2'] = (pivot_df['Count_R2'] / t_r2_o) * 1e6 if t_r2_o > 0 else 0
    pivot_df['CPM_R3'] = (pivot_df['Count_R3'] / t_r3_o) * 1e6 if t_r3_o > 0 else 0
    
    pivot_df['CPM_R4'] = (pivot_df['Count_R4'] / t_r4_o) * 1e6 if t_r4_o > 0 else 0
    ok23 = pivot_df['CPM_R3'] >= pivot_df['CPM_R2']
    ok12 = pivot_df['CPM_R2'] >= pivot_df['CPM_R1']
    ok34 = pivot_df['CPM_R4'] >= pivot_df['CPM_R3']
    
    if t_r4_o > 0:
        enriched_df = pivot_df[ok12 & ok23 & ok34].reset_index()
    else:
        enriched_df = pivot_df[ok12 & ok23].reset_index()
    
    cols_to_return = ['RepresentativeSequence', 'Count_R1', 'CPM_R1', 'Count_R2', 'CPM_R2', 'Count_R3', 'CPM_R3']
    if t_r4_o > 0:
        cols_to_return.extend(['Count_R4', 'CPM_R4'])
        
    final_cols = [col for col in cols_to_return if col in enriched_df.columns]
    return antigen_id, enriched_df[final_cols]


# --- Step V Helpers ---

def worker_process_antigen_specificity(args):
    target_antigen, df_target_enriched, df_agg_cluster_counts, all_other_antigens, total_reads_s1, q_value_cutoff, fc_cutoff, target_analysis_round = args
    
    print(f"  - Starting specificity analysis for {len(df_target_enriched)} clusters of target: {target_antigen} (using R{target_analysis_round})")
    
    results = []
    total_reads_target = total_reads_s1.get((target_antigen, target_analysis_round), 0)
    if total_reads_target == 0: return pd.DataFrame()

    target_reps = set(df_target_enriched['RepresentativeSequence'])
    subset_df = df_agg_cluster_counts[df_agg_cluster_counts['RepresentativeSequence'].isin(target_reps)]
    counts_map = subset_df.set_index(['RepresentativeSequence', 'AntigenID', 'Round'])['Count'].to_dict()

    for rep_sequence in target_reps:
        count_target = counts_map.get((rep_sequence, target_antigen, target_analysis_round), 0)
        if count_target == 0: continue
        
        prop_target = count_target / total_reads_target
        p_values = []
        comparisons_data = []

        for other_antigen in all_other_antigens:
            other_round_to_check = target_analysis_round
            total_reads_other = total_reads_s1.get((other_antigen, other_round_to_check), 0)
            # Fall back to R3 if the comparison antigen lacks R4 data
            if total_reads_other == 0 and target_analysis_round == 4:
                other_round_to_check = 3
                total_reads_other = total_reads_s1.get((other_antigen, other_round_to_check), 0)
                
            count_other = counts_map.get((rep_sequence, other_antigen, other_round_to_check), 0)
            
            prop_other, fold_change, p_value = 0.0, np.inf, 1.0
            if total_reads_other > 0:
                prop_other = count_other / total_reads_other
                fold_change = prop_target / (prop_other + 1e-9)
                table = [[count_target, int(max(0, total_reads_target - count_target))], 
                         [count_other, int(max(0, total_reads_other - count_other))]]
                try:
                    _, p_value = fisher_exact(table, alternative='greater')
                except ValueError:
                    pass
            
            p_values.append(p_value)
            comparisons_data.append({
                'ComparisonAntigen': other_antigen, 
                'RoundChecked': other_round_to_check, 
                'P_Value': p_value, 
                'FoldChange': fold_change, 
                'Prop_Other': prop_other
            })

        is_specific = True
        if p_values:
            try:
                reject, q_values, _, _ = multipletests(p_values, alpha=q_value_cutoff, method='fdr_bh')
                for i, comp_data in enumerate(comparisons_data):
                    comp_data['Q_Value'] = float(q_values[i])
                    is_cross_reactive = not (q_values[i] < q_value_cutoff and comp_data['FoldChange'] > fc_cutoff)
                    if is_cross_reactive and comp_data.get('Prop_Other', 0) > 1e-9:
                        is_specific = False
            except Exception:
                is_specific = False
        
        results.append({
            'RepresentativeSequence': rep_sequence,
            'TargetAntigen': target_antigen,
            'IsSpecific': is_specific,
            'SpecificityDetails': str(comparisons_data)
        })
        
    return pd.DataFrame(results)


### --- Step VI Helpers ---

def calculate_kl_divergence(cluster_alignment, background_dist):
    """
    Calculates KL Divergence using a minimal epsilon to avoid zero-frequency
    errors, preserving the original background distribution.
    """
    if not cluster_alignment or not background_dist: return np.nan
    
    aln_len = cluster_alignment.get_alignment_length()
    if aln_len != len(background_dist): return np.nan

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    total_kl_div = 0.0

    for i in range(aln_len):
        # Calculate cluster frequencies (P) for this position
        column_aas = [seq[i] for seq in cluster_alignment]
        counts = Counter(c for c in column_aas if c in amino_acids)
        total_in_col = sum(counts.values())
        if total_in_col == 0: continue
        
        p_dist = {aa: count / total_in_col for aa, count in counts.items()}
        q_dist = background_dist[i]
        
        position_kl_div = 0.0
        for aa, p_freq in p_dist.items():
            if p_freq == 0: continue
            
            q_freq = q_dist.get(aa, 0.0)
            if q_freq == 0.0:
                q_freq = 1e-9 # Use a tiny epsilon to prevent division by zero

            position_kl_div += p_freq * np.log2(p_freq / q_freq)
        
        total_kl_div += position_kl_div
        
    return total_kl_div

def run_mafft_alignment_step6(sequences, mafft_exe, num_threads=1):
    """Runs MAFFT multiple sequence alignment via subprocess, returns AlignIO object."""
    global AlignIO
    if AlignIO is None: return None
    if not sequences or len(sequences) < 2: return None
    if not shutil.which(mafft_exe):
        print(f"  MAFFT executable '{mafft_exe}' not found. Skipping MSA.")
        return None

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as tmp_fasta:
        valid_seqs = 0
        for i, seq_str in enumerate(sequences):
            if isinstance(seq_str, str) and seq_str.strip():
                tmp_fasta.write(f">s{i}\n{seq_str}\n")
                valid_seqs += 1
        tmp_fasta_path = tmp_fasta.name

    if valid_seqs < 2:
        os.remove(tmp_fasta_path)
        return None

    alignment = None
    try:
        cmd = [
            mafft_exe,
            "--auto",
            "--quiet",
            "--thread", str(num_threads),
            tmp_fasta_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            alignment = AlignIO.read(io.StringIO(result.stdout), "fasta")
            
    except subprocess.CalledProcessError as e:
        print(f"  Error during MAFFT execution: {e.stderr}")
    except Exception as e:
        print(f"  An unexpected error occurred during MAFFT alignment: {e}")
    finally:
        if os.path.exists(tmp_fasta_path):
            os.remove(tmp_fasta_path)
            
    return alignment
    
def calculate_pssm_and_entropy_step6(alignment, background_freq=None):
    if not alignment or len(alignment) == 0 or alignment.get_alignment_length() == 0: return None, None, None
    num_seqs, aln_len = len(alignment), alignment.get_alignment_length(); amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    pssm_counts = pd.DataFrame(0, index=list(amino_acids), columns=range(aln_len))
    for col_idx in range(aln_len):
        for aa_char in alignment[:, col_idx]:
            if aa_char in amino_acids: pssm_counts.loc[aa_char, col_idx] += 1
    effective_counts_per_pos = pssm_counts.sum(axis=0)
    pssm_freq = pd.DataFrame(0.0, index=list(amino_acids), columns=range(aln_len))
    for col_idx in range(aln_len):
        if effective_counts_per_pos[col_idx] > 0: pssm_freq.iloc[:, col_idx] = pssm_counts.iloc[:, col_idx] / effective_counts_per_pos[col_idx]
    shannon_entropy = [-sum(p*math.log2(p) for p in pssm_freq[col_idx] if p>0) for col_idx in pssm_freq.columns]
    avg_shannon_entropy = np.mean(shannon_entropy) if shannon_entropy else 0.0; pssm_log_odds = None
    if background_freq:
        pssm_log_odds = pd.DataFrame(0.0, index=list(amino_acids), columns=range(aln_len))
        for col_idx in range(aln_len):
            for aa in amino_acids:
                p_i, b_i = pssm_freq.loc[aa, col_idx], background_freq.get(aa, 1e-9)
                if p_i > 0 and b_i > 0: pssm_log_odds.loc[aa, col_idx] = math.log2(p_i / b_i)
    return pssm_freq, pssm_log_odds, avg_shannon_entropy

def calculate_r0_positional_background(all_data_s1):
    """Calculates position-specific amino acid frequencies from R0, grouped by sequence length."""
    print("  - Calculating length- and position-specific R0 background frequencies...")
    r0_df = all_data_s1.get(('R0_Naive', 0))
    if r0_df is None or r0_df.empty:
        print("    WARNING: R0 data not found. KL Divergence calculation will be skipped.")
        return {}

    # Group sequences by length
    r0_df['Length'] = r0_df['Sequence'].str.len()
    seqs_by_len = {length: group['Sequence'].tolist() for length, group in r0_df.groupby('Length')}
    
    backgrounds = {}
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    for length, sequences in seqs_by_len.items():
        if length == 0 or len(sequences) < 100:
            continue
            
        positional_freqs = []
        for i in range(length):
            column_aas = [seq[i] for seq in sequences if len(seq) > i]
            total_in_col = len(column_aas)
            if total_in_col == 0:
                positional_freqs.append({aa: 1e-9 for aa in amino_acids})
                continue
            
            counts = Counter(column_aas)
            freq_dict = {aa: counts.get(aa, 0) / total_in_col for aa in amino_acids}
            positional_freqs.append(freq_dict)
        
        backgrounds[length] = positional_freqs
        
    print(f"    - Done. Calculated positional backgrounds for {len(backgrounds)} different sequence lengths.")
    return backgrounds

def calculate_r0_background_frequencies_step6(all_data_s1_with_cpms, default_bg):
    """
    Calculates length-specific background amino acid frequencies from the R0 naive library.
    
    Returns:
        dict: A dictionary where keys are CDR3 lengths and values are the corresponding
              background frequency dictionaries for that length.
              e.g., {8: {'A': 0.08, ...}, 9: {'A': 0.075, ...}}
    """
    print("  Calculating length-specific R0 background frequencies...")
    r0_df = all_data_s1_with_cpms.get(('R0_Naive', 0))
    if r0_df is None or r0_df.empty:
        print("  WARNING: R0 data for background not found, using default for all lengths.")
        return {}

    # Use a dictionary to hold a Counter for each CDR3 length
    counts_by_len = defaultdict(Counter)
    
    # Iterate through sequences to get counts per length
    for seq in r0_df['Sequence']:
        if not isinstance(seq, str):
            continue
        
        # We only need the CDR3 length for this calculation
        _, _, _, cdr3_len = parse_cdrs(seq)
        
        if cdr3_len > 0:
            # Update the counter for this specific CDR3 length
            counts_by_len[cdr3_len].update(seq)
            
    if not counts_by_len:
        print("  WARNING: No valid sequences found in R0 to calculate backgrounds.")
        return {}

    # Calculate frequencies for each length
    backgrounds_by_len = {}
    for length, counts in counts_by_len.items():
        total_aas = sum(c for aa, c in counts.items() if aa.isalpha() and aa.isupper())
        if total_aas > 0:
            backgrounds_by_len[length] = {
                aa: count / total_aas for aa, count in counts.items() if aa.isalpha() and aa.isupper()
            }
            
    print(f"  - Done. Calculated backgrounds for {len(backgrounds_by_len)} different CDR3 lengths.")
    return backgrounds_by_len

def generate_antigen_plots_step6(antigen_name, antigen_report_df, plots_output_dir):
    """Generates cluster size histogram and enrichment profile plots for an antigen."""
    global plt, sns, TOP_N_CLUSTERS_TO_PLOT
    if plt is None or sns is None: print(f"Plotting libraries not available for Step VI plots for {antigen_name}."); return
    if antigen_report_df.empty: print(f"  No candidates for {antigen_name} to plot."); return
    
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen_name))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(antigen_report_df['ClusterSize'], bins=max(1, min(20, antigen_report_df['ClusterSize'].nunique())), kde=False)
    plt.title(f'Cluster Sizes - Antigen: {antigen_name}')
    plt.xlabel('Cluster Size')
    plt.ylabel('# Clusters (Log Scale)')
    plt.yscale('log') # Set Y-axis to log scale
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_dir, f"plot_step6_clustersize_{safe_name}.png"))
    plt.close()
    print(f"    Saved Step VI cluster size plot for {antigen_name}")

    top_n = min(TOP_N_CLUSTERS_TO_PLOT, len(antigen_report_df))
    if top_n > 0:
        top_df = antigen_report_df.head(top_n)
        plt.figure(figsize=(12, 8))
        for _, r in top_df.iterrows():
            plt.plot(['R1','R2','R3'], [r.get('R1_CPM',np.nan), r.get('R2_CPM',np.nan), r.get('R3_CPM',np.nan)], marker='o', label=f"{r['RepSequence'][:12]}.. (S:{r['ClusterSize']})")
        plt.title(f'Top {top_n} Enriched Specific Cluster Profiles - Antigen: {antigen_name}')
        plt.xlabel('Round')
        plt.ylabel('CPM (Log Scale)')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', title="RepSeq (Size)")
        plt.grid(True,which="both",ls="-",alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.savefig(os.path.join(plots_output_dir, f"plot_step6_enrich_profiles_{safe_name}.png"))
        plt.close()
        print(f"    Saved Step VI enrichment profiles plot for top {top_n} clusters for {antigen_name}")

### --- Step VII Helpers ---




# ==============================================================================
# === PIPELINE STEPS (I-VI) ===
# ==============================================================================

def main_step1():
    print(f"--- Step I: Data Preparation & CPM Calculation (Using {NUM_PROCESSES} processes) ---", flush=True)
    all_data, total_reads_per_sample = {}, {}
    r0_result = load_ngs_data_worker_step1(NAIVE_LIBRARY_R0_FILE, is_naive_lib_arg=True)
    if r0_result: key, df_r0, total_r0_reads, _ = r0_result; all_data[key], total_reads_per_sample[key] = df_r0, total_r0_reads; print(f"Loaded R0: {len(df_r0)} seqs, {total_r0_reads} reads.", flush=True)
    else: print(f"CRITICAL ERROR: Failed R0 load: {NAIVE_LIBRARY_R0_FILE}."); return None, None
    if not os.path.isdir(RANKED_SEQS_DIR): print(f"CRITICAL ERROR: Dir not found: {RANKED_SEQS_DIR}."); return None, None
    ranked_files_paths = glob.glob(os.path.join(RANKED_SEQS_DIR, "*.csv"))
    if ranked_files_paths:
        print(f"Found {len(ranked_files_paths)} ranked files.", flush=True)
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool: results = pool.map(load_ngs_data_worker_step1, ranked_files_paths)
        processed_count = sum(1 for r in results if r and r[0][0] is not None and (all_data.update({r[0]: r[1]}) or True) and (total_reads_per_sample.update({r[0]: r[2]}) or True))
        print(f"Processed {processed_count} ranked files.", flush=True)
    else: print(f"Warning: No CSVs in {RANKED_SEQS_DIR}.")

    print("  Adding CPM columns to Step I data...", flush=True)
    for key, df_sample in all_data.items():
        total_reads = total_reads_per_sample.get(key)
        if df_sample is not None and total_reads and total_reads > 0: df_sample['CPM'] = (df_sample['Count'] / total_reads) * 1_000_000
        elif df_sample is not None: df_sample['CPM'] = 0.0

    step1_output_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step1_all_loaded_data_with_cpms.pkl")
    with open(step1_output_path, 'wb') as f: pickle.dump(all_data, f)
    print(f"Saved Step I all_data (with CPMs) to: {step1_output_path}", flush=True)
    step1_total_reads_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step1_total_reads_per_sample.pkl")
    with open(step1_total_reads_path, 'wb') as f: pickle.dump(total_reads_per_sample, f)
    print(f"Saved Step I total_reads_per_sample to: {step1_total_reads_path}", flush=True)
    print("Step I completed.", flush=True)
    return all_data, total_reads_per_sample

def main_step2(all_data_s1, total_reads_s1):
    """Filters sequences: R3 CPM gate, then per-antigen enrichment via Fisher's exact test."""
    print(f"\n--- Step II: Sequence Filtering (Using {NUM_PROCESSES} processes) ---", flush=True)
    
    all_samples_list = [df for (ag, rn), df in all_data_s1.items() if ag != 'R0_Naive' and df is not None]
    if not all_samples_list:
        print("CRITICAL: No sample data (R1-R3) to filter."); return {}
    master_df = pd.concat(all_samples_list, ignore_index=True)

    print(f"  - Applying R3 CPM >= {CPM_CUTOFF_R3} AND Count >= {MIN_R3_READS} as the primary gatekeeper...")
    r3_df = master_df[master_df['Round'] == 3]
    passing_mask = (r3_df['CPM'] >= CPM_CUTOFF_R3) & (r3_df['Count'] >= MIN_R3_READS)
    passing_r3_seqs = set(r3_df[passing_mask]['Sequence'])
    print(f"  - {len(passing_r3_seqs)} unique sequences passed the R3 gate.")
    
    df_after_r3_gate = master_df[master_df['Sequence'].isin(passing_r3_seqs)]

    data_for_enrichment_input = df_after_r3_gate

    enriched_final = {}
    antigen_ids = sorted(list(set(k[0] for k in all_data_s1 if k[0] != 'R0_Naive')))
    enrich_args = []
    
    for ag_id in antigen_ids:
        antigen_df = data_for_enrichment_input[data_for_enrichment_input['AntigenID'] == ag_id]
        
        df_r1 = antigen_df[antigen_df['Round'] == 1][['Sequence', 'Count']]
        df_r2 = antigen_df[antigen_df['Round'] == 2][['Sequence', 'Count']]
        df_r3 = antigen_df[antigen_df['Round'] == 3][['Sequence', 'Count']]
        df_r4 = antigen_df[antigen_df['Round'] == 4][['Sequence', 'Count']]
        
        totals_123 = [total_reads_s1.get((ag_id, r), 0) for r in [1, 2, 3]]
        total_r4 = total_reads_s1.get((ag_id, 4), 0)

        if any(t == 0 for t in totals_123):
            print(f"Skipping {ag_id} for enrichment: zero total reads in R1/R2/R3.")
            enriched_final[ag_id] = pd.DataFrame()
            continue

        totals_all = totals_123 + [total_r4]
            
        enrich_args.append((ag_id, df_r1, df_r2, df_r3, df_r4, *totals_all, FISHER_P_VALUE_THRESHOLD_FOR_DECREASE))
    
    if enrich_args:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            results_enr = pool.map(worker_enrichment_filter_step2, enrich_args)
        for ag_id, df_e in results_enr:
            enriched_final[ag_id] = df_e
            print(f"  {ag_id}: {len(df_e)} sequences passed final enrichment.", flush=True)
    else:
        print("No antigens eligible for sequence enrichment processing.")

    step2_output_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step2_enriched_sequences.pkl")
    with open(step2_output_path, 'wb') as f: pickle.dump(enriched_final, f)
    
    enriched_df_list = [df.assign(EnrichedForAntigen=ag) for ag, df in enriched_final.items() if df is not None and not df.empty]
    if enriched_df_list:
        pd.concat(enriched_df_list).to_csv(os.path.join(CURRENT_RUN_OUTPUT_DIR, "step2_enriched_sequences_report.csv"), index=False)
    
    print("Step II completed.", flush=True)
    return enriched_final

def main_step2_5_pre_clustering_specificity(enriched_data_s2, all_data_s1):
    """Removes polyreactive sequences (present above CPM threshold for multiple antigens in R3)."""
    print("\n--- Step 2.5: Pre-Clustering Cross-Reactivity Analysis ---", flush=True)
    if CROSS_REACTIVITY_EXCLUSION_GROUPS:
        print(f"  Using exclusion groups: {CROSS_REACTIVITY_EXCLUSION_GROUPS}")

    enriched_seqs = set(s for df in enriched_data_s2.values() if df is not None and not df.empty for s in df['Sequence'])
    if not enriched_seqs:
        print("CRITICAL: No enriched sequences from Step II to analyze."); return None

    r3_dfs = [df[['Sequence', 'AntigenID', 'CPM']] for (ag, rn), df in all_data_s1.items() if rn == 3 and df is not None and 'CPM' in df.columns]
    if not r3_dfs:
        print("  WARNING: No Round 3 data found. Passing all enriched sequences to clustering."); return enriched_seqs

    all_r3_data = pd.concat(r3_dfs, ignore_index=True)
    significant_r3_hits = all_r3_data[all_r3_data['CPM'] >= CPM_CUTOFF_R3]

    def get_normalized_antigen_count(group):
        antigens = set(group)
        for ex_group in CROSS_REACTIVITY_EXCLUSION_GROUPS:
            intersection = antigens.intersection(ex_group)
            if len(intersection) > 1:
                antigens = (antigens - ex_group) | {sorted(list(ex_group))[0]}
        return len(antigens)

    seq_antigen_counts = significant_r3_hits.groupby('Sequence')['AntigenID'].apply(get_normalized_antigen_count)
    
    cross_reactive_seq_ids = set(seq_antigen_counts[seq_antigen_counts > 1].index)
    
    if cross_reactive_seq_ids:
        filtered_details = significant_r3_hits[significant_r3_hits['Sequence'].isin(cross_reactive_seq_ids)]
        reason_map = filtered_details.groupby('Sequence')['AntigenID'].apply(lambda x: 'Found_In_R3_Of_' + '&'.join(x))
        del all_r3_data, significant_r3_hits
        all_samples_df = pd.concat([df for (ag, rn), df in all_data_s1.items() if ag != 'R0_Naive'], ignore_index=True)
        filtered_out_df = all_samples_df[all_samples_df['Sequence'].isin(cross_reactive_seq_ids)].copy()
        del all_samples_df
        filtered_out_df['Reason_for_Filtering'] = filtered_out_df['Sequence'].map(reason_map)
        filtered_out_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step2.5_filtered_crossreactive_sequences.csv")
        filtered_out_df.to_csv(filtered_out_path, index=False)
        print(f"  Identified {len(cross_reactive_seq_ids)} cross-reactive sequences. Saved details to: {os.path.basename(filtered_out_path)}")

    specific_sequences_for_clustering = enriched_seqs - cross_reactive_seq_ids
    print(f"  After filtering, {len(specific_sequences_for_clustering)} specific sequences will be passed to clustering.")

    output_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step2_5_specific_sequences_for_clustering.pkl")
    with open(output_path, 'wb') as f: pickle.dump(specific_sequences_for_clustering, f)
    
    print("Step 2.5 completed.", flush=True)
    return specific_sequences_for_clustering

def main_step3(specific_sequences_for_clustering, mmseqs_threads):
    """Clusters specific sequences using MMseqs2."""
    global COVERAGE
    mmseqs_output_dir_s3 = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step3_mmseqs_clustering")
    print(f"\n--- Step III: Clustering (MMseqs2 using {mmseqs_threads} threads, COVERAGE={COVERAGE}) ---", flush=True)
    
    if not specific_sequences_for_clustering:
        print("CRITICAL ERROR: No specific sequences from Step 2.5 were provided for clustering."); return None, None

    unique_seqs = sorted(list(specific_sequences_for_clustering))
    print(f"Received {len(unique_seqs)} unique, specific sequences for clustering.", flush=True)
    
    os.makedirs(mmseqs_output_dir_s3, exist_ok=True)
    
    fasta_path = os.path.join(mmseqs_output_dir_s3, "specific_sequences_for_clustering.fasta")
    db_path = os.path.join(mmseqs_output_dir_s3, "targetDB"); clusterdb_path = os.path.join(mmseqs_output_dir_s3, "clusterResultDB")
    tmp_dir = os.path.join(mmseqs_output_dir_s3, "mmseqs_tmp_pipeline"); os.makedirs(tmp_dir, exist_ok=True)
    tsv_path = os.path.join(mmseqs_output_dir_s3, "clusters.tsv")
    
    write_fasta_file_step3(unique_seqs, fasta_path)
    if not shutil.which(MMSEQS_EXECUTABLE): print(f"CRITICAL ERROR: MMseqs2 not found: {MMSEQS_EXECUTABLE}."); shutil.rmtree(tmp_dir,ignore_errors=True); return None, fasta_path
    
    try:
        cmds = [[MMSEQS_EXECUTABLE,"createdb",fasta_path,db_path], [MMSEQS_EXECUTABLE,"cluster",db_path,clusterdb_path,tmp_dir,"--min-seq-id",str(MIN_SEQ_ID),"-c",str(COVERAGE),"--cov-mode",str(COV_MODE),"--cluster-mode",str(CLUSTER_MODE),"--threads",str(mmseqs_threads)], [MMSEQS_EXECUTABLE,"createtsv",db_path,db_path,clusterdb_path,tsv_path,"--threads",str(mmseqs_threads)]]
        for i, cmd in enumerate(cmds): print(f"  Executing MMseqs2 cmd {i+1}: {' '.join(cmd)}", flush=True); subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  MMseqs2 clustering completed. TSV: {tsv_path}", flush=True)
    except subprocess.CalledProcessError as e: print(f"CRITICAL ERROR MMseqs2: {' '.join(e.cmd)}\n{e.stderr}"); return None, fasta_path
    finally: shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("Step III completed. Files saved in:", mmseqs_output_dir_s3, flush=True);
    return tsv_path, fasta_path

def main_step3_5_initial_cluster_analysis(cluster_tsv_path, fasta_path, all_data_s1):
    """Generates a diagnostic report of cluster composition: member sources, sizes, antigens."""
    print("\n--- Step 3.5: Initial Cluster Composition Analysis ---", flush=True)
    try:
        cluster_df = pd.read_csv(cluster_tsv_path, sep='\t', header=None, names=['representative_id', 'member_id'], dtype=str)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Cluster TSV not found for analysis: {cluster_tsv_path}"); return
    id_to_seq_map = load_id_to_sequence_map_from_fasta_generic(fasta_path)
    if not id_to_seq_map:
        print("CRITICAL ERROR: Could not load ID-to-sequence map."); return
    cluster_df['RepresentativeSequence'] = cluster_df['representative_id'].map(id_to_seq_map)
    cluster_df['MemberSequence'] = cluster_df['member_id'].map(id_to_seq_map)
    seq_to_rep_map = cluster_df[['MemberSequence', 'RepresentativeSequence']].dropna().rename(columns={'MemberSequence': 'Sequence'})
    all_counts_list = [df for df in all_data_s1.values() if df is not None]
    all_counts_df = pd.concat(all_counts_list, ignore_index=True)
    merged_data = pd.merge(all_counts_df, seq_to_rep_map, on='Sequence', how='inner')
    num_clusters = merged_data['RepresentativeSequence'].nunique()
    print(f"  Aggregating composition for {num_clusters} total clusters.")
    all_reps = merged_data['RepresentativeSequence'].unique()
    all_rep_info_r3 = all_counts_df[(all_counts_df['Sequence'].isin(all_reps)) & (all_counts_df['Round'] == 3)].copy()
    rep_primary_info = all_rep_info_r3.sort_values('Count', ascending=False).drop_duplicates(subset=['Sequence'], keep='first').rename(columns={'Sequence': 'RepresentativeSequence', 'AntigenID': 'Rep_Primary_Antigen', 'Count': 'Rep_Primary_R3_Count'})[['RepresentativeSequence', 'Rep_Primary_Antigen', 'Rep_Primary_R3_Count']]
    merged_data['Member_Source'] = merged_data['AntigenID'].astype(str) + "_R" + merged_data['Round'].astype(str)
    agg_ops = { 'Cluster_Size': ('Sequence', 'nunique'), 'Member_Sequences': ('Sequence', lambda x: ";".join(x.unique())), 'Member_Sources(Antigen_Round)': ('Member_Source', lambda x: ";".join(x)), 'Member_Counts': ('Count', lambda x: ";".join(x.astype(str))) }
    cluster_summary = merged_data.groupby('RepresentativeSequence').agg(**agg_ops).reset_index()
    final_summary = pd.merge(cluster_summary, rep_primary_info, on='RepresentativeSequence', how='left')
    
    final_summary['Rep_Primary_Antigen'] = final_summary['Rep_Primary_Antigen'].fillna("N/A")
    final_summary['Rep_Primary_R3_Count'] = final_summary['Rep_Primary_R3_Count'].fillna(0)
    final_summary = final_summary[['RepresentativeSequence', 'Rep_Primary_Antigen', 'Rep_Primary_R3_Count', 'Cluster_Size', 'Member_Sequences', 'Member_Sources(Antigen_Round)', 'Member_Counts']]
    final_summary = final_summary.sort_values(by='Cluster_Size', ascending=False)
    
    output_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step3.5_initial_cluster_composition.csv")
    final_summary.to_csv(output_path, index=False)
    print(f"  Saved initial cluster composition report to: {output_path}")
    print("Step 3.5 completed.", flush=True)

    return final_summary

def main_step4(cluster_tsv_path, fasta_path_s3, all_data_s1, total_reads_s1):
    """Aggregates cluster-level counts and filters clusters for enrichment (one task per antigen)."""
    print(f"\n--- Step IV: Cluster-wise Enrichment (Using {NUM_PROCESSES} processes) ---", flush=True)
    if not cluster_tsv_path or not fasta_path_s3: print("CRITICAL ERROR: Missing cluster TSV or FASTA from Step III."); return {}, None

    try:
        cluster_df = pd.read_csv(cluster_tsv_path, sep='\t', header=None, names=['representative_id', 'member_id'], dtype=str)
    except FileNotFoundError: print(f"CRITICAL ERROR: Cluster TSV not found: {cluster_tsv_path}"); return {}, None
    
    id_to_seq_map = load_id_to_sequence_map_from_fasta_generic(fasta_path_s3)
    if not id_to_seq_map: return {}, None

    df_agg_cluster_counts = aggregate_cluster_counts_step4(cluster_df, id_to_seq_map, all_data_s1)
    if df_agg_cluster_counts.empty: print("No cluster counts to enrich."); return {}, df_agg_cluster_counts
    
    step4_agg_counts_path_csv = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step4_aggregated_cluster_counts.csv")
    df_agg_cluster_counts.to_csv(step4_agg_counts_path_csv, index=False)
    print(f"  Saved aggregated cluster counts to: {os.path.basename(step4_agg_counts_path_csv)}")
    
    enriched_clusters_final = {}
    antigen_ids_clustering = sorted(list(df_agg_cluster_counts['AntigenID'].unique()))
    
    worker_args = []
    for ag_id in antigen_ids_clustering:
        df_agg_for_antigen = df_agg_cluster_counts[df_agg_cluster_counts['AntigenID'] == ag_id]
        worker_args.append((ag_id, df_agg_for_antigen, total_reads_s1, FISHER_P_VALUE_THRESHOLD_FOR_DECREASE))
    
    if worker_args:
        print(f"Launching {len(worker_args)} large-grained analysis tasks for all antigens...")
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            results_cl_enr = pool.map(worker_cluster_enrichment_filter_step4, worker_args)
        
        for ag_id, df_e_cl in results_cl_enr:
            enriched_clusters_final[ag_id] = df_e_cl
            print(f"    {ag_id}: {len(df_e_cl)} clusters passed enrichment.", flush=True)
    else:
        print("No antigens eligible for cluster enrichment processing.")

    step4_enriched_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step4_enriched_clusters.pkl")
    step4_agg_counts_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step4_aggregated_cluster_counts.pkl")
    with open(step4_enriched_path, 'wb') as f: pickle.dump(enriched_clusters_final, f)
    with open(step4_agg_counts_path, 'wb') as f: pickle.dump(df_agg_cluster_counts, f)

    print("Step IV completed.", flush=True)
    return enriched_clusters_final, df_agg_cluster_counts

def main_step5(enriched_clusters_s4, df_agg_cluster_counts_s4, total_reads_s1,
               cluster_tsv_path_s3=None, fasta_path_s3=None):
    print(f"\n--- Step V: Post-Clustering Cross-Reactivity Analysis (Using {NUM_PROCESSES} processes) ---", flush=True)
    if not enriched_clusters_s4 or not any(df is not None and not df.empty for df in enriched_clusters_s4.values()):
        print("No enriched clusters from Step IV for cross-reactivity."); return pd.DataFrame()

    all_antigens_list = sorted(list(df_agg_cluster_counts_s4['AntigenID'].unique()))
    worker_args = []
    
    for target_antigen, df_target_enriched_clusters in enriched_clusters_s4.items():
        if df_target_enriched_clusters is None or df_target_enriched_clusters.empty:
            continue
        
        if (target_antigen, 4) in total_reads_s1 and total_reads_s1[(target_antigen, 4)] > 0:
            analysis_round_for_antigen = 4
        else:
            analysis_round_for_antigen = 3
        
        print(f"  - Using R{analysis_round_for_antigen} as final round for target: {target_antigen}")

        excluded_peers = set()
        for group in CROSS_REACTIVITY_EXCLUSION_GROUPS:
            if target_antigen in group:
                excluded_peers.update(g for g in group if g != target_antigen)
        
        other_antigens_to_test = [ag for ag in all_antigens_list if ag != target_antigen and ag not in excluded_peers]
        if excluded_peers:
            print(f"  For target '{target_antigen}', excluding comparisons against: {excluded_peers}")

        worker_args.append((
            target_antigen, df_target_enriched_clusters, df_agg_cluster_counts_s4,
            other_antigens_to_test, total_reads_s1,
            Q_VALUE_CUTOFF_SPECIFICITY, FOLD_CHANGE_CUTOFF_SPECIFICITY,
            analysis_round_for_antigen 
        ))

    if not worker_args:
        print("No cluster/antigen pairs for cross-reactivity analysis."); return pd.DataFrame()
    
    print(f"Launching {len(worker_args)} large-grained analysis tasks for all antigens...")
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results_list_of_dfs = pool.map(worker_process_antigen_specificity, worker_args)
    
    final_specificity_df = pd.concat(results_list_of_dfs, ignore_index=True) if results_list_of_dfs else pd.DataFrame()

    # ------------------------------------------------------------------
    # Annotate each row with its cluster's full member list so that the
    # nonspecific-cluster CSV downstream consumers (and reviewers) can
    # inspect every sequence that was rejected, not just the rep.
    # The map is built from the same MMseqs2 outputs Step VI uses.
    # ------------------------------------------------------------------
    if not final_specificity_df.empty and cluster_tsv_path_s3 and fasta_path_s3 \
            and os.path.exists(cluster_tsv_path_s3) and os.path.exists(fasta_path_s3):
        try:
            mmseqs_clusters_df = pd.read_csv(cluster_tsv_path_s3, sep='\t', header=None,
                                              names=['representative_id', 'member_id'], dtype=str)
            id_to_seq = load_id_to_sequence_map_from_fasta_generic(fasta_path_s3)
            rep_to_member_ids = mmseqs_clusters_df.groupby('representative_id')['member_id'].apply(list).to_dict()
            # Map of rep_sequence -> list of member_sequences (skipping any unresolved IDs)
            rep_to_members_aa = {
                id_to_seq.get(rep_id): [id_to_seq.get(m_id) for m_id in members if id_to_seq.get(m_id)]
                for rep_id, members in rep_to_member_ids.items()
                if id_to_seq.get(rep_id)
            }
            final_specificity_df['ClusterMembers_Count'] = final_specificity_df['RepresentativeSequence'].map(
                lambda r: len(rep_to_members_aa.get(r, [])))
            final_specificity_df['ClusterMembers_List'] = final_specificity_df['RepresentativeSequence'].map(
                lambda r: ";".join(rep_to_members_aa.get(r, [])) if rep_to_members_aa.get(r) else "N/A")
            print(f"  - Annotated {len(final_specificity_df)} clusters with member lists "
                  f"(mean cluster size = {final_specificity_df['ClusterMembers_Count'].mean():.1f}).")
        except Exception as exc:
            print(f"  WARNING: could not annotate cluster members onto Step V output: {exc}")
            final_specificity_df['ClusterMembers_Count'] = 0
            final_specificity_df['ClusterMembers_List'] = "N/A"

    if not final_specificity_df.empty:
        final_specificity_df.to_csv(os.path.join(CURRENT_RUN_OUTPUT_DIR, "step5_cross_reactivity_analysis_report.csv"), index=False)
        with open(os.path.join(CURRENT_RUN_OUTPUT_DIR, "step5_cross_reactivity_analysis.pkl"), 'wb') as f:
            pickle.dump(final_specificity_df, f)
        non_specific_df = final_specificity_df[final_specificity_df['IsSpecific'] == False]
        if not non_specific_df.empty:
            non_specific_df.to_csv(os.path.join(CURRENT_RUN_OUTPUT_DIR, "step5_filtered_nonspecific_clusters.csv"), index=False)

    found_specific_count = final_specificity_df['IsSpecific'].sum() if 'IsSpecific' in final_specificity_df.columns else 0
    print(f"Step V completed. Analyzed {len(final_specificity_df)} total clusters. Found {found_specific_count} specific.", flush=True)
    
    return final_specificity_df


def worker_generate_cluster_report_step6(args):
    """Generates a full report row for one cluster: CDR extraction, enrichment, physicochemical properties, PSSM."""
    row_data, target_ag, positional_backgrounds, aa_sequence_to_cluster_size_map, aa_rep_to_aa_members_map, enriched_clusters_s4 = args
    
    rep_seq = row_data['RepresentativeSequence']
    cdr1, cdr2, cdr3, cdr3_len = parse_cdrs(rep_seq)
    seq_len = len(rep_seq)

    cluster_size = aa_sequence_to_cluster_size_map.get(rep_seq, 1)
    cluster_members = aa_rep_to_aa_members_map.get(rep_seq, [])
    
    avg_entropy_val = np.nan
    kl_divergence_val = np.nan
    pssm_freq_file, pssm_logodds_file = "N/A", "N/A"
    pssm_f_data, pssm_lo_data = None, None
    
    # Only compute PSSM/entropy for clusters with enough members
    if cluster_members and len(cluster_members) >= MIN_CLUSTER_SIZE_FOR_METRICS:
        alignment = run_mafft_alignment_step6(cluster_members, MAFFT_EXECUTABLE, num_threads=1)
        if alignment:
            pssm_f_data, pssm_lo_data, avg_entropy_val = calculate_pssm_and_entropy_step6(alignment, DEFAULT_AA_BACKGROUND)
            background_dist_for_len = positional_backgrounds.get(seq_len)
            if background_dist_for_len:
                kl_divergence_val = calculate_kl_divergence(alignment, background_dist_for_len)

            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', f"{target_ag}_{rep_seq[:10]}_{len(rep_seq)}")
            if pssm_f_data is not None: pssm_freq_file = os.path.join("pssms", f"pssm_freq_{safe_name}.csv")
            if pssm_lo_data is not None: pssm_logodds_file = os.path.join("pssms", f"pssm_logodds_{safe_name}.csv")

    enrich_details_df = enriched_clusters_s4.get(target_ag)
    r1c, r1cpm, r2c, r2cpm, r3c, r3cpm, r4c, r4cpm = (np.nan,) * 8
    if enrich_details_df is not None and not enrich_details_df.empty:
        info = enrich_details_df[enrich_details_df['RepresentativeSequence'] == rep_seq]
        if not info.empty:
            std_cols = ['Count_R1', 'CPM_R1', 'Count_R2', 'CPM_R2', 'Count_R3', 'CPM_R3']
            r1c, r1cpm, r2c, r2cpm, r3c, r3cpm = info.iloc[0][std_cols]
            
            if 'Count_R4' in info.columns:
                r4c, r4cpm = info.iloc[0][['Count_R4', 'CPM_R4']]

    # --- Physicochemical properties ---
    seq_pI, seq_gravy, seq_mw = np.nan, np.nan, np.nan
    cdr3_pI, cdr3_gravy, cdr3_mw, cdr3_charge_pH7 = np.nan, np.nan, np.nan, np.nan
    try:
        pa_seq = ProteinAnalysis(rep_seq)
        seq_pI = pa_seq.isoelectric_point()
        seq_gravy = pa_seq.gravy()
        seq_mw = pa_seq.molecular_weight()
    except Exception:
        pass
    if cdr3 and cdr3 != "N/A":
        try:
            pa_cdr3 = ProteinAnalysis(cdr3)
            cdr3_pI = pa_cdr3.isoelectric_point()
            cdr3_gravy = pa_cdr3.gravy()
            cdr3_mw = pa_cdr3.molecular_weight()
            cdr3_charge_pH7 = pa_cdr3.charge_at_pH(7.0)
        except Exception:
            pass

    report_dict = {
        'RepSequence': rep_seq, 'TargetAntigen': target_ag, 'ClusterSize': cluster_size,
        'CDR1': cdr1, 'CDR2': cdr2, 'CDR3': cdr3, 'CDR3_Length': cdr3_len,
        'R1_Count': r1c, 'R1_CPM': r1cpm, 'R2_Count': r2c, 'R2_CPM': r2cpm, 'R3_Count': r3c, 'R3_CPM': r3cpm, 'R4_Count': r4c, 'R4_CPM': r4cpm,
        'Seq_pI': seq_pI, 'Seq_GRAVY': seq_gravy, 'Seq_MW': seq_mw,
        'CDR3_pI': cdr3_pI, 'CDR3_GRAVY': cdr3_gravy, 'CDR3_MW': cdr3_mw, 'CDR3_Charge_pH7': cdr3_charge_pH7,
        'AvgClusterShannonEntropy': avg_entropy_val,
        'KL_Divergence_from_R0': kl_divergence_val,
        'PSSM_FreqFile': pssm_freq_file, 'PSSM_LogOddsFile': pssm_logodds_file,
        'SpecificityData_JSON': str(row_data['SpecificityDetails']),
        'ClusterMembers_Count': len(cluster_members),
        'ClusterMembers_List': ";".join(c for c in cluster_members if c) if cluster_members else "N/A"
    }
    
    return (report_dict, pssm_f_data, pssm_lo_data)


def main_step6(specificity_results_s5, enriched_clusters_s4, cluster_tsv_path_s3, fasta_path_s3, all_data_s1_with_cpms):
    """Generates final candidate reports with CDR properties, PSSMs, and enrichment data."""
    step6_output_dir = os.path.join(CURRENT_RUN_OUTPUT_DIR, "step6_final_reports_advanced")
    plots_subdir = os.path.join(step6_output_dir, "plots"); pssm_subdir = os.path.join(step6_output_dir, "pssms")
    os.makedirs(plots_subdir, exist_ok=True); os.makedirs(pssm_subdir, exist_ok=True)
    print(f"\n--- Step VI: Final Candidate Reporting & Viz (Output to {step6_output_dir}) ---", flush=True)

    if specificity_results_s5.empty: print("No specificity results. Cannot generate reports."); return pd.DataFrame()
    
    positional_backgrounds = calculate_r0_positional_background(all_data_s1_with_cpms)
    
    mmseqs_clusters_df = pd.read_csv(cluster_tsv_path_s3, sep='\t', header=None, names=['representative_id', 'member_id'], dtype=str)
    id_to_sequence_map_s3 = load_id_to_sequence_map_from_fasta_generic(fasta_path_s3)
    
    print("  Mapping cluster members and sizes...")
    cluster_sizes_map = mmseqs_clusters_df.groupby('representative_id').size().to_dict()
    rep_to_members_map = mmseqs_clusters_df.groupby('representative_id')['member_id'].apply(list).to_dict()
    aa_sequence_to_cluster_size_map = {id_to_sequence_map_s3.get(rep_id): size for rep_id, size in cluster_sizes_map.items()}
    aa_rep_to_aa_members_map = {id_to_sequence_map_s3.get(rep_id): [id_to_sequence_map_s3.get(m_id) for m_id in members] for rep_id, members in rep_to_members_map.items()}
            
    specific_candidates_df = specificity_results_s5[specificity_results_s5['IsSpecific'] == True].copy()
    if specific_candidates_df.empty: print("No specific clusters to report."); return pd.DataFrame()
    
    print(f"Found {len(specific_candidates_df)} specific clusters for parallel report generation.", flush=True)

    worker_args = [(row, row['TargetAntigen'], positional_backgrounds, aa_sequence_to_cluster_size_map, aa_rep_to_aa_members_map, enriched_clusters_s4) for _, row in specific_candidates_df.iterrows()]
    final_report_list, pssm_freq_to_write, pssm_logodds_to_write = [], {}, {}
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(worker_generate_cluster_report_step6, worker_args)

    for report_dict, pssm_f, pssm_lo in results:
        final_report_list.append(report_dict)
        if pssm_f is not None and report_dict['PSSM_FreqFile'] != "N/A": pssm_freq_to_write[report_dict['PSSM_FreqFile']] = pssm_f
        if pssm_lo is not None and report_dict['PSSM_LogOddsFile'] != "N/A": pssm_logodds_to_write[report_dict['PSSM_LogOddsFile']] = pssm_lo

    for rel_path, df_pssm in pssm_freq_to_write.items(): df_pssm.to_csv(os.path.join(step6_output_dir, rel_path))
    for rel_path, df_pssm in pssm_logodds_to_write.items(): df_pssm.to_csv(os.path.join(step6_output_dir, rel_path))

    if not final_report_list: print("No final candidates to report after processing."); return pd.DataFrame()
    
    overall_df = pd.DataFrame(final_report_list)
    for antigen_name, group_df in overall_df.groupby('TargetAntigen'):
        if group_df.empty: continue
        group_df_sorted = group_df.sort_values(by=['ClusterSize', 'KL_Divergence_from_R0'], ascending=[False, False])
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen_name))
        csv_path = os.path.join(step6_output_dir, f"final_candidates_{safe_name}.csv")
        group_df_sorted.to_csv(csv_path, index=False)
        print(f"  - Saved report for {antigen_name} ({len(group_df_sorted)}) to {csv_path}")
        generate_antigen_plots_step6(antigen_name, group_df_sorted, plots_subdir)

    print("Step VI completed.", flush=True); return overall_df

# ==============================================================================
# === MAIN PIPELINE EXECUTION ===
# ==============================================================================

def run_full_pipeline(base_output_dir, min_id_setting, coverage_setting, skip_step5=False, fdr_cutoff=0.05):
    """Orchestrates the full pipeline (Steps I through VI)."""
    global CURRENT_RUN_OUTPUT_DIR, MIN_SEQ_ID, COVERAGE, Q_VALUE_CUTOFF_SPECIFICITY
    MIN_SEQ_ID = min_id_setting
    COVERAGE = coverage_setting
    Q_VALUE_CUTOFF_SPECIFICITY = fdr_cutoff

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_id_{min_id_setting}_cov_{coverage_setting}"
    
    CURRENT_RUN_OUTPUT_DIR = os.path.join(base_output_dir, run_name)
    os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print(f"STARTING FULL PIPELINE RUN for MIN_SEQ_ID = {MIN_SEQ_ID}")
    print(f"Using fixed COVERAGE = {COVERAGE} and FDR = {Q_VALUE_CUTOFF_SPECIFICITY}")
    print(f"Pipeline run output directory: {CURRENT_RUN_OUTPUT_DIR}")
    print(f"Global NUM_PROCESSES: {NUM_PROCESSES}")
    print("="*80)
    
    start_time_pipeline = time.time()

    # --- Step I: Data Loading ---
    s_time = time.time(); all_data_s1, total_reads_s1 = main_step1(); print(f"Step I duration: {time.time()-s_time:.2f}s.")
    if all_data_s1 is None: print("Pipeline halted: Step I failed."); return

    # --- Step II: Sequence-level Enrichment ---
    s_time = time.time(); enriched_data_s2 = main_step2(all_data_s1, total_reads_s1); print(f"Step II duration: {time.time()-s_time:.2f}s.")
    if not enriched_data_s2: print("No sequences after Step II. Halting."); return
    plot_top_precluster_clones(enriched_data_s2, CURRENT_RUN_OUTPUT_DIR) 

    # --- Step 2.5: Pre-Clustering Cross-Reactivity Filter ---
    s_time = time.time(); specific_seqs_for_clustering_s2_5 = main_step2_5_pre_clustering_specificity(enriched_data_s2, all_data_s1); print(f"Step 2.5 duration: {time.time()-s_time:.2f}s.")
    if not specific_seqs_for_clustering_s2_5: print("Pipeline halted: Step 2.5 resulted in zero sequences."); return

    # --- Step III: Clustering ---
    s_time = time.time(); cluster_tsv_path_s3, fasta_path_s3 = main_step3(specific_seqs_for_clustering_s2_5, NUM_PROCESSES); print(f"Step III duration: {time.time()-s_time:.2f}s.")
    if not cluster_tsv_path_s3: print("Pipeline halted: Step III failed."); return

    # --- Step 3.5: Initial Cluster Composition Analysis ---
    s_time = time.time(); main_step3_5_initial_cluster_analysis(cluster_tsv_path_s3, fasta_path_s3, all_data_s1); print(f"Step 3.5 duration: {time.time()-s_time:.2f}s.")

    # --- Step IV: Cluster-wise Enrichment ---
    s_time = time.time(); enriched_clusters_s4, df_agg_cluster_counts_s4 = main_step4(cluster_tsv_path_s3, fasta_path_s3, all_data_s1, total_reads_s1); print(f"Step IV duration: {time.time()-s_time:.2f}s.")
    if not enriched_clusters_s4: print("No enriched clusters after Step IV. Halting."); return

    # --- Step V: Conditional Specificity Analysis ---
    if not skip_step5:
        s_time = time.time()
        specificity_results_s5 = main_step5(enriched_clusters_s4, df_agg_cluster_counts_s4, total_reads_s1,
                                            cluster_tsv_path_s3=cluster_tsv_path_s3,
                                            fasta_path_s3=fasta_path_s3)
        print(f"Step V duration: {time.time()-s_time:.2f}s.")
        if specificity_results_s5.empty: 
            print("No specific clusters found after Step V. Halting."); return
    else:
        print("\n--- SKIPPING Step V: Post-Clustering Specificity Analysis ---", flush=True)
        all_enriched_list = []
        for antigen, df_enriched in enriched_clusters_s4.items():
            if df_enriched is not None and not df_enriched.empty:
                temp_df = pd.DataFrame({
                    'RepresentativeSequence': df_enriched['RepresentativeSequence'],
                    'TargetAntigen': antigen,
                    'IsSpecific': True,
                    'SpecificityDetails': '[{"Reason": "Step_V_Skipped"}]'
                })
                all_enriched_list.append(temp_df)
        
        if not all_enriched_list:
            print("No enriched clusters from Step IV to pass to reporting. Halting.")
            return
        
        specificity_results_s5 = pd.concat(all_enriched_list, ignore_index=True)
        print(f"  Treating all {len(specificity_results_s5)} enriched clusters from Step IV as specific candidates.")

    # --- Step VI: Final Reporting ---
    s_time = time.time(); final_candidates_s6_df = main_step6(specificity_results_s5, enriched_clusters_s4, cluster_tsv_path_s3, fasta_path_s3, all_data_s1); print(f"Step VI duration: {time.time()-s_time:.2f}s.")

    print(f"\nTotal pipeline (Steps 1-6) duration for MIN_SEQ_ID={MIN_SEQ_ID}: {time.time() - start_time_pipeline:.2f} seconds.")
    print(f"All outputs and intermediate files saved in: {CURRENT_RUN_OUTPUT_DIR}")
    
    print(f"FINAL_RUN_DIRECTORY:{CURRENT_RUN_OUTPUT_DIR}")  # read by the calling shell script
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultimate Nanobody Analysis Pipeline (Steps I-VIII).")
    parser.add_argument("--min-seq-id", type=float, required=True, help="MMseqs2 min-seq-id setting for the clustering step (e.g., 0.8).")
    parser.add_argument("--coverage", type=float, required=True, help="MMseqs2 coverage setting (e.g., 0.9).")
    parser.add_argument("--exclude-groups", type=str, default="[]", help="Antigen groups to exclude from cross-reactivity checks.")
    parser.add_argument("--skip-step5-specificity", action='store_true', help="If specified, skips the post-clustering specificity analysis (Step V).")
    parser.add_argument("--fdr", type=float, default=0.05, help="The False Discovery Rate (FDR) q-value cutoff for specificity tests. Default: 0.05.")
    parser.add_argument("--base-output-dir", type=str, required=True, help="The base directory for all pipeline run outputs.")
    args = parser.parse_args()

    try:
        CROSS_REACTIVITY_EXCLUSION_GROUPS = [set(g) for g in ast.literal_eval(args.exclude_groups)]
        print(f"INFO: Using cross-reactivity exclusion groups: {CROSS_REACTIVITY_EXCLUSION_GROUPS}")
    except (ValueError, SyntaxError) as e:
        print(f"CRITICAL ERROR: Could not parse --exclude-groups argument. Error: {e}"); exit(1)

    run_full_pipeline(
        base_output_dir=args.base_output_dir,
        min_id_setting=args.min_seq_id, 
        coverage_setting=args.coverage,
        skip_step5=args.skip_step5_specificity,
        fdr_cutoff=args.fdr
    )