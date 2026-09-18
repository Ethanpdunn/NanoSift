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
PIPELINE_OUTPUT_BASE_DIR = "."
RANKED_SEQS_DIR = "."
NAIVE_LIBRARY_R0_FILE = "."
MMSEQS_EXECUTABLE = "mmseqs"
MAFFT_EXECUTABLE = "mafft"

# --- Step II: Sequence Filtering Parameters ---
CPM_CUTOFF = 0.0
CPM_CUTOFF_R3 = 5.0

FISHER_P_VALUE_THRESHOLD_FOR_DECREASE = 0.05

# --- Step III: Clustering Parameters (MIN_SEQ_ID is set via command line) ---
MIN_SEQ_ID = 0.8
COVERAGE = 0.9  
COV_MODE = 0
CLUSTER_MODE = 0

# --- Step V: Specificity Analysis Parameters ---
Q_VALUE_CUTOFF_SPECIFICITY = 0.05
FINAL_ANALYSIS_ROUND = 4
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
        if df_enriched is None or df_enriched.empty or 'CPM_R3' not in df_enriched.columns:
            print(f"  - No enriched data with R3 CPM for antigen '{antigen}'. Skipping plot.")
            continue
            
        top_20_clones = df_enriched.nlargest(20, 'CPM_R3')
        
        if top_20_clones.empty:
            print(f"  - No top clones found for antigen '{antigen}'. Skipping plot.")
            continue
            
        print(f"  - Generating enrichment profile plot for top {len(top_20_clones)} clones for {antigen}...")
        
        plt.figure(figsize=(12, 8))
        for _, row in top_20_clones.iterrows():
            cpm_values = [
                row.get('CPM_R1', np.nan),
                row.get('CPM_R2', np.nan),
                row.get('CPM_R3', np.nan)
            ]
            label = f"{row['Sequence'][:15]}..."
            plt.plot(['R1', 'R2', 'R3'], cpm_values, marker='o', linestyle='-', label=label)
            
        plt.title(f'Top {len(top_20_clones)} Pre-Clustering Clone Profiles - Antigen: {antigen}')
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

def get_r0_biased_sequences_step2(df_r0, total_r0_r, cutoff_perc):
    if df_r0 is None or df_r0.empty or total_r0_r == 0: return set()
    df_c = df_r0.copy()
    if 'CPM' in df_c.columns: df_c['Proportion'] = df_c['CPM']
    elif total_r0_r > 0: df_c['Proportion'] = df_c['Count'] / total_r0_r
    else: df_c['Proportion'] = 0.0
    if df_c['Proportion'].empty: return set()
    thresh = np.percentile(df_c['Proportion'], cutoff_perc)
    biased = set(df_c[df_c['Proportion'] >= thresh]['Sequence'])
    print(f"  Identified {len(biased)} R0-biased sequences (top {100-cutoff_perc:.6f}%). Threshold: {thresh:.3e}")
    return biased

def perform_enrichment_filtering_for_antigen_generic(df_r1_p, df_r2_p, df_r3_p, t_r1_o, t_r2_o, t_r3_o, p_val_thresh, id_col_name='Sequence'):
    """Filters sequences/clusters for monotonic enrichment across rounds using Fisher's exact test."""
    cols_needed = [id_col_name, 'Count']
    df_r1 = df_r1_p if df_r1_p is not None else pd.DataFrame(columns=cols_needed)
    df_r2 = df_r2_p if df_r2_p is not None else pd.DataFrame(columns=cols_needed)
    df_r3 = df_r3_p if df_r3_p is not None else pd.DataFrame(columns=cols_needed)
    
    for df in [df_r1, df_r2, df_r3]:
        for col in cols_needed:
            if col not in df.columns:
                df[col] = 0 if col == 'Count' else pd.NA

    df_m = pd.merge(df_r1[cols_needed], df_r2[cols_needed], on=id_col_name, how='outer', suffixes=('_R1', '_R2'))
    df_r3_to_merge = df_r3[cols_needed].rename(columns={'Count': 'Count_R3'})
    df_m = pd.merge(df_m, df_r3_to_merge, on=id_col_name, how='outer')

    df_m = df_m.fillna(0)
    for col in ['Count_R1', 'Count_R2', 'Count_R3']: df_m[col] = df_m[col].astype(int)

    df_m['CPM_R1'] = df_m['Count_R1'] / (t_r1_o + 1e-9) * 1e6
    df_m['CPM_R2'] = df_m['Count_R2'] / (t_r2_o + 1e-9) * 1e6
    df_m['CPM_R3'] = df_m['Count_R3'] / (t_r3_o + 1e-9) * 1e6

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

    enriched_df = df_m[ok12 & ok23].copy()
    
    final_cols = {id_col_name: id_col_name, 'Count_R1': 'Count_R1', 'CPM_R1': 'CPM_R1', 
                  'Count_R2': 'Count_R2', 'CPM_R2': 'CPM_R2', 'Count_R3': 'Count_R3', 'CPM_R3': 'CPM_R3'}
    
    final_df_cols = [col for col in final_cols.keys() if col in enriched_df.columns]
    return enriched_df[final_df_cols].rename(columns=final_cols)


def worker_enrichment_filter_step2(args):
    antigen_id, df_r1_p, df_r2_p, df_r3_p, t_r1_o, t_r2_o, t_r3_o, p_val_thresh = args
    return antigen_id, perform_enrichment_filtering_for_antigen_generic(df_r1_p, df_r2_p, df_r3_p, t_r1_o, t_r2_o, t_r3_o, p_val_thresh, id_col_name='Sequence')

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
    
    pivot_df = df_agg_for_antigen.pivot_table(
        index='RepresentativeSequence', 
        columns='Round', 
        values='Count', 
        fill_value=0
    )

    for round_num in [1, 2, 3]:
        if round_num not in pivot_df.columns:
            pivot_df[round_num] = 0
            
    pivot_df = pivot_df.rename(columns={1: 'Count_R1', 2: 'Count_R2', 3: 'Count_R3'})

    t_r1_o = total_reads_s1.get((antigen_id, 1), 0)
    t_r2_o = total_reads_s1.get((antigen_id, 2), 0)
    t_r3_o = total_reads_s1.get((antigen_id, 3), 0)

    pivot_df['CPM_R1'] = (pivot_df['Count_R1'] / t_r1_o) * 1e6 if t_r1_o > 0 else 0
    pivot_df['CPM_R2'] = (pivot_df['Count_R2'] / t_r2_o) * 1e6 if t_r2_o > 0 else 0
    pivot_df['CPM_R3'] = (pivot_df['Count_R3'] / t_r3_o) * 1e6 if t_r3_o > 0 else 0

    ok23 = pivot_df['CPM_R3'] >= pivot_df['CPM_R2']
    ok12 = pivot_df['CPM_R2'] >= pivot_df['CPM_R1']
    
    enriched_df = pivot_df[ok12 & ok23].reset_index()
    
    return antigen_id, enriched_df[['RepresentativeSequence', 'Count_R1', 'CPM_R1', 'Count_R2', 'CPM_R2', 'Count_R3', 'CPM_R3']]
# --- Step V Helpers ---

def worker_process_antigen_specificity(args):
    """Runs specificity analysis for all clusters of a single target antigen."""
    target_antigen, df_target_enriched, df_agg_r3, all_other_antigens, total_reads_s1_r3, q_value_cutoff, fc_cutoff = args
    
    print(f"  - Starting specificity analysis for {len(df_target_enriched)} clusters of target: {target_antigen}")
    
    r3_pivot = df_agg_r3.pivot_table(index='RepresentativeSequence', columns='AntigenID', values='Count', fill_value=0)
    
    target_clusters_df = r3_pivot[r3_pivot.index.isin(df_target_enriched['RepresentativeSequence'])]
    
    if target_clusters_df.empty:
        return pd.DataFrame()

    results = []
    total_reads_target = total_reads_s1_r3.get(target_antigen, 0)
    if total_reads_target == 0: return pd.DataFrame()

    for rep_sequence, row in target_clusters_df.iterrows():
        count_target = row[target_antigen]
        prop_target = count_target / total_reads_target
        
        p_values = []
        comparisons_data = []

        for other_antigen in all_other_antigens:
            if other_antigen not in row.index: continue
            count_other = row[other_antigen]
            total_reads_other = total_reads_s1_r3.get(other_antigen, 0)
            
            prop_other, fold_change, p_value = 0.0, np.inf, 1.0
            if total_reads_other > 0:
                prop_other = count_other / total_reads_other
                fold_change = prop_target / (prop_other + 1e-9)
                table = [[count_target, int(total_reads_target - count_target)], [count_other, int(total_reads_other - count_other)]]
                table[0][1], table[1][1] = max(0, table[0][1]), max(0, table[1][1])
                try:
                    _, p_value = fisher_exact(table, alternative='greater')
                except ValueError:
                    pass
            
            p_values.append(p_value)
            comparisons_data.append({'ComparisonAntigen': other_antigen, 'P_Value': p_value, 'FoldChange': fold_change})

        is_specific = True
        if p_values:
            try:
                reject, q_values, _, _ = multipletests(p_values, alpha=q_value_cutoff, method='fdr_bh')
                for i, comp_data in enumerate(comparisons_data):
                    comp_data['Q_Value'] = float(q_values[i])
                    is_cross_reactive = not (q_values[i] < q_value_cutoff and comp_data['FoldChange'] > fc_cutoff)
                    if is_cross_reactive and target_clusters_df.loc[rep_sequence, comp_data['ComparisonAntigen']] > 0:
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
                q_freq = 1e-9

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
    """
    Generates cluster size histogram and enrichment profile plots for an antigen.
    """
    global plt, sns, TOP_N_CLUSTERS_TO_PLOT
    if plt is None or sns is None: print(f"Plotting libraries not available for Step VI plots for {antigen_name}."); return
    if antigen_report_df.empty: print(f"  No candidates for {antigen_name} to plot."); return
    
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen_name))
    
    # --- Cluster Size Plot (Now with Log Scale) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(antigen_report_df['ClusterSize'], bins=max(1, min(20, antigen_report_df['ClusterSize'].nunique())), kde=False)
    plt.title(f'Cluster Sizes - Antigen: {antigen_name}')
    plt.xlabel('Cluster Size')
    plt.ylabel('# Clusters (Log Scale)') # Updated Label
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

def analyze_and_plot_cpm_weighted_pssms(output_dir, all_data_s1, step6_reports_dir):
    """
    Generates CPM-weighted PSSM visualizations for final candidates, grouped by length.
    """
    print("\n--- Generating CPM-Weighted PSSM Logo Visualizations ---")
    pssm_output_dir = os.path.join(output_dir, "cpm_weighted_pssm_visualizations")
    os.makedirs(pssm_output_dir, exist_ok=True)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # --- 1. Load data from all rounds (R3 and R4) ---
    all_final_round_data_list = [
        df.assign(Antigen=k[0], Round=k[1]) 
        for k, df in all_data_s1.items() 
        if (k[1] == 3 or k[1] == 4) and 'CPM' in df.columns
    ]
    
    if not all_final_round_data_list:
        print("  [WARNING] No Round 3 or 4 data with CPMs found. Skipping CPM-weighted PSSM analysis.")
        return
    all_final_rounds_df = pd.concat(all_final_round_data_list)

    # --- 2. Analyze and plot final candidates for each antigen ---
    print("  - Generating PSSMs for final candidate pools...")
    final_candidate_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    if not final_candidate_files:
        print("    No final candidate files found. Skipping.")
        return
        
    for report_file in final_candidate_files:
        antigen_name = re.search(r"final_candidates_(.+)\.csv", os.path.basename(report_file)).group(1)
        print(f"    - Processing antigen: {antigen_name}")
        df_report = pd.read_csv(report_file)
        
        antigen_data = all_final_rounds_df[all_final_rounds_df['Antigen'] == antigen_name]
        if antigen_data.empty:
            continue
            
        # Default to R3, but use R4 if it exists
        analysis_round_for_antigen = 3
        if 4 in antigen_data['Round'].values:
            analysis_round_for_antigen = 4
            
        print(f"      - Using R{analysis_round_for_antigen} data for weighted PSSM.")
        antigen_cpm_df = antigen_data[antigen_data['Round'] == analysis_round_for_antigen][['Sequence', 'CPM']]

        # Aggregate all member sequences from the report
        all_members = df_report['ClusterMembers_List'].dropna().str.split(';').explode()
        if all_members.empty:
            continue
            
        # Get final round CPM for each unique member sequence
        df_members = pd.DataFrame({'Sequence': all_members}).drop_duplicates()
        df_members_with_cpm = pd.merge(df_members, antigen_cpm_df, on='Sequence', how='inner')
        df_members_with_cpm['Length'] = df_members_with_cpm['Sequence'].str.len()
        
        # --- 3. Group by target lengths and generate a weighted PSSM for each ---
        target_lengths = [24, 27, 30]
        for length in target_lengths:
            group_df = df_members_with_cpm[df_members_with_cpm['Length'] == length]

            if len(group_df) < 10: # Using a default minimum of 10 sequences
                continue

            # Calculate CPM-weighted PSSM
            total_cpm_for_length = group_df['CPM'].sum()
            if total_cpm_for_length == 0: continue
            
            positional_cpm_sums = [defaultdict(float) for _ in range(length)]
            for _, row in group_df.iterrows():
                seq, cpm = row['Sequence'], row['CPM']
                for i, aa in enumerate(seq):
                    if aa in amino_acids:
                        positional_cpm_sums[i][aa] += cpm
            
            positional_freqs = []
            for pos_sums in positional_cpm_sums:
                freq_dict = {aa: cpm_sum / total_cpm_for_length for aa, cpm_sum in pos_sums.items()}
                positional_freqs.append(freq_dict)

            if not positional_freqs: continue
            freq_df = pd.DataFrame(positional_freqs).fillna(0).T
            freq_df.columns = range(len(freq_df.columns))

            # Generate and save the plot
            title = f"CPM-Weighted PSSM (R{analysis_round_for_antigen}) for Final Candidates\nAntigen: {antigen_name} (Length = {length}, n={len(group_df)})"
            out_path = os.path.join(pssm_output_dir, f"cpm_weighted_summary_logo_{antigen_name}_len{length}.png")
            generate_sequence_logo_plot(freq_df, title, out_path)
            
    print("    Done.")


def generate_sequence_logo_plot(freq_df, title, output_path):
    """
    Generates and saves a PSSM sequence logo using matplotlib.text.TextPath.
    
    Args:
        freq_df (pd.DataFrame): A DataFrame with positions as columns and amino acids as rows,
                                containing frequency data.
        title (str): The title for the plot.
        output_path (str): The path to save the output PNG file.
    """
    if plt is None: return
    
    protein_colormap = {
        'A': '#6DD7A1', 'I': '#55C08C', 'L': '#55C08C', 'V': '#55C08C', 'M': '#55C08C',
        'F': '#B897EC', 'Y': '#B897EC', 'W': '#A180D2', 'S': '#FFBE74', 'T': '#FFBE74',
        'N': '#77EAF4', 'Q': '#77EAF4', 'D': '#EE8485', 'E': '#EE8485', 'H': '#96C4FF',
        'K': '#7FADEA', 'R': '#7FADEA', 'C': '#FAED70', 'G': '#E2DEDD', 'P': '#FFB1F1'
    }
    
    fig, ax = plt.subplots(figsize=(max(10, len(freq_df.columns) * 0.8), 5))
    
    # Calculate the total information content (height) for each position
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore log(0) warnings
        shannon_entropy = -np.sum(freq_df * np.log2(freq_df), axis=0).fillna(0)
    max_entropy = np.log2(20) # Max possible entropy for 20 amino acids
    information_content = max_entropy - shannon_entropy

    for i, pos in enumerate(freq_df.columns):
        y0 = 0
        # Sort amino acids by frequency for clean stacking in the logo
        sorted_aas = freq_df[pos].sort_values(ascending=True)
        
        for aa, freq in sorted_aas.items():
            if freq > 0.001: # Only plot if frequency is non-trivial
                # The height of each letter is its frequency scaled by the total info content
                letter_height = freq * information_content[pos]
                
                path = TextPath((0, 0), aa, size=1, prop={'family': 'monospace', 'weight': 'bold'})
                transform = Affine2D().scale(1, letter_height).translate(i - 0.4, y0)
                path = path.transformed(transform)
                
                patch = PathPatch(path, facecolor=protein_colormap.get(aa, '#93908F'), edgecolor='black', lw=0.5)
                ax.add_patch(patch)
                y0 += letter_height
                
    ax.set_xlim(-0.5, len(freq_df.columns) - 0.5)
    ax.set_ylim(0, max_entropy)
    ax.set_xticks(range(len(freq_df.columns)))
    ax.set_xticklabels(freq_df.columns)
    ax.set_xlabel('Position')
    ax.set_ylabel('Information Content (bits)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def analyze_kl_divergence_metrics(output_dir, step6_reports_dir):
    """
    Generates violin and scatter plots for KL Divergence metrics.
    """
    if plt is None or sns is None: print("Plotting libraries not available. Skipping KL Divergence analysis."); return
    print("\n--- Analyzing KL Divergence Metrics ---")
    
    final_candidate_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    if not final_candidate_files: print("  - No final candidate files found. Skipping."); return
        
    all_candidates_df = pd.concat([pd.read_csv(f) for f in final_candidate_files], ignore_index=True)
    plot_df = all_candidates_df[all_candidates_df['ClusterSize'] >= 5].copy().dropna(subset=['KL_Divergence_from_R0'])
    del all_candidates_df
    if plot_df.empty:
        print("  - No clusters with size >= 5 and valid KL Divergence scores found. Skipping plots.")
        return

    # Plot 1: Violin plot of KL Divergence distributions
    plt.figure(figsize=(max(10, len(plot_df['TargetAntigen'].unique()) * 1.2), 8))
    sns.violinplot(data=plot_df, x='TargetAntigen', y='KL_Divergence_from_R0')
    plt.title('Selection Pressure (KL Divergence from R0) by Antigen\n(For Clusters with Size >= 5)')
    plt.ylabel('KL Divergence (bits)')
    plt.xlabel('Target Antigen')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_divergence_distribution.png")); plt.close()
    print("  - Saved plot: KL Divergence Distribution by Antigen")

    # Plot 2: Scatter plot of KL Divergence vs. Cluster Size
    g = sns.FacetGrid(plot_df, col="TargetAntigen", col_wrap=min(4, len(plot_df['TargetAntigen'].unique())), height=5, sharex=False, sharey=False)
    g.map(sns.scatterplot, "ClusterSize", "KL_Divergence_from_R0", alpha=0.7)
    g.set_axis_labels("Cluster Size", "KL Divergence (bits)")
    g.set_titles("{col_name}")
    g.fig.suptitle("KL Divergence vs. Cluster Size", y=1.03); plt.xscale('log')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "kl_divergence_vs_clustersize.png")); plt.close()
    print("  - Saved plot: KL Divergence vs. Cluster Size")


def plot_pssm_logo(positional_freqs, title, output_path):
    """
    Generates and saves a PSSM sequence logo plot from positional frequency data.
    Uses the matplotlib TextPath approach for creating the logo.
    """
    if not positional_freqs:
        return
        
    # Standard protein amino acid colormap
    protein_colormap = {
        'A': '#6DD7A1', 'I': '#55C08C', 'L': '#55C08C', 'V': '#55C08C', 'M': '#55C08C',
        'F': '#B897EC', 'Y': '#B897EC', 'W': '#A180D2', 'S': '#FFBE74', 'T': '#FFBE74',
        'N': '#77EAF4', 'Q': '#77EAF4', 'D': '#EE8485', 'E': '#EE8485', 'H': '#96C4FF',
        'K': '#7FADEA', 'R': '#7FADEA', 'C': '#FAED70', 'G': '#E2DEDD', 'P': '#FFB1F1',
        'X': '#93908F', '-': '#FFFFFF', '.': '#3F3F3F'
    }

    fig, ax = plt.subplots(figsize=(max(10, len(positional_freqs) * 0.5), 5))
    
    for i, freqs in enumerate(positional_freqs):
        y0 = 0
        # Sort amino acids by frequency for clean stacking in the logo
        for aa, freq in sorted(freqs.items(), key=lambda x: x[1]):
            if freq > 0.001: # Only plot if frequency is non-trivial
                try:
                    path = TextPath((0, 0), aa, size=1)
                    bbox = path.get_extents()
                    transform = Affine2D().scale(1.0 / bbox.width, freq / bbox.height).translate(i, y0)
                    path = path.transformed(transform)
                    patch = PathPatch(path, facecolor=protein_colormap.get(aa, '#93908F'), edgecolor='none')
                    ax.add_patch(patch)
                    y0 += freq
                except Exception:
                    continue # Skip characters that can't be rendered

    ax.set_xlim(-0.5, len(positional_freqs) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_and_plot_pssms(output_dir, all_data_s1, step6_reports_dir):
    """
    Generates PSSM logo visualizations for R0 background and final candidates, split by length.
    """
    print("\n--- Generating PSSM Logo Visualizations ---")
    pssm_output_dir = os.path.join(output_dir, "pssm_visualizations")
    os.makedirs(pssm_output_dir, exist_ok=True)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    print("  - Generating PSSMs for R0 background...")
    r0_backgrounds = calculate_r0_positional_background(all_data_s1)
    for length, freqs_list_of_dicts in r0_backgrounds.items():
        if not freqs_list_of_dicts: continue
        bg_freq_df = pd.DataFrame(freqs_list_of_dicts).T
        bg_freq_df.columns = range(len(bg_freq_df.columns))
        title = f"R0 Naive Library Background Frequencies (Length = {length})"
        out_path = os.path.join(pssm_output_dir, f"pssm_background_R0_len{length}.png")
        generate_sequence_logo_plot(bg_freq_df, title, out_path)
    print("    Done.")

    # --- 2. Analyze and plot final candidates for each antigen ---
    print("  - Generating PSSMs for final candidate pools...")
    final_candidate_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    if not final_candidate_files:
        print("    No final candidate files found. Skipping."); return
        
    for report_file in final_candidate_files:
        antigen_name = re.search(r"final_candidates_(.+)\.csv", os.path.basename(report_file)).group(1)
        print(f"    - Processing antigen: {antigen_name}")
        df_candidates = pd.read_csv(report_file)
        
        all_members = [seq for member_list in df_candidates['ClusterMembers_List'].dropna() for seq in member_list.split(';')]
        if not all_members: continue

        df_members = pd.DataFrame({'Sequence': all_members})
        df_members['Length'] = df_members['Sequence'].str.len()
        
        # Group all sequences by their length
        for length, group in df_members.groupby('Length'):
            # Only process specified lengths with enough data
            if length not in [24, 27, 30] or len(group) < 10: continue
            
            sequences_for_pssm = group['Sequence'].tolist()
            
            positional_freqs = []
            for i in range(length):
                column_aas = [seq[i] for seq in sequences_for_pssm if len(seq) > i]
                total_in_col = len(column_aas)
                if total_in_col == 0: continue
                counts = Counter(column_aas)
                freq_dict = {aa: counts.get(aa, 0) / total_in_col for aa in amino_acids}
                positional_freqs.append(freq_dict)
            
            if not positional_freqs: continue

            freq_df = pd.DataFrame(positional_freqs).T
            # Ensure columns are named correctly for the plotting function
            freq_df.columns = range(len(freq_df.columns))

            title = f"Consolidated PSSM for All Final Candidates\nAntigen: {antigen_name} (Length = {length})"
            out_path = os.path.join(pssm_output_dir, f"pssm_final_candidates_{antigen_name}_len{length}.png")
            generate_sequence_logo_plot(freq_df, title, out_path)
            
    print("    Done.")


def init_worker_s7(data_s1):
    """Initializer for the multiprocessing pool to make data available to each worker."""
    global global_all_data_s1
    global_all_data_s1 = data_s1

def worker_prep_jaccard_set_s7(args):
    """
    Creates a set of sequences above CPM threshold for a single antigen.
    """
    antigen_id, analysis_round, cpm_threshold = args
    
    df_sample = global_all_data_s1.get((antigen_id, analysis_round))
    
    if df_sample is None or df_sample.empty:
        return antigen_id, set()
        
    df_cpm = df_sample.groupby('Sequence')['CPM'].sum()
    
    sig_set = set(df_cpm[df_cpm >= cpm_threshold].index)
    
    return antigen_id, sig_set

def analyze_cdr3_length_progression(output_dir, all_data_s1):
    """Generates CDR3 length distribution plots across rounds using weighted calculations."""
    if plt is None or sns is None: print("Plotting libraries not available. Skipping CDR3 progression analysis."); return
    print("\n--- Analyzing CDR3 Length Progression Across Rounds ---")
    
    # --- Step 1: Prepare data efficiently without exploding the list ---
    print("  - Preparing data for analysis...")
    all_cdr_data = []
    antigen_ids = sorted(list(set(k[0] for k in all_data_s1.keys() if k[0] != 'R0_Naive')))
    for antigen in antigen_ids:
        for round_num in [1, 2, 3, 4]:
            df_sample = all_data_s1.get((antigen, round_num))
            if df_sample is None or df_sample.empty: continue
            
            # This part is now much more efficient
            df_filtered = df_sample.copy()
            df_filtered['CDR3_Length'] = df_filtered['Sequence'].apply(lambda s: len(s[CDR3_START_POS:]) if isinstance(s, str) and len(s) > CDR3_START_POS else 0)
            df_filtered = df_filtered[df_filtered['CDR3_Length'] > 0]
            if df_filtered.empty: continue
            
            # Keep the data aggregated
            df_filtered['Antigen'] = antigen
            df_filtered['Round'] = f"R{round_num}"
            all_cdr_data.append(df_filtered[['Antigen', 'Round', 'CDR3_Length', 'Count']])

    if not all_cdr_data:
        print("  - No data available for analysis. Skipping.")
        return
        
    df_lengths = pd.concat(all_cdr_data, ignore_index=True)
            
    # --- 1. Faceted Stacked Bar Plot of Proportions ---
    print("  - Generating faceted stacked bar plot of CDR3 length proportions...")
    try:
        lengths_to_plot = [9, 12, 15]
        
        # Calculate proportions for just the specified lengths
        counts = df_lengths[df_lengths['CDR3_Length'].isin(lengths_to_plot)].groupby(['Antigen', 'Round', 'CDR3_Length']).size()
        group_totals = counts.groupby(['Antigen', 'Round']).transform('sum')
        proportions = (counts / group_totals).unstack(level='CDR3_Length', fill_value=0)
        
        # Ensure all required columns exist, even if there's no data for a length
        for length in lengths_to_plot:
            if length not in proportions.columns:
                proportions[length] = 0.0
        proportions = proportions[lengths_to_plot] # Enforce order
        
        # Reset index to make 'Antigen' and 'Round' columns for plotting
        plot_df = proportions.reset_index()
        
        # Use seaborn's FacetGrid to create a plot for each antigen
        g = sns.FacetGrid(plot_df, col="Antigen", col_wrap=min(3, len(plot_df['Antigen'].unique())), height=5, sharey=True)
        
        # For each subplot, create the stacked bar plot
        g.map_dataframe(lambda data, color: data.set_index('Round')[lengths_to_plot].plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8, rot=0))

        g.set_titles("{col_name}")
        g.set_axis_labels("Selection Round", "Proportion of Sequences")
        g.add_legend(title="CDR3 Length")
        g.fig.suptitle("CDR3 Length Proportion Progression by Antigen", y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_cdr3_proportions.png"))
        plt.close()
        print("    Done.")
    except Exception as e:
        print(f"    WARNING: Could not generate faceted proportion plot. Error: {e}")
    
    # --- Plot 2: Bar Plot of Weighted Average CDR3 Length ---
    print("  - Generating weighted average length bar plot...")
    
    def weighted_avg(group):
        return np.average(group['CDR3_Length'], weights=group['Count'])

    # Calculate the weighted average for each antigen and round
    avg_lengths = df_lengths.groupby(['Antigen', 'Round']).apply(weighted_avg, include_groups=False).reset_index(name='Weighted_Avg_CDR3_Length')
    
    plt.figure(figsize=(max(10, len(avg_lengths['Antigen'].unique()) * 0.9), 7))
    sns.barplot(x='Antigen', y='Weighted_Avg_CDR3_Length', hue='Round', data=avg_lengths, hue_order=['R1', 'R2', 'R3', 'R4'])
    plt.title('Weighted Average CDR3 Length by Antigen and Round')
    plt.ylabel("Weighted Average CDR3 Length (amino acids)")
    plt.xlabel("Antigen")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barplot_cdr3_average_length.png")); plt.close()
    print("    Done.")
    
    # --- NOTE on Box Plot ---
    # The original box plot was removed because it cannot handle weighted data efficiently
    # and was a primary cause of the memory crash. The weighted average bar plot is a
    # more statistically accurate and memory-safe representation of the central tendency.
    




# ==============================================================================
# === POST-PIPELINE ANALYSIS (formerly Step 7) ===
# ==============================================================================


def analyze_top_n_sequence_retention(run_directory, all_data_s1, step6_reports_dir, output_dir):
    """
    Tracks survival rate of top sequences across pipeline stages.
    """
    print("\n--- Analyzing Top N Sequence Retention ---")
    
    ranks_to_check = [100, 1000, 10000]
    
    # --- 1. Load all sequence sets that passed filters ---
    s2_data = load_pickle(os.path.join(run_directory, "step2_enriched_sequences.pkl"), "Step 2 Enriched Seqs")
    s2_5_data = load_pickle(os.path.join(run_directory, "step2_5_specific_sequences_for_clustering.pkl"), "Step 2.5 Specific Seqs")
    
    s2_set = set(seq for df in s2_data.values() if df is not None and 'Sequence' in df.columns for seq in df['Sequence'])
    s2_5_set = set(s2_5_data) if s2_5_data is not None else set()
    
    # Load all sequences that ended up in a final, specific cluster
    s6_member_sets_by_antigen = {}
    s6_rep_sets_by_antigen = {}
    final_candidate_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    if not final_candidate_files:
        print("  - No Step 6 reports found. Skipping.")
        return
        
    for report_file in final_candidate_files:
        try:
            df = pd.read_csv(report_file)
            if df.empty: continue
            antigen_name = df['TargetAntigen'].iloc[0]
            # Get all unique members from all clusters for this antigen
            members = df['ClusterMembers_List'].dropna().str.split(';').explode().unique()
            s6_member_sets_by_antigen[antigen_name] = set(members)
            # Get all representative sequences for this antigen
            s6_rep_sets_by_antigen[antigen_name] = set(df['RepSequence'].unique())
        except Exception as e:
            print(f"  - Warning: could not process Step 6 report {os.path.basename(report_file)}. Error: {e}")

    # --- 2. Loop through each antigen and check retention ---
    results = []
    antigen_list = sorted(list(set(k[0] for k in all_data_s1.keys() if k[0] != 'R0_Naive')))
    
    for antigen in antigen_list:
        # Determine this antigen's final round
        if (antigen, 4) in all_data_s1:
            final_round = 4
        elif (antigen, 3) in all_data_s1:
            final_round = 3
        else:
            continue # Skip if no R3 or R4 data
            
        df_final_round = all_data_s1.get((antigen, final_round))
        if df_final_round is None or df_final_round.empty:
            continue
            
        # Rank sequences by count in the final round
        df_final_round['Rank'] = df_final_round['Count'].rank(ascending=False, method='first')
        
        # Get this antigen's specific filter sets
        s6_member_set = s6_member_sets_by_antigen.get(antigen, set())
        s6_rep_set = s6_rep_sets_by_antigen.get(antigen, set())
        
        for n_rank in ranks_to_check:
            top_n_set = set(df_final_round[df_final_round['Rank'] <= n_rank]['Sequence'])
            total_in_top_n = len(top_n_set)
            if total_in_top_n == 0:
                continue
            
            # Calculate how many survived each step
            survived_s2 = len(top_n_set.intersection(s2_set))
            survived_s2_5 = len(top_n_set.intersection(s2_5_set))
            survived_s6_members = len(top_n_set.intersection(s6_member_set))
            survived_s6_reps = len(top_n_set.intersection(s6_rep_set)) # How many became a rep
            
            results.append({
                'Antigen': antigen,
                'Top_N': n_rank,
                'Total_Top_N_Seqs': total_in_top_n,
                'Survived_Step2_Enrich': survived_s2,
                'Survived_Step2.5_PreCluster_Specific': survived_s2_5,
                'Survived_Step6_As_Cluster_Member': survived_s6_members,
                'Survived_Step6_As_Cluster_Rep': survived_s6_reps
            })

    if not results:
        print("  - No data to report for Top N retention.")
        return

    # --- 3. Save Report ---
    df_results = pd.DataFrame(results)
    report_path = os.path.join(output_dir, "top_n_sequence_retention_report.csv")
    df_results.to_csv(report_path, index=False)
    print(f"  - Saved Top N retention report to: {os.path.basename(report_path)}")
    
    # --- 4. Generate Plot ---
    try:
        # Calculate percentages
        df_plot = df_results.melt(
            id_vars=['Antigen', 'Top_N', 'Total_Top_N_Seqs'], 
            var_name='Filter_Step', 
            value_name='Count'
        )
        # Avoid division by zero
        df_plot['Percentage'] = 100 * (df_plot['Count'] / (df_plot['Total_Top_N_Seqs'] + 1e-9))
        
        g = sns.catplot(
            data=df_plot,
            x='Top_N',
            y='Percentage',
            hue='Filter_Step',
            col='Antigen',
            kind='bar',
            col_wrap=4,
            height=4,
            aspect=1.2
        )
        g.fig.suptitle('Percentage of Top N Sequences (from Final Round) Surviving Each Filter', y=1.03)
        g.set_axis_labels("Top N Rank Group", "Survival Rate (%)")
        g.set_titles("{col_name}")
        plot_path = os.path.join(output_dir, "plot_top_n_sequence_retention.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved Top N retention plot to: {os.path.basename(plot_path)}")
        
    except Exception as e:
        print(f"  - WARNING: Could not generate Top N retention plot. Error: {e}")


def main_step7_post_pipeline_insights(run_directory):
    """
    Main post-pipeline analysis orchestrator (Step VII).
    """
    print("\n--- Step VII: Post-Pipeline Insights ---")
    output_dir = os.path.join(run_directory, "post_pipeline_insights_report")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Post-pipeline analysis outputs will be saved in: {output_dir}")

    # --- Load all necessary data from the run ---
    print("\nLoading prerequisite files for post-pipeline analysis...")
    all_data_s1 = load_pickle(os.path.join(run_directory, "step1_all_loaded_data_with_cpms.pkl"), "Step 1 Data")
    enriched_data_s2 = load_pickle(os.path.join(run_directory, "step2_enriched_sequences.pkl"), "Step 2 Data")
    enriched_clusters_s4 = load_pickle(os.path.join(run_directory, "step4_enriched_clusters.pkl"), "Step 4 Data")
    specificity_results_s5 = load_pickle(os.path.join(run_directory, "step5_cross_reactivity_analysis.pkl"), "Step 5 Data")
    
    step3_dir = os.path.join(run_directory, "step3_mmseqs_clustering")
    clusters_tsv_path = os.path.join(step3_dir, "clusters.tsv") if os.path.isdir(step3_dir) else ""
    # The FASTA path from Step 3 needs to point to the correct file name we now use
    unique_seqs_fasta_path = os.path.join(step3_dir, "specific_sequences_for_clustering.fasta") if os.path.isdir(step3_dir) else ""
    
    step6_reports_dir = os.path.join(run_directory, "step6_final_reports_advanced")
    pssm_files_dir = os.path.join(step6_reports_dir, "pssms") if os.path.exists(step6_reports_dir) else ""

    # Check for critical missing data before proceeding
    if all_data_s1 is None:
        print("CRITICAL: Step 1 data is missing. Some post-analyses will be skipped.")
        return

    # --- Run Analysis Functions with Correct Arguments ---
    analyze_top_n_sequence_retention(run_directory, all_data_s1, step6_reports_dir, output_dir)
    # Call to plot_funnel_metrics with all required data
    plot_funnel_metrics(
        output_dir,
        all_data_s1,
        enriched_data_s2,
        unique_seqs_fasta_path,
        clusters_tsv_path,
        enriched_clusters_s4,
        specificity_results_s5,
        step6_reports_dir
    )
    
    # Call to analyze_antigen_similarity with all required data
    analyze_antigen_similarity(
        output_dir,
        all_data_s1,
        ANTIGEN_SIMILARITY_ROUND,
        ANTIGEN_SIMILARITY_JACCARD_CPM_THRESHOLD
    )

    # Call to the new CDR3 length progression analysis
    analyze_cdr3_length_progression(output_dir, all_data_s1)
    
    # Check if Step 6 reports exist before trying to analyze them
    if os.path.exists(step6_reports_dir):
        analyze_cdr_properties(output_dir, all_data_s1, step6_reports_dir)
        
        visualize_top_pssms_as_logos(
            output_dir,
            step6_reports_dir,
            pssm_files_dir,
            TOP_N_CLUSTERS_FOR_LOGO
        )
        
        analyze_cdr3_physicochemical_properties(output_dir, step6_reports_dir)
        analyze_cluster_entropy_distribution(output_dir, step6_reports_dir)
        correlate_cluster_size_with_metrics(output_dir, step6_reports_dir)
        
        analyze_kl_divergence_metrics(output_dir, step6_reports_dir)
        
        analyze_and_plot_pssms(output_dir, all_data_s1, step6_reports_dir)
        analyze_and_plot_cpm_weighted_pssms(output_dir, all_data_s1, step6_reports_dir)
    else:
        print("\nWarning: Step 6 reports directory not found. Skipping detailed candidate analyses in Step VII.")
    
    print("\nStep VII (Post-Pipeline Insights) finished.")


def plot_funnel_metrics(output_dir, all_data_s1, enriched_data_s2, unique_seqs_for_clustering_path, clusters_tsv_path, enriched_clusters_s4, specificity_results_s5, step6_reports_dir):
    """
    Generates funnel plots showing sequence/cluster attrition through pipeline stages.
    """
    if plt is None or sns is None: print("Plotting libraries not available. Skipping funnel metrics."); return
    print("\n--- [VII-A] Generating Funnel Metrics Plot ---")
    
    metrics, antigen_list = [], []
    if enriched_data_s2: antigen_list = sorted(list(enriched_data_s2.keys()))
    elif enriched_clusters_s4: antigen_list = sorted(list(enriched_clusters_s4.keys()))
    elif specificity_results_s5 is not None and not specificity_results_s5.empty: antigen_list = sorted(list(specificity_results_s5['TargetAntigen'].unique()))
    else: print("  Warning: Could not determine antigen list for funnel metrics.")
    for antigen in antigen_list:
        df_r1 = all_data_s1.get((antigen, 1)); metrics.append({'Antigen': antigen, 'Stage': '1_Initial_R1_Seqs', 'Count': len(df_r1) if df_r1 is not None else 0})
        df_s2 = enriched_data_s2.get(antigen); metrics.append({'Antigen': antigen, 'Stage': '2_Post_Seq_Enrich', 'Count': len(df_s2) if df_s2 is not None else 0})
        df_s4 = enriched_clusters_s4.get(antigen); metrics.append({'Antigen': antigen, 'Stage': '4_Post_Cluster_Enrich', 'Count': len(df_s4) if df_s4 is not None else 0})
        if specificity_results_s5 is not None and not specificity_results_s5.empty:
            specific_count = len(specificity_results_s5[(specificity_results_s5['TargetAntigen'] == antigen) & (specificity_results_s5['IsSpecific'] == True)])
            metrics.append({'Antigen': antigen, 'Stage': '5_Specific_Clusters', 'Count': specific_count})
        else: metrics.append({'Antigen': antigen, 'Stage': '5_Specific_Clusters', 'Count': 0})
        report_path = os.path.join(step6_reports_dir, f"final_candidates_{re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen))}.csv")
        final_count = 0
        if os.path.exists(report_path):
            try: final_count = len(pd.read_csv(report_path))
            except pd.errors.EmptyDataError: final_count = 0
        metrics.append({'Antigen': antigen, 'Stage': '6_Final_Candidates', 'Count': final_count})
    metrics_df = pd.DataFrame(metrics)
    # ... (Logic to save the full funnel metrics summary CSV remains unchanged) ...
    
    per_antigen_df = metrics_df.copy()
    plt.figure(figsize=(max(10, len(antigen_list) * 1.5 + 4), 8)); sns.barplot(x='Stage', y='Count', hue='Antigen', data=per_antigen_df, dodge=True)
    plt.title('Pipeline Funnel Metrics by Antigen (Full View, Linear Scale)'); plt.xticks(rotation=45, ha='right'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "funnel_metrics_by_antigen_full_linear.png")); plt.close()
    print("  Saved full linear scale funnel plot.")
    plt.figure(figsize=(max(10, len(antigen_list) * 1.5 + 4), 8)); per_antigen_df['LogCount'] = per_antigen_df['Count'] + 1; sns.barplot(x='Stage', y='LogCount', hue='Antigen', data=per_antigen_df, dodge=True); plt.yscale('log'); plt.ylabel("Count (Log Scale)"); plt.title('Pipeline Funnel Metrics by Antigen (Full View, Log Scale)'); plt.xticks(rotation=45, ha='right'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "funnel_metrics_by_antigen_full_log.png")); plt.close()
    print("  Saved full log scale funnel plot.")
    
    print("  - Generating zoomed-in funnel plots (excluding initial sequences)...")
    zoomed_df = per_antigen_df[per_antigen_df['Stage'] != '1_Initial_R1_Seqs'].copy()
    # Linear Zoomed Plot
    plt.figure(figsize=(max(10, len(antigen_list) * 1.5 + 4), 8)); sns.barplot(x='Stage', y='Count', hue='Antigen', data=zoomed_df, dodge=True)
    plt.title('Pipeline Funnel Metrics by Antigen (Zoomed View, Linear Scale)'); plt.xticks(rotation=45, ha='right'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "funnel_metrics_by_antigen_zoomed_linear.png")); plt.close()
    print("  Saved zoomed linear scale funnel plot.")
    # Log Zoomed Plot
    plt.figure(figsize=(max(10, len(antigen_list) * 1.5 + 4), 8)); zoomed_df['LogCount'] = zoomed_df['Count'] + 1; sns.barplot(x='Stage', y='LogCount', hue='Antigen', data=zoomed_df, dodge=True); plt.yscale('log'); plt.ylabel("Count (Log Scale)"); plt.title('Pipeline Funnel Metrics by Antigen (Zoomed View, Log Scale)'); plt.xticks(rotation=45, ha='right'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "funnel_metrics_by_antigen_zoomed_log.png")); plt.close()
    print("  Saved zoomed log scale funnel plot.")


def analyze_antigen_similarity(output_dir, all_data_s1, analysis_round, jaccard_cpm_threshold):
    """Calculates pairwise Jaccard similarity across antigens using CPM-filtered sequence sets."""
    if plt is None or sns is None: 
        print("Plotting libraries not available. Skipping antigen similarity analysis.")
        return
        
    print(f"\n--- [VII-B] Analyzing Antigen Output Similarity (Jaccard, R{analysis_round}) ---")

    all_antigens = sorted(list(set(k[0] for k in all_data_s1.keys() if k[0] != 'R0_Naive')))
    if len(all_antigens) < 2:
        print("  Not enough antigens with data to compare."); return

    antigen_pairs = list(itertools.combinations(all_antigens, 2))
    
    # --- Step 1: Prep data in parallel (1 task per antigen) ---
    print(f"  Pre-processing significant sequence sets for {len(all_antigens)} antigens in parallel...")
    worker_args = [(ag, analysis_round, jaccard_cpm_threshold) for ag in all_antigens]
    
    sig_sets_dict = {}
    
    with multiprocessing.Pool(
        processes=NUM_PROCESSES,
        initializer=init_worker_s7,
        initargs=(all_data_s1,)
    ) as pool:
        results = pool.map(worker_prep_jaccard_set_s7, worker_args)
    
    for antigen_id, sig_set in results:
        if sig_set:
            sig_sets_dict[antigen_id] = sig_set

    # --- Step 2: Calculate Jaccard Index (FAST) ---
    print(f"  Calculating Jaccard index for {len(antigen_pairs)} pairs...")
    jaccard_matrix = pd.DataFrame(1.0, index=all_antigens, columns=all_antigens)
    
    for ag1, ag2 in antigen_pairs:
        set_ag1 = sig_sets_dict.get(ag1, set())
        set_ag2 = sig_sets_dict.get(ag2, set())
        
        intersection_size = len(set_ag1 & set_ag2)
        union_size = len(set_ag1) + len(set_ag2) - intersection_size
        jaccard_index = intersection_size / union_size if union_size > 0 else 0.0
        
        jaccard_matrix.loc[ag1, ag2] = jaccard_index
        jaccard_matrix.loc[ag2, ag1] = jaccard_index

    # --- Step 3: Save plots and data ---
    def format_annot(x):
        if x == 0.0:
            return "0"
        if x < 0.01:
            return f"{x:.1e}" # Use sci-notation for small numbers
        return f"{x:.2f}" # Use 2 decimals for others
        
    annot_labels = jaccard_matrix.map(format_annot)

    plt.figure(figsize=(14,12))
    sns.heatmap(
        jaccard_matrix, 
        annot=annot_labels,  # Use custom labels
        fmt="",              # Disable default formatting
        cmap="YlGnBu", 
        vmin=0, 
        vmax=1
    )
    plt.title(f'Jaccard Index (R{analysis_round}, CPM > {jaccard_cpm_threshold})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"antigen_similarity_jaccard_R{analysis_round}.png"))
    plt.close()
    
    jaccard_matrix.to_csv(os.path.join(output_dir, f"antigen_similarity_jaccard_matrix_R{analysis_round}.csv"))
    
    print("  Antigen similarity analysis (Jaccard only) complete.")

def analyze_cdr_properties(output_dir, all_data_s1, step6_reports_dir):
    if plt is None or sns is None: print("Plotting libraries not available. Skipping CDR analysis."); return
    print("\n--- [VII-C] Analyzing CDR Properties ---")
    if not (all_data_s1 and os.path.exists(step6_reports_dir)):
        print("  Skipping CDR properties: missing R0 data or Step 6 reports directory.")
        return

    r0_df = all_data_s1.get(('R0_Naive', 0))
    r0_cdr3_lengths = []
    if r0_df is not None and not r0_df.empty:
        for seq in r0_df['Sequence']:
            _, _, _, cdr3_len = parse_cdrs(seq)
            if cdr3_len > 0: r0_cdr3_lengths.append(cdr3_len)
    
    if r0_cdr3_lengths:
        plt.figure(figsize=(10, 6)); sns.histplot(r0_cdr3_lengths, discrete=True, stat="density", common_norm=False, label="R0 Naive"); plt.title('CDR3 Length Distribution in R0'); plt.xlabel('CDR3 Length (AA)'); plt.ylabel('Density'); plt.legend(); plt.savefig(os.path.join(output_dir, "cdr3_length_dist_R0.png")); plt.close()
        print("  Saved R0 CDR3 length distribution plot.")

    antigen_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    if not antigen_files: print("  No Step VI final candidate report files found for CDR analysis."); return

    all_cdr_data = []
    for report_file in antigen_files:
        try:
            df_s6 = pd.read_csv(report_file)
            if df_s6.empty or 'CDR3_Length' not in df_s6.columns: continue
            antigen_name = df_s6['TargetAntigen'].iloc[0]
            all_cdr_data.append(df_s6[['TargetAntigen', 'CDR3_Length']].rename(columns={'TargetAntigen': 'Antigen'}))
        except Exception as e: print(f"  Warning: Could not process CDR properties for {report_file}. Error: {e}")

    if all_cdr_data:
        combined_cdr_df = pd.concat(all_cdr_data).dropna()
        if not combined_cdr_df.empty:
            plt.figure(figsize=(max(8, len(combined_cdr_df['Antigen'].unique()) * 0.8 + 2), 8)); sns.boxplot(x='Antigen', y='CDR3_Length', data=combined_cdr_df); plt.title('CDR3 Length Distribution Across Antigens (Final Candidates)'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "cdr3_length_boxplot_all_antigens.png")); plt.close()
            print("  Saved CDR3 length boxplot for all antigens.")
    print("  CDR property analysis complete.")

def visualize_top_pssms_as_logos(output_dir, step6_reports_dir, pssm_files_dir, top_n_config):
    """
    Generates sequence logos for the top N clusters per antigen.
    """
    if plt is None: print("Plotting libraries not available. Skipping logo visualization."); return
    print("\n--- [VII-D] Visualizing PSSMs as Sequence Logos ---")
    
    logo_output_dir = os.path.join(output_dir, "pssm_logos_top_clusters")
    os.makedirs(logo_output_dir, exist_ok=True)

    final_candidate_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    for report_file in final_candidate_files:
        antigen_name = re.search(r"final_candidates_(.+)\.csv", os.path.basename(report_file)).group(1)
        print(f"  Processing PSSM logos for antigen: {antigen_name}")
        try:
            df_s6 = pd.read_csv(report_file)
            if df_s6.empty: continue
            
            top_candidates = df_s6.sort_values(by='ClusterSize', ascending=False).head(top_n_config)
            for _, row in top_candidates.iterrows():
                rep_seq = row.get('RepSequence')
                pssm_freq_file_rel_path = row.get('PSSM_FreqFile')
                if not rep_seq or pd.isna(pssm_freq_file_rel_path) or pssm_freq_file_rel_path == "N/A": continue
                
                pssm_freq_full_path = os.path.join(step6_reports_dir, pssm_freq_file_rel_path)
                if not os.path.exists(pssm_freq_full_path):
                    print(f"    WARNING: PSSM file not found: {pssm_freq_file_rel_path}")
                    continue
                
                pssm_df = pd.read_csv(pssm_freq_full_path, index_col=0)
                if pssm_df.empty: continue

                title = f"PSSM Logo for Top Cluster - Antigen: {antigen_name}\nRep: {rep_seq[:15]}... (Size: {row.get('ClusterSize','N/A')})"
                safe_rep_name = re.sub(r'[^a-zA-Z0-9]', '_', rep_seq[:15])
                out_path = os.path.join(logo_output_dir, f"logo_{antigen_name}_{safe_rep_name}.png")
                
                generate_sequence_logo_plot(pssm_df, title, out_path)
        except Exception as e:
            print(f"  ERROR processing report file {os.path.basename(report_file)} for logos: {e}")


def analyze_cdr3_physicochemical_properties(output_dir, step6_reports_dir):
    if ProteinAnalysis is None: print("BioPython.ProtParam not available. Skipping physicochemical analysis."); return
    print("\n--- [VII-E] Analyzing CDR3 Physicochemical Properties ---")
    if not os.path.exists(step6_reports_dir): print(f"  Skipping CDR3 physchem: Step 6 reports directory missing."); return
    
    all_physchem_data = []
    antigen_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    for report_file in antigen_files:
        try:
            df_s6 = pd.read_csv(report_file)
            if df_s6.empty or 'CDR3' not in df_s6.columns: continue
            antigen_name = df_s6['TargetAntigen'].iloc[0]
            for cdr3_seq in df_s6['CDR3'].dropna().astype(str):
                if cdr3_seq != "N/A" and len(cdr3_seq) > 0:
                    try:
                        valid_cdr3_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', cdr3_seq.upper())
                        if not valid_cdr3_seq: continue
                        pa = ProteinAnalysis(valid_cdr3_seq)
                        all_physchem_data.append({'Antigen': antigen_name, 'CDR3': cdr3_seq, 'pI': pa.isoelectric_point(), 'Hydrophobicity_KD': pa.gravy()})
                    except Exception: pass
        except Exception as e_report: print(f"  Error processing report {report_file} for physchem: {e_report}")
    
    if not all_physchem_data: print("  No CDR3 data for physicochemical analysis."); return
    physchem_df = pd.DataFrame(all_physchem_data)
    physchem_df.to_csv(os.path.join(output_dir, "cdr3_physicochemical_properties.csv"), index=False)
    print(f"  Saved CDR3 physicochemical properties summary.")
    
    if plt and sns:
        plt.figure(figsize=(max(8,len(physchem_df['Antigen'].unique())*0.8+2),6)); sns.boxplot(x='Antigen',y='pI',data=physchem_df); plt.title('CDR3 Isoelectric Point (pI) by Antigen'); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "cdr3_pI_distribution.png")); plt.close()
        plt.figure(figsize=(max(8,len(physchem_df['Antigen'].unique())*0.8+2),6)); sns.boxplot(x='Antigen',y='Hydrophobicity_KD',data=physchem_df); plt.title('CDR3 Hydrophobicity (Kyte-Doolittle) by Antigen'); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "cdr3_hydrophobicity_distribution.png")); plt.close()
        print(f"  Saved CDR3 physchem plots.")

def analyze_cluster_entropy_distribution(output_dir, step6_reports_dir):
    if plt is None or sns is None: print("Plotting libraries not available. Skipping entropy analysis."); return
    print("\n--- [VII-F] Analyzing Cluster Entropy Distribution ---")
    if not os.path.exists(step6_reports_dir): print(f"  Skipping cluster entropy: Step 6 reports directory missing."); return
    
    all_entropy_data = []
    antigen_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    for report_file in antigen_files:
        try:
            df_s6 = pd.read_csv(report_file)
            if df_s6.empty or 'AvgClusterShannonEntropy' not in df_s6.columns: continue
            antigen_name = df_s6['TargetAntigen'].iloc[0]
            for entropy_val in df_s6['AvgClusterShannonEntropy'].dropna():
                if pd.notna(entropy_val) and isinstance(entropy_val, (int, float)):
                    all_entropy_data.append({'Antigen': antigen_name, 'AvgClusterEntropy': entropy_val})
        except Exception as e_report: print(f"  Error processing report {report_file} for entropy: {e_report}")
    
    if not all_entropy_data: print("  No cluster entropy data found."); return
    entropy_df = pd.DataFrame(all_entropy_data)
    
    plt.figure(figsize=(max(8, len(entropy_df['Antigen'].unique())*0.8 + 2), 6))
    sns.histplot(data=entropy_df, x='AvgClusterEntropy', hue='Antigen', kde=True, element="step")
    plt.title('Distribution of Average Cluster Shannon Entropy (Specific Clusters)'); plt.xlabel('Average Shannon Entropy per Cluster'); plt.ylabel('Number of Clusters'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_entropy_distribution.png")); plt.close()
    print(f"  Saved cluster entropy distribution plot.")

def correlate_cluster_size_with_metrics(output_dir, step6_reports_dir):
    if plt is None or sns is None: print("Plotting libraries not available. Skipping correlation analysis."); return
    print("\n--- [VII-G] Correlating Cluster Size with Other Metrics ---")
    if not os.path.exists(step6_reports_dir): print(f"  Skipping cluster size correlation: Step 6 reports directory missing."); return
    
    cpm_col = f'R{FINAL_ANALYSIS_ROUND}_CPM'

    antigen_files = glob.glob(os.path.join(step6_reports_dir, "final_candidates_*.csv"))
    for report_file in antigen_files:
        match = re.search(r"final_candidates_(.+)\.csv", os.path.basename(report_file))
        if not match: continue
        antigen_name = match.group(1)
        try:
            df_s6 = pd.read_csv(report_file)
            required_cols = ['ClusterSize', cpm_col, 'AvgClusterShannonEntropy']
            if not all(col in df_s6.columns for col in required_cols):
                # Fallback to R3 if R4 wasn't present for this antigen
                if 'R3_CPM' in df_s6.columns:
                    cpm_col = 'R3_CPM'
                    required_cols = ['ClusterSize', cpm_col, 'AvgClusterShannonEntropy']
                else:
                    print(f"  Skipping {antigen_name} for correlation: missing one of {required_cols}."); continue

            df_s6_filtered = df_s6[required_cols].copy().dropna()
            if len(df_s6_filtered) < 2: print(f"  Not enough complete data points for correlation for {antigen_name}."); continue

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=df_s6_filtered, x='ClusterSize', y=cpm_col)
            plt.title(f'{antigen_name}: Size vs. {cpm_col}'); plt.xscale('log'); plt.yscale('log')
            corr_cpm, _ = spearmanr(df_s6_filtered['ClusterSize'], df_s6_filtered[cpm_col])
            plt.text(0.05, 0.95, f'Spearman R: {corr_cpm:.2f}', transform=plt.gca().transAxes, va='top')

            plt.subplot(1, 2, 2)
            sns.scatterplot(data=df_s6_filtered, x='ClusterSize', y='AvgClusterShannonEntropy')
            plt.title(f'{antigen_name}: Size vs. Avg Entropy'); plt.xscale('log')
            corr_ent, _ = spearmanr(df_s6_filtered['ClusterSize'], df_s6_filtered['AvgClusterShannonEntropy'])
            plt.text(0.05, 0.95, f'Spearman R: {corr_ent:.2f}', transform=plt.gca().transAxes, va='top')
            
            plt.tight_layout()
            safe_antigen_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen_name))
            plt.savefig(os.path.join(output_dir, f"correlation_plots_{safe_antigen_name}.png")); plt.close()
        except Exception as e_report: print(f"  Error processing report {report_file} for correlations: {e_report}")
    print("  Correlation analysis plots generated per antigen.")

# ==============================================================================
# === SPECIFICITY DIAGNOSIS (formerly further_analysis.py) ===
# ==============================================================================

def analyze_nonspecific_failure_reasons(specificity_df, output_dir):
    """
    Analyzes non-specific clusters to identify the most common cross-reactivity failures.
    """
    print("\n--- Analyzing Specificity Failure Reasons ---")

    # Filter for non-specific clusters with an actionable cross-reactivity failure reason
    # (skipping data-quality issues like missing counts or FDR errors)
    nonspecific_df = specificity_df[
        (specificity_df['IsSpecific'] == False) &
        (specificity_df['FailureReason'].notna()) &
        (specificity_df['FailureReason'] != 'FDR_Error') &
        (specificity_df['FailureReason'] != 'Not_In_Specificity_Test') &
        (specificity_df['FailureReason'] != 'No_R3_Count_For_Target') &
        (specificity_df['FailureReason'] != 'No_R4_Count_For_Target') &
        (specificity_df['FailureReason'] != 'Zero_Total_R3_Reads') &
        (specificity_df['FailureReason'] != 'Zero_Total_R4_Reads')
    ].copy()
    
    if nonspecific_df.empty:
        print("  - No non-specific clusters with valid failure reasons found. Skipping analysis.")
        return

    # 2. Explode the '+' separated FailureReason column
    # This creates a new row for each antigen in the FailureReason string
    nonspecific_df['FailingAntigen'] = nonspecific_df['FailureReason'].str.split('+')
    exploded_df = nonspecific_df.explode('FailingAntigen')

    # 3. Count total failures per antigen
    failure_counts = exploded_df.groupby('FailingAntigen').size().reset_index(name='Total_Failure_Count')
    
    # 4. Sort to find the worst offenders (the outliers)
    failure_counts_sorted = failure_counts.sort_values(by='Total_Failure_Count', ascending=False)

    # 5. Save the summary report
    report_path = os.path.join(output_dir, "specificity_failure_reason_summary.csv")
    failure_counts_sorted.to_csv(report_path, index=False)
    
    print(f"  - Saved specificity failure summary to: {os.path.basename(report_path)}")
    
    # 6. Report the top offenders to the log (the "outliers")
    print("  - Top 5 Antigens Causing Specificity Failures:")
    print(failure_counts_sorted.head(5).to_string(index=False))
    
    # 7. Break down failures by target antigen to see which cross-reactants are the worst offenders per target
    failures_by_target = exploded_df.groupby(['TargetAntigen', 'FailingAntigen']).size().reset_index(name='Failure_Count')
    failures_by_target_sorted = failures_by_target.sort_values(by=['TargetAntigen', 'Failure_Count'], ascending=[True, False])
    
    report_path_detailed = os.path.join(output_dir, "specificity_failure_reason_by_target.csv")
    failures_by_target_sorted.to_csv(report_path_detailed, index=False)
    print(f"  - Saved detailed per-target failure summary to: {os.path.basename(report_path_detailed)}")

    # Bar plot of top 20 cross-reactive offenders
    if plt is None or sns is None:
        print("  - Plotting libraries not found. Skipping failure reason plot.")
        return
        
    try:
        top_n_offenders = failure_counts_sorted.head(20)
        plt.figure(figsize=(max(10, len(top_n_offenders) * 0.5), 7))
        sns.barplot(data=top_n_offenders, x='FailingAntigen', y='Total_Failure_Count', palette="viridis")
        plt.title('Top 20 Antigens Causing Specificity Failures (Overall)')
        plt.ylabel('Total Times Cited as Failure Reason')
        plt.xlabel('Failing Antigen')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "plot_specificity_failure_outliers.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved failure reason outlier plot to: {os.path.basename(plot_path)}")
    except Exception as e:
        print(f"  - WARNING: Could not generate failure reason plot. Error: {e}")


def main_step8_specificity_diagnosis(run_directory, q_value_cutoff, fc_cutoff):
    print("\n--- Step VIII: Specificity Diagnosis ---")
    output_dir = os.path.join(run_directory, "specificity_diagnosis_report")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Specificity diagnosis outputs will be saved in: {output_dir}")

    enriched_clusters_s4 = load_pickle(os.path.join(run_directory, "step4_enriched_clusters.pkl"), "Step 4 Enriched Clusters")
    df_agg_cluster_counts_s4 = load_pickle(os.path.join(run_directory, "step4_aggregated_cluster_counts.pkl"), "Step 4 Aggregated Counts")
    total_reads_s1 = load_pickle(os.path.join(run_directory, "step1_total_reads_per_sample.pkl"), "Step 1 Total Reads")
    if enriched_clusters_s4 is None or df_agg_cluster_counts_s4 is None or total_reads_s1 is None:
        print("  CRITICAL ERROR: Prerequisite files not loaded. Aborting diagnosis."); return

    print("\n  Re-running specificity analysis for enriched clusters to diagnose failures...")
    all_antigens_list = sorted(list(df_agg_cluster_counts_s4['AntigenID'].unique()))
    
    worker_args = []
    for antigen, df_enriched in enriched_clusters_s4.items():
        if df_enriched is None or df_enriched.empty: continue
        
        if (antigen, 4) in total_reads_s1 and total_reads_s1[(antigen, 4)] > 0:
            analysis_round_for_antigen = 4
        else:
            analysis_round_for_antigen = 3 
            
        excluded_peers = set()
        for group in CROSS_REACTIVITY_EXCLUSION_GROUPS:
            if antigen in group:
                excluded_peers.update(g for g in group if g != antigen)
        other_antigens_to_test = [ag for ag in all_antigens_list if ag != antigen and ag not in excluded_peers]
        
        for rep_seq in df_enriched['RepresentativeSequence']:
            worker_args.append((
                antigen, rep_seq, df_agg_cluster_counts_s4, other_antigens_to_test, 
                total_reads_s1, q_value_cutoff, fc_cutoff, analysis_round_for_antigen
            ))
    
    if not worker_args: print("  No enriched clusters to analyze."); return
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        specificity_results_list = pool.map(worker_rerun_specificity_analysis_s8, worker_args)
    
    specificity_recalculated_df = pd.DataFrame(specificity_results_list, columns=['RepSequence', 'TargetAntigen', 'IsSpecific', 'FailureReason'])
    print("  Specificity analysis re-run complete.")
    analyze_nonspecific_failure_reasons(specificity_recalculated_df, output_dir)
    all_enriched_clusters_list = [df.assign(EnrichedForAntigen=antigen) for antigen, df in enriched_clusters_s4.items() if df is not None and not df.empty]
    all_enriched_df = pd.concat(all_enriched_clusters_list).rename(columns={'RepresentativeSequence': 'RepSequence'})
    merged_df = pd.merge(all_enriched_df, specificity_recalculated_df, how='left', left_on=['RepSequence', 'EnrichedForAntigen'], right_on=['RepSequence', 'TargetAntigen'])

    merged_df['IsSpecific'] = merged_df['IsSpecific'].fillna(False)
    merged_df['FailureReason'] = merged_df['FailureReason'].fillna('Not_In_Specificity_Test')
    merged_df['SpecificityStatus'] = np.where(merged_df['IsSpecific'], 'Specific', 'Non-Specific')
    
    marker_map = {"Specific": "o", "Non-Specific": "X"}
    all_reasons = sorted(merged_df['FailureReason'].unique())
    palette = sns.color_palette("tab20", n_colors=len(all_reasons))
    color_map = {reason: color for reason, color in zip(all_reasons, palette)}
    color_map['Specific'] = '#0072B2'
    
    for antigen in sorted(merged_df['EnrichedForAntigen'].unique()):
        plot_df = merged_df[merged_df['EnrichedForAntigen'] == antigen].copy()
        if plot_df.empty: continue
        
        # Use R4 if available for this antigen, otherwise fall back to R3
        if (antigen, 4) in total_reads_s1 and total_reads_s1[(antigen, 4)] > 0:
            current_round = 4
        else:
            current_round = 3
        r_final_agg_counts = df_agg_cluster_counts_s4[df_agg_cluster_counts_s4['Round'] == current_round]

        max_other_count_list = [r_final_agg_counts[(r_final_agg_counts['RepresentativeSequence'] == row['RepSequence']) & (r_final_agg_counts['AntigenID'] != antigen)]['Count'].max() if not r_final_agg_counts[(r_final_agg_counts['RepresentativeSequence'] == row['RepSequence']) & (r_final_agg_counts['AntigenID'] != antigen)].empty else 0 for _, row in plot_df.iterrows()]
        plot_df['PromiscuityScore_MaxOtherCount'] = max_other_count_list
        plt.figure(figsize=(12, 8))
        
        cpm_col = f'CPM_R{current_round}'
        if cpm_col not in plot_df.columns:
             print(f"  - WARNING: {cpm_col} not in plot_df for {antigen}. Skipping plot.")
             plt.close()
             continue
             
        sns.scatterplot(data=plot_df, x='PromiscuityScore_MaxOtherCount', y=cpm_col, hue='FailureReason', style='SpecificityStatus', s=100, alpha=0.8, palette=color_map, markers=marker_map)
        
        plt.xlabel(f'Promiscuity Score (Max R{current_round} Count in any non-{antigen} antigen)')
        plt.ylabel(f'Enrichment Score ({cpm_col} in {antigen})')

        plt.xscale('log'); plt.yscale('log'); plt.xlim(left=0.5); plt.ylim(bottom=1)
        plt.title(f'Enrichment vs. Promiscuity for {antigen}'); plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Failure Reason')
        safe_antigen_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(antigen))
        plot_path = os.path.join(output_dir, f"enrichment_vs_promiscuity_{safe_antigen_name}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {os.path.basename(plot_path)}")

    print("\nStep VIII (Specificity Diagnosis) finished.")


def worker_rerun_specificity_analysis_s8(args):
    target_antigen, rep_sequence, df_agg_cluster_counts, all_other_antigens, total_reads_s1, q_value_cutoff, fc_cutoff, analysis_round = args
    is_specific = True
    failing_antigens = set()
    
    # Fast map lookup for the specific rep sequence
    subset_df = df_agg_cluster_counts[df_agg_cluster_counts['RepresentativeSequence'] == rep_sequence]
    counts_map = subset_df.set_index(['AntigenID', 'Round'])['Count'].to_dict()
    
    count_target = counts_map.get((target_antigen, analysis_round), 0)
    total_reads_target = total_reads_s1.get((target_antigen, analysis_round), 0)
    
    if total_reads_target == 0: return rep_sequence, target_antigen, False, f'Zero_Total_R{analysis_round}_Reads'
    if count_target == 0: return rep_sequence, target_antigen, False, f'No_R{analysis_round}_Count_For_Target'
    
    prop_target = count_target / total_reads_target
    comparison_details = []
    
    for other_antigen in all_other_antigens:
        if other_antigen == target_antigen: continue
        
        # Fall back to R3 if comparison antigen lacks R4 data
        other_round_to_check = analysis_round
        total_reads_other = total_reads_s1.get((other_antigen, other_round_to_check), 0)
        
        if total_reads_other == 0 and analysis_round == 4:
            other_round_to_check = 3
            total_reads_other = total_reads_s1.get((other_antigen, other_round_to_check), 0)
            
        count_other = counts_map.get((other_antigen, other_round_to_check), 0)
        # ---------------------------
        
        prop_other, fold_change, p_value = 0.0, np.inf, 1.0
        if total_reads_other > 0:
            prop_other = count_other / total_reads_other
            fold_change = prop_target / (prop_other + 1e-9)
            table = [[count_target, int(max(0, total_reads_target - count_target))], 
                     [count_other, int(max(0, total_reads_other - count_other))]]
            try: _, p_value = fisher_exact(table, alternative='greater')
            except ValueError: pass
            
        comparison_details.append({
            'ComparisonAntigen': other_antigen, 'RoundChecked': other_round_to_check, 
            'P_Value': p_value, 'FoldChange': fold_change, 'Prop_Other': prop_other
        })
    
    if not comparison_details:
        failure_reason = 'Specific'
    else:
        p_values = [d['P_Value'] for d in comparison_details]
        try:
            reject, q_values, _, _ = multipletests(p_values, alpha=q_value_cutoff, method='fdr_bh')
            for i, detail in enumerate(comparison_details):
                if not (q_values[i] < q_value_cutoff and detail['FoldChange'] > fc_cutoff):
                    if detail.get('Prop_Other', 0) > 1e-9:
                        is_specific = False
                        # Note the round where it failed in the reason string
                        failing_antigens.add(f"{detail['ComparisonAntigen']}(R{detail['RoundChecked']})")
        except Exception:
            is_specific = False
            failing_antigens.add('FDR_Error')
            
        if not is_specific: failure_reason = "+".join(sorted(list(failing_antigens)))
        else: failure_reason = 'Specific'
        
    return rep_sequence, target_antigen, is_specific, failure_reason


def generate_pipeline_summary_report(run_dir):
    print("\n--- Generating Final Pipeline Summary Report ---")
    summary_data = []

    def safe_read_csv_len(path):
        try: return len(pd.read_csv(path))
        except (FileNotFoundError, pd.errors.EmptyDataError): return 0
    
    def safe_read_csv_unique(path, col):
        try: return pd.read_csv(path)[col].nunique()
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError): return 0

    all_data_s1 = load_pickle(os.path.join(run_dir, "step1_all_loaded_data_with_cpms.pkl"))
    if all_data_s1:
        all_samples_df = pd.concat([df for (ag, rn), df in all_data_s1.items() if ag != 'R0_Naive'], ignore_index=True)
        summary_data.append({'Stage': '1. Initial Unique Sequences', 'Antigen': 'Overall', 'Count': all_samples_df['Sequence'].nunique()})
        del all_samples_df, all_data_s1

    s2_data = load_pickle(os.path.join(run_dir, "step2_enriched_sequences.pkl"))
    if s2_data:
        s2_seqs = set(seq for df in s2_data.values() if df is not None and 'Sequence' in df.columns for seq in df['Sequence'])
        summary_data.append({'Stage': '2. Sequences Passing Enrichment', 'Antigen': 'Overall', 'Count': len(s2_seqs)})
        del s2_data

    s2_5_filtered_path = os.path.join(run_dir, "step2.5_filtered_crossreactive_sequences.csv")
    summary_data.append({'Stage': '2.5. Filtered (Pre-Cluster Cross-Reactive)', 'Antigen': 'Overall', 'Count': safe_read_csv_unique(s2_5_filtered_path, 'Sequence')})

    s3_input_seqs = load_pickle(os.path.join(run_dir, "step2_5_specific_sequences_for_clustering.pkl"))
    summary_data.append({'Stage': '3. Sequences for Clustering', 'Antigen': 'Overall', 'Count': len(s3_input_seqs) if s3_input_seqs else 0})
    del s3_input_seqs

    s3_5_path = os.path.join(run_dir, "step3.5_initial_cluster_composition.csv")
    summary_data.append({'Stage': '3.5. Total Initial Clusters Formed', 'Antigen': 'Overall', 'Count': safe_read_csv_len(s3_5_path)})

    s4_data = load_pickle(os.path.join(run_dir, "step4_enriched_clusters.pkl"))
    if s4_data:
        s4_clusters = set(seq for df in s4_data.values() if df is not None and 'RepresentativeSequence' in df.columns for seq in df['RepresentativeSequence'])
        summary_data.append({'Stage': '4. Enriched Clusters', 'Antigen': 'Overall', 'Count': len(s4_clusters)})
        del s4_data

    s5_filtered_path = os.path.join(run_dir, "step5_filtered_nonspecific_clusters.csv")
    s5_count = safe_read_csv_unique(s5_filtered_path, 'RepresentativeSequence')
    summary_data.append({'Stage': '5. Filtered (Non-Specific Clusters)', 'Antigen': 'Overall', 'Count': s5_count})
    
    final_rep_files = glob.glob(os.path.join(run_dir, "step6_final_reports_advanced", "final_candidates_*.csv"))
    total_final_count = 0
    for f in final_rep_files:
        count = safe_read_csv_len(f)
        antigen_name = os.path.basename(f).replace("final_candidates_", "").replace(".csv", "")
        summary_data.append({'Stage': '6. Final Candidate Clusters', 'Antigen': antigen_name, 'Count': count})
        total_final_count += count
    summary_data.append({'Stage': '6. Final Candidate Clusters', 'Antigen': 'Overall', 'Count': total_final_count})

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(run_dir, "pipeline_summary_statistics.csv"), index=False)
    print("  Pipeline summary statistics report saved.")

# ==============================================================================
# === MAIN EXECUTION BLOCK ===
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nanobody Analysis Pipeline - Post-Analysis (Steps VII & VIII).")
    parser.add_argument("--run_directory", required=True, help="Path to the top-level directory of a completed pipeline run (e.g., '.../run_20251107_120000_id_0.8_cov_0.8')")
    parser.add_argument("--exclude-groups", type=str, default="[]", help="Antigen groups to exclude from cross-reactivity checks (e.g., \"[['A', 'B'], ['C', 'D']]\").")
    parser.add_argument("--fdr", type=float, default=0.05, help="The False Discovery Rate (FDR) q-value cutoff for specificity tests. Default: 0.05.")
    parser.add_argument("--fc", type=float, default=3.0, help="The Fold Change cutoff for specificity tests. Default: 3.0.")
    args = parser.parse_args()

    try:
        CROSS_REACTIVITY_EXCLUSION_GROUPS = [set(g) for g in ast.literal_eval(args.exclude_groups)]
        print(f"INFO: Using cross-reactivity exclusion groups: {CROSS_REACTIVITY_EXCLUSION_GROUPS}")
    except (ValueError, SyntaxError) as e:
        print(f"CRITICAL ERROR: Could not parse --exclude-groups argument. It must be a Python-style list of lists. Error: {e}"); exit(1)

    Q_VALUE_CUTOFF_SPECIFICITY = args.fdr
    FOLD_CHANGE_CUTOFF_SPECIFICITY = args.fc

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

    print("="*80)
    print("STARTING POST-PIPELINE ANALYSIS (Steps VII & VIII)")
    print(f"Analyzing run directory: {args.run_directory}")
    print(f"Using {NUM_PROCESSES} processes.")
    print(f"Settings: FDR={Q_VALUE_CUTOFF_SPECIFICITY}, FC={FOLD_CHANGE_CUTOFF_SPECIFICITY}")
    print("="*80)
    
    start_time_analysis = time.time()

    # --- Step VII: Post-Pipeline Insights ---
    s_time = time.time()
    main_step7_post_pipeline_insights(args.run_directory)
    print(f"Step VII duration: {time.time()-s_time:.2f}s.")

    # --- Step VIII: Specificity Diagnosis ---
    s_time = time.time()
    main_step8_specificity_diagnosis(args.run_directory, Q_VALUE_CUTOFF_SPECIFICITY, FOLD_CHANGE_CUTOFF_SPECIFICITY)
    print(f"Step VIII duration: {time.time()-s_time:.2f}s.")
    
    # --- Final Summary Report ---
    generate_pipeline_summary_report(args.run_directory)

    print(f"\nTotal analysis duration: {time.time() - start_time_analysis:.2f} seconds.")
    print("Post-analysis complete. All outputs saved in sub-folders of the run directory.")
    print("="*80)