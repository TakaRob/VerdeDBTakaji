# functions.py
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import json
import lmfit as lm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Try to import from db_dataverse.py; provide dummy classes if not found
try:
    from db_dataverse import SampleDataverseTable, JVScanDataverseTable
except ImportError:
    print("WARNING: db_dataverse.py not found. Database operations will use dummy classes.")
    class SampleDataverseTable:
        def __init__(self, *args, **kwargs): pass
        def recent_entries(self, num): return [], pd.DataFrame(), None
        def get_entries_by_criteria(self, *args, **kwargs): return [], pd.DataFrame(), None # Add dummy
        def check_existence(self, *args, **kwargs): return False, None, None # Add dummy
    class JVScanDataverseTable:
        def __init__(self, *args, **kwargs): pass
        def recent_entries(self, num): return [], pd.DataFrame(), None
        def get_entries_by_criteria(self, *args, **kwargs): return [], pd.DataFrame(), None # Add dummy
        def insert_data(self, *args, **kwargs): return None # Add dummy
        def calc_and_save_parameters_db(self, *args, **kwargs): return pd.DataFrame() # Add dummy


# --- Database Interaction Functions (Data Fetching Wrappers) ---
def get_sample_data_by_criteria_nb(sample_table_obj, sample_ids=None, date_from=None, date_to=None, limit=None):
    """Fetches sample entries based on criteria using the SampleDataverseTable object."""
    sample_ids_found, sample_values_df, result = sample_table_obj.get_entries_by_criteria(
        sample_ids_list=sample_ids, 
        date_from_str=date_from, 
        date_to_str=date_to,
        num_samples_to_fetch=limit
    )
    return sample_ids_found, sample_values_df, result

def get_jv_data_by_criteria_nb(jv_table_obj, sample_ids=None, date_from=None, date_to=None, limit=None):
    """Fetches JV test entries based on criteria using the JVScanDataverseTable object and adds 'abs_epoch_time'."""
    jv_test_ids, jv_test_values_df, jv_test_result = jv_table_obj.get_entries_by_criteria(
        sample_ids_list=sample_ids,
        date_from_str=date_from,
        date_to_str=date_to,
        num_samples_to_fetch=limit
    )
    
    if jv_test_values_df is not None and not jv_test_values_df.empty:
        # Calculate 'abs_epoch_time'
        if 'abs_epoch_time' not in jv_test_values_df.columns and \
           'base_time' in jv_test_values_df.columns and \
           'elapsed_time' in jv_test_values_df.columns:
            try:
                # Ensure 'base_time' and 'elapsed_time' are numeric, coercing errors
                base_time_numeric = pd.to_numeric(jv_test_values_df['base_time'], errors='coerce')
                elapsed_time_numeric = pd.to_numeric(jv_test_values_df['elapsed_time'], errors='coerce')
                
                # Calculate 'abs_epoch_time'
                # Ensure that the calculation handles potential NaNs from coercion
                jv_test_values_df['abs_epoch_time'] = base_time_numeric + (60.0 * elapsed_time_numeric)
                
                if jv_test_values_df['abs_epoch_time'].isnull().any():
                    print("WARNING (functions.py): Some 'abs_epoch_time' values are NaN after calculation. Check 'base_time' or 'elapsed_time' for non-numeric data or missing values.")
            except Exception as e:
                print(f"WARNING (functions.py): Could not create 'abs_epoch_time' in get_jv_data_by_criteria_nb: {e}")
    else:
        # If df is None or empty, ensure a consistent return structure
        if jv_test_values_df is None:
             jv_test_values_df = pd.DataFrame() # Return empty DataFrame
        if jv_test_ids is None:
            jv_test_ids = []

    return jv_test_ids, jv_test_values_df, jv_test_result

# --- Timestamp Conversion Functions ---
def epoch_to_timestamp(ts):
    '''Convert a unix epoch to a YYYY-MM-DD hh:mm:ss timestamp'''
    try:
        return datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError, OSError): # Added OSError for robustness
        return None

def timestamp_to_epoch(dt_str):
    '''Convert YYYY-MM-DD hh:mm:ss timestamp string to a unix epoch'''
    try:
        datetime_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return datetime_obj.timestamp()
    except (ValueError, TypeError):
        return None

# --- Data Processing Functions ---
def dataframe_clumping(jv_test_data_list_dfs, min_gap_width=1.0, max_clump_width=2.0):
    jv_test_data_list_2 = []
    for i, df_orig in enumerate(jv_test_data_list_dfs):
        if df_orig is None or df_orig.empty:
            jv_test_data_list_2.append(pd.DataFrame())
            continue

        df = df_orig.copy()

        required_cols_for_clumping = ['base_time', 'elapsed_time']
        if not all(col in df.columns for col in required_cols_for_clumping):
            print(f"WARNING (dataframe_clumping): DataFrame for sample index {i} is missing 'base_time' or 'elapsed_time'. Skipping clumping.")
            jv_test_data_list_2.append(df) # Append original (copied) df if clumping can't proceed
            continue
        
        df2 = df.copy() # Work with another copy for modifications

        try:
            # Coerce to numeric, turning errors into NaT/NaN
            base_time_numeric = pd.to_numeric(df['base_time'], errors='coerce')
            elapsed_time_numeric = pd.to_numeric(df['elapsed_time'], errors='coerce')
            
            # Handle potential NaNs before calculation or ensure epoch_to_timestamp can handle them
            tsl = base_time_numeric + 60.0 * elapsed_time_numeric
            
            # If tsl contains NaNs because base_time or elapsed_time were not numeric, handle this
            if tsl.isnull().any():
                 print(f"WARNING (dataframe_clumping): Sample index {i} has non-numeric 'base_time' or 'elapsed_time', affecting 'tsl' and 'time_hrs'.")

        except Exception as e:
            print(f"WARNING (dataframe_clumping): Error creating tsl for clumping for sample index {i}: {e}")
            jv_test_data_list_2.append(df) # Append original (copied) df
            continue
            
        timestamp_series = tsl.apply(epoch_to_timestamp) # This will pass NaNs if tsl has them
        
        df2.insert(0, "measured_on", timestamp_series)
        
        # Ensure min_base_time_val calculation is robust to NaNs
        min_base_time_val = pd.to_numeric(df['base_time'], errors='coerce').dropna().min()

        if pd.isna(min_base_time_val):
            print(f"WARNING (dataframe_clumping): Cannot determine min_base_time for clumping sample index {i}. 'time_hrs' will be NaN.")
            df2.insert(1, "time_hrs", np.nan)
        else:
            df2.insert(1, "time_hrs", (tsl - min_base_time_val) / 3600)

        # Sort and drop rows where 'time_hrs' is NaN (essential for diff operations)
        df2_sorted = df2.sort_values(by='time_hrs').copy()
        df2_sorted.dropna(subset=['time_hrs'], inplace=True)

        time_hrs_list = df2_sorted['time_hrs'].tolist()
        if not time_hrs_list: 
            jv_test_data_list_2.append(df2_sorted) 
            continue

        time_hrs_list_shifted = [time_hrs_list[0]] + time_hrs_list[:-1] # First element will have interval of 0
        time_intervals = [t2 - t1 for t2, t1 in zip(time_hrs_list, time_hrs_list_shifted)]

        clump_assignments = [0] * len(df2_sorted) # Initialize with zeros
        if not df2_sorted.empty: # Should always be true if time_hrs_list is not empty
            clump_assignments[0] = 0 # First point is in clump 0
            clump_number = 0
            clump_elapsed_time = 0.0 # Initialize elapsed time for the current clump
            
            for ind in range(1, len(time_intervals)):
                clump_elapsed_time += time_intervals[ind]
                if time_intervals[ind] >= min_gap_width: # Start of a new clump
                    clump_number += 1
                    clump_elapsed_time = 0.0 # Reset elapsed time for the new clump
                elif clump_elapsed_time > max_clump_width:
                    # This condition means a clump has exceeded max_clump_width without a min_gap_width occurring.
                    # Depending on desired behavior, this might also trigger a new clump or just a warning.
                    # Original notebook prints a warning. Let's keep that.
                    print(f"WARNING (dataframe_clumping): Clump elapsed time {round(clump_elapsed_time,5)} exceeded max_clump_width {max_clump_width} for a sample.")
                clump_assignments[ind] = clump_number
            df2_sorted["clump_number"] = clump_assignments # Use .loc for safer assignment if needed, but direct should work
        
        jv_test_data_list_2.append(df2_sorted)
    return jv_test_data_list_2


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error (read_json_file): Degradation test JSON file not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error (read_json_file): Invalid JSON format in '{file_path}'")
        return None
    except Exception as e:
         print(f"An unexpected error occurred (read_json_file) while reading JSON: {e}")
         return None

# --- Plotting Functions ---
def get_parameter_details(parameter_ylabel_form):
    param_map = {
        'PCE (%)': ("pce", 'PCE (%)'), 'FF (%)': ("ff", 'FF (%)'),
        'Voc (V)': ("voc_v", r'V$_{oc}$ (V)'), 'Jsc (mA/cm^2)': ("jsc_ma", r'J$_{sc}$ (mA/cm$^2$)'),
        'MPP (V)': ("mpp_v", 'MPP (V)'), 'Rser (Ohm-cm^2)': ("rser", r'R$_{ser}$ ($\Omega$-cm$^2$)'),
        'Rsh (Ohm-cm^2)': ("rsh", r'R$_{sh}$ ($\Omega$-cm$^2$)')
    }
    if parameter_ylabel_form in param_map:
        return param_map[parameter_ylabel_form]
    else:
        print(f"Warning (get_parameter_details): Invalid parameter ylabel for plot: {parameter_ylabel_form}. Defaulting to PCE.")
        return "pce", "PCE (%)" # Default to PCE

def plot_time_series_from_form(df_list_processed, sample_ids_selected, parameter_to_plot, 
                               parameter_ylabel, direction_response,
                               stability_tests_df=None, min_base_time_overall=None,
                               y_limits_param=None, plot_title_override=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})

    if not df_list_processed or not any(not df.empty for df in df_list_processed if df is not None):
        print("WARNING (plot_time_series_from_form): No data available for time series plotting.")
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        return fig

    # Single sample detailed plot
    if len(df_list_processed) == 1 and len(sample_ids_selected) == 1:
        sample_id = sample_ids_selected[0]
        df2 = df_list_processed[0]

        if df2.empty or parameter_to_plot not in df2.columns:
            print(f"WARNING (plot_time_series_from_form): No data or parameter '{parameter_to_plot}' not found for sample {sample_id}.")
            ax.text(0.5, 0.5, f"No data for {sample_id}", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Ensure the parameter column is numeric
        df2[parameter_to_plot] = pd.to_numeric(df2[parameter_to_plot], errors='coerce')
        
        # Filter by scan type and drop NaNs for the parameter to plot
        df2_fwd = df2[df2['scan_type'] == 'F'].dropna(subset=[parameter_to_plot])
        df2_rev = df2[df2['scan_type'] == 'R'].dropna(subset=[parameter_to_plot])

        alpha_val = 0.75
        symbols = ['.', 's--', 'o--', '^--', 'v--', '*--', '>--'] # Define outside loop

        if direction_response.lower() == "both":
            alpha_val = 0.5
            if not df2_rev.empty: ax.plot(df2_rev["time_hrs"], df2_rev[parameter_to_plot], 'co', label="Reverse sweep", alpha=alpha_val)
            if not df2_fwd.empty: ax.plot(df2_fwd["time_hrs"], df2_fwd[parameter_to_plot], 'ro', label="Forward sweep", alpha=alpha_val)
        elif direction_response.lower() == "forward":
            if not df2_fwd.empty:
                for cellnum_str in sorted(df2_fwd['cell_number'].astype(str).unique()):
                    try:
                        # cellnum = int(cellnum_str) # For modulo, but ensure it's okay if not purely numeric
                        df2_onecell = df2_fwd[df2_fwd['cell_number'] == cellnum_str]
                        if not df2_onecell.empty: 
                            ax.plot(df2_onecell["time_hrs"], df2_onecell[parameter_to_plot], 
                                    symbols[int(cellnum_str) % len(symbols) if cellnum_str.isdigit() else 0], 
                                    label=f"Fwd cell #{cellnum_str}", alpha=alpha_val)
                    except ValueError: print(f"WARNING (plot_time_series_from_form): Cannot plot Fwd cell '{cellnum_str}'.")
            else: print("INFO (plot_time_series_from_form): No Forward scan data for this sample.")
        elif direction_response.lower() == "reverse":
            if not df2_rev.empty:
                for cellnum_str in sorted(df2_rev['cell_number'].astype(str).unique()):
                    try:
                        # cellnum = int(cellnum_str)
                        df2_onecell = df2_rev[df2_rev['cell_number'] == cellnum_str]
                        if not df2_onecell.empty: 
                            ax.plot(df2_onecell["time_hrs"], df2_onecell[parameter_to_plot], 
                                    symbols[int(cellnum_str) % len(symbols) if cellnum_str.isdigit() else 0], 
                                    label=f"Rev cell #{cellnum_str}", alpha=alpha_val)
                    except ValueError: print(f"WARNING (plot_time_series_from_form): Cannot plot Rev cell '{cellnum_str}'.")
            else: print("INFO (plot_time_series_from_form): No Reverse scan data for this sample.")
        
        current_title = plot_title_override if plot_title_override else sample_id
        ax.set_title(current_title, fontsize=14)
        
        # Add degradation test spans
        if stability_tests_df is not None and not stability_tests_df.empty and min_base_time_overall is not None:
            relevant_tests = stability_tests_df[stability_tests_df['samples'].apply(lambda x: sample_id in x if isinstance(x, list) else False)]
            plotted_deg_labels = set()
            for _, row in relevant_tests.iterrows():
                try:
                    ss_epoch = timestamp_to_epoch(row['start'])
                    ee_epoch = timestamp_to_epoch(row['end'])
                    if ss_epoch is not None and ee_epoch is not None:
                        ss_hrs = (ss_epoch - min_base_time_overall) / 3600
                        ee_hrs = (ee_epoch - min_base_time_overall) / 3600
                        label = f"{row['test']} (Deg Test)"
                        if label not in plotted_deg_labels:
                            ax.axvspan(ss_hrs, ee_hrs, color='yellow', alpha=0.2, label=label)
                            plotted_deg_labels.add(label)
                        else: # Avoid duplicate legend entries
                            ax.axvspan(ss_hrs, ee_hrs, color='yellow', alpha=0.2)
                except Exception as e:
                    print(f"WARNING (plot_time_series_from_form): Could not plot degradation span for test '{row.get('test', 'Unknown')}': {e}")

    # Multiple samples overview plot
    else: 
        try:
            colors_mpl = plt.colormaps.get_cmap('viridis').resampled(len(sample_ids_selected))
        except AttributeError: # older matplotlib
            colors_mpl = plt.cm.get_cmap('viridis', len(sample_ids_selected))

        for idx, (sample_id, df_sample) in enumerate(zip(sample_ids_selected, df_list_processed)):
            if df_sample.empty or parameter_to_plot not in df_sample.columns: continue
            
            df_sample[parameter_to_plot] = pd.to_numeric(df_sample[parameter_to_plot], errors='coerce')
            df_to_plot_multi = df_sample.copy()

            if direction_response.lower() == "forward": 
                df_to_plot_multi = df_sample[df_sample['scan_type'] == 'F']
            elif direction_response.lower() == "reverse": 
                df_to_plot_multi = df_sample[df_sample['scan_type'] == 'R']
            
            df_to_plot_multi = df_to_plot_multi.dropna(subset=[parameter_to_plot])
            if df_to_plot_multi.empty: continue

            # Group by clump number to plot median and Best/Max for each clump
            clump_median_time = df_to_plot_multi.groupby('clump_number')['time_hrs'].median()
            clump_median_param = df_to_plot_multi.groupby('clump_number')[parameter_to_plot].median()
            clump_best_param = df_to_plot_multi.groupby('clump_number')[parameter_to_plot].max() # Best is max
            
            color_val = colors_mpl(idx / len(sample_ids_selected) if len(sample_ids_selected) > 1 else 0.5)
            
            ax.plot(clump_median_time, clump_median_param, linestyle='--', color=color_val, label=f"{sample_id} - Median")
            ax.plot(clump_median_time, clump_best_param, linestyle='-.', color=color_val, label=f"{sample_id} - Best/Max")
        
        current_title = plot_title_override if plot_title_override else f"Comparison: {', '.join(sample_ids_selected)} ({direction_response} scans)"
        ax.set_title(current_title, fontsize=14)

    # Common plot settings
    ax.set_xlabel("Elapsed time (hours)", fontsize=12)
    ax.set_ylabel(parameter_ylabel, fontsize=12)
    
    if y_limits_param and y_limits_param[0] is not None and y_limits_param[1] is not None:
        ax.set_ylim(y_limits_param[0], y_limits_param[1])
    else: # Default y-limits if not provided
        if parameter_to_plot == "pce": ax.set_ylim(0, 25); ax.set_yticks([0,4,8,12,16,20]) # Example ticks
        elif parameter_to_plot == "voc_v": ax.set_ylim(0.5, 1.3)
        elif parameter_to_plot == "jsc_ma": ax.set_ylim(0, 30)
        elif parameter_to_plot == "ff": ax.set_ylim(0, 90)
        elif parameter_to_plot == "rser": ax.set_ylim(0, 50)
        elif parameter_to_plot == "rsh": ax.set_ylim(0, 20000) # Example of a large range
        else: ax.autoscale(enable=True, axis='y') # Default autoscale for other params

    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def make_error_boxes(ax, xdata, xerror, yerror_25, yerror_75, facecolor='r', edgecolor='none', alpha=0.5):
    errorboxes = []
    xe = xerror # Assuming xerror is half-width
    for x, ye25, ye75 in zip(xdata, yerror_25, yerror_75):
        rect = patches.Rectangle((x - xe, ye25), 2.0 * xe, ye75 - ye25, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        errorboxes.append(rect)
    for box in errorboxes: # Add patches to the axes
        ax.add_patch(box)
    # Note: This function modifies 'ax' in place. No return needed unless specifically for the list of patches.

def filter_dataframe_for_boxplot(main_df_with_abs_time, time_interval_epoch, sample_id_to_filter, scan_direction_to_filter):
    if main_df_with_abs_time is None or main_df_with_abs_time.empty:
        return pd.DataFrame() # Return empty DataFrame if input is invalid
    
    df_filtered = main_df_with_abs_time.copy() 

    # Ensure 'abs_epoch_time' exists and is numeric
    if 'abs_epoch_time' in df_filtered.columns:
        df_filtered['abs_epoch_time'] = pd.to_numeric(df_filtered['abs_epoch_time'], errors='coerce')
        df_filtered.dropna(subset=['abs_epoch_time'], inplace=True) # Remove rows where conversion failed
    else:
        print("WARNING (filter_dataframe_for_boxplot): 'abs_epoch_time' column missing.")
        return pd.DataFrame()

    # Filter by time interval
    if time_interval_epoch and len(time_interval_epoch) == 2 and \
       pd.notna(time_interval_epoch[0]) and pd.notna(time_interval_epoch[1]):
        df_filtered = df_filtered[
            (df_filtered['abs_epoch_time'] >= time_interval_epoch[0]) & 
            (df_filtered['abs_epoch_time'] <= time_interval_epoch[1])
        ]
    
    # Filter by sample ID
    df_filtered = df_filtered[df_filtered['sample_id'] == sample_id_to_filter]
    
    # Filter by scan direction
    if scan_direction_to_filter.upper() in ['F', 'R']:
        df_filtered = df_filtered[df_filtered['scan_type'] == scan_direction_to_filter.upper()]
    elif scan_direction_to_filter.lower() != 'both':
        # Default to 'Both' if an invalid scan_direction is provided, or just don't filter by scan_type
        print(f"WARNING (filter_dataframe_for_boxplot): Invalid scan_direction '{scan_direction_to_filter}'. Using data for both directions if available, or not filtering by scan type.")
    
    return df_filtered


def my_boxplot_comparison1(main_df_with_abs_time, time_interval_epoch, list_of_sample_ids, 
                          parameter_to_plot, scan_direction_to_plot, 
                          plot_title, y_axis_label, y_limits, 
                          short_sample_labels=None, box_colors=None):
    data_for_boxplot = []
    legend_stats_list = []

    if not short_sample_labels or len(short_sample_labels) != len(list_of_sample_ids):
        if short_sample_labels:
             print("WARNING (my_boxplot_comparison1): Mismatch in length of short_sample_labels and list_of_sample_ids. Using full IDs.")
        tick_labels_for_plot = list_of_sample_ids
    else:
        tick_labels_for_plot = short_sample_labels

    for idx, sample_id in enumerate(list_of_sample_ids):
        df_filtered = filter_dataframe_for_boxplot(main_df_with_abs_time, 
                                                   time_interval_epoch, 
                                                   sample_id, 
                                                   scan_direction_to_plot)
        parameter_data = [] # Initialize as empty list
        if parameter_to_plot in df_filtered.columns:
            try:
                parameter_data_series = pd.to_numeric(df_filtered[parameter_to_plot], errors='coerce').dropna()
                if not parameter_data_series.empty:
                    parameter_data = list(parameter_data_series)
            except Exception as e:
                print(f"WARNING (my_boxplot_comparison1): Could not convert {parameter_to_plot} for {sample_id}: {e}")
        else:
            print(f"WARNING (my_boxplot_comparison1): Column {parameter_to_plot} not found for sample {sample_id}.")
        
        data_for_boxplot.append(parameter_data) # Append (possibly empty) list
        
        current_tick_label = tick_labels_for_plot[idx]
        scan_dir_label = scan_direction_to_plot if scan_direction_to_plot.upper() in ["F","R"] else "Both"

        if parameter_data: # Only calculate stats if data exists
            stats_str = (f'{current_tick_label} ({scan_dir_label}): '
                         f'Avg={np.average(parameter_data):.2f}, Med={np.median(parameter_data):.2f}, N={len(parameter_data)}')
        else:
            stats_str = f'{current_tick_label} ({scan_dir_label}): No data'
        legend_stats_list.append(stats_str)

    num_samples = len(list_of_sample_ids)
    fig_width = max(8, 2.0 * num_samples) 
    fig, ax = plt.subplots(1, figsize=(fig_width, 7))
    
    params = {'legend.fontsize': 'medium', 'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large', 'xtick.labelsize':'large',
              'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
        
    bplot = None 
    # Filter out empty lists from data_for_boxplot to avoid errors in boxplot and ensure labels match
    bplot_data_to_plot = [d for d in data_for_boxplot if d] 
    bplot_tick_labels = [tick_labels_for_plot[i] for i, d in enumerate(data_for_boxplot) if d]

    if bplot_data_to_plot: # Only plot if there's data
        bplot = ax.boxplot(bplot_data_to_plot, tick_labels=bplot_tick_labels, patch_artist=True, widths=0.6)
        
        colors_to_use_for_plot = []
        actual_plotted_indices = [i for i, d in enumerate(data_for_boxplot) if d] # Indices of samples that actually have data

        if box_colors and len(box_colors) == num_samples: # if custom colors are provided for all original samples
            colors_to_use_for_plot = [box_colors[i] for i in actual_plotted_indices]
        else:
            if box_colors: 
                 print("WARNING (my_boxplot_comparison1): 'box_colors' length mismatch or not provided. Defaulting.")
            try: # Matplotlib 3.x
                colors_cmap = plt.colormaps.get_cmap('Pastel1').resampled(len(bplot_data_to_plot))
            except AttributeError: # older matplotlib
                colors_cmap = plt.cm.get_cmap('Pastel1', len(bplot_data_to_plot))
            colors_to_use_for_plot = [colors_cmap(i % colors_cmap.N) for i in range(len(bplot_data_to_plot))]
        
        legend_handles = []
        # Ensure legend_stats_list is also filtered to match plotted data
        processed_labels_for_legend = [legend_stats_list[i] for i in actual_plotted_indices]
        
        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(colors_to_use_for_plot[i])
            legend_handles.append(patch)
        
        # Jitter plot for individual data points
        box_idx_for_scatter = 0
        for i, data_points in enumerate(data_for_boxplot): # Iterate through original list to maintain sample order for jitter
            if data_points: # Only plot if there is data
                jitter_strength = 0.08 if len(bplot_data_to_plot) > 2 else 0.04
                jitter = np.random.normal(0, jitter_strength, size=len(data_points))
                ax.scatter([box_idx_for_scatter + 1 + j for j in jitter], data_points, alpha=0.5, color='black', s=20)
                box_idx_for_scatter += 1 # Increment only for boxes that are actually plotted
        
        if legend_handles:
             ax.legend(legend_handles, processed_labels_for_legend, title="Legend (Avg, Median, N)", loc='best', fontsize='small')
    else: 
        ax.text(0.5, 0.5, "No data to display for selected criteria.", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # Print summary stats even if no plot, as they might contain "No data" messages
        print("\nSummary Stats (for the selected time interval and scan direction):")
        for stat_line in legend_stats_list: 
            print(stat_line)
    
    ax.set_title(plot_title, fontsize=plt.rcParams.get('axes.titlesize', 'x-large')) # Main title for figure
    plt.ylabel(y_axis_label) 
    
    if y_limits and y_limits[0] is not None and y_limits[1] is not None:
        plt.ylim(y_limits[0], y_limits[1])
    
    # Rotate x-axis labels if they are long or numerous
    if len(bplot_tick_labels) > 4 or any(len(str(label)) > 15 for label in bplot_tick_labels):
         plt.xticks(rotation=30, ha="right")
    else:
         plt.xticks(rotation=0, ha="center")

    plt.grid(True, alpha=0.25, axis='y') 
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    return fig


# Lmfit models (assuming these are used elsewhere or will be, keeping them)
def Constant(x, constant):
    return constant

def Stretch_Exp(x, amp, tau, shift, beta):
    return amp * np.exp(-((x - shift) / tau)**beta)

# Helper class for colors (from notebook, useful for print statements if needed)
class color:
   PURPLE = '\033[95m'; CYAN = '\033[96m'; DARKCYAN = '\033[36m'
   BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
   RED = '\033[91m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
   END = '\033[0m'