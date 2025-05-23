'''
This file contains classes `SampleDataverseTable` and `JVScanDataverseTable`.
    * `SampleDataverseTable` has functions `recent_entries(num_samples)`, 
      `get_entries_by_criteria(...)`, and `check_existence(sample_id)`.
    * `JVScanDataverseTable` has functions `recent_entries(num_samples)`,
      `get_entries_by_criteria(...)`, and 
      `insert_data(row, image_path, image_column_name, sample_lookup_id, sample_column_name)`.
    
    RLH 051224. Updated with get_entries_by_criteria methods.
'''
import os
import subprocess
import pandas as pd
import re
import sys

#######################################
#####  class SampleDataverseTable #####
#######################################
class SampleDataverseTable:
    def __init__(self, crm_url, table_name, col_logical_names=None):
        self.search_path = os.getcwd() + "/scripts/recent-entries_V2.ps1" #PowerShell script to list recent sample_id's.
        self.check_path = os.getcwd() + "/scripts/check-existence.ps1"    #PowerShell script to check existence of a specific sample_id.
        self.crm_url = crm_url
        self.table_name = table_name

        if col_logical_names is None:
            self.columnheadings = [
                "Sample ID", "Operator", "Perovskite Composition", "HTL Material", "ETL Material",
                "Top Capping Material", "Bottom Capping Material", "Bulk Passivation Materials", "Is Encapsulated"
            ]
            self.colstrs = [ # Logical names for the columns
                "cr69a_sample_id", "cr69a_operator", "cr69a_perovskite_composition", "cr69a_htl_material",
                "cr69a_etl_material", "cr69a_top_capping_material", "cr69a_bottom_capping_material",
                "cr69a_bulk_passivation_materials", "cr69a_is_encapsulated"
            ]
        else:
            self.columnheadings = list(col_logical_names.keys())
            self.colstrs = list(col_logical_names.values())  

        self.checkheadings = ["Sample_ID", "Sample_data_ID" , "Operator_name","Created_on","Cell_active_area"]
        self.sample_data_heading = "Sample_data_ID"
        self.checkstrs = ["crf3d_sampleid", "crf3d_sample_dataid", "crf3d_operatorname", "createdon", "crf3d_cellactiveareacm2"]
   
    def recent_entries(self, num_samples):
        """
        Retrieve 'num_samples' records of sample data, most recent first.
        """
        if sys.platform.startswith('win32'):
            querystring = '?$select=' + ','.join(self.colstrs)
            orderbystring = '&$orderby=createdon desc'
        else:
            querystring = '?\\$select=' + ','.join(self.colstrs)
            orderbystring = '&"\\$orderby"=createdon desc'
        querystring += orderbystring
       
        # The PowerShell script recent-entries_V2.ps1 uses its 3rd positional argument for -Top
        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}" "{querystring}" {num_samples} "{self.crm_url}" -cols "{",".join(self.colstrs)}"'
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script (recent_entries).")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None, None, result

        if len(self.columnheadings) != len(self.colstrs):
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.colstrs)} values.')
            return None, None, result
    
        result_lines = result.stdout.strip().splitlines()
        if not result_lines or (len(result_lines) == 1 and not result_lines[0].strip()):
            return [], pd.DataFrame(columns=self.columnheadings), result
            
        parsed_lines = []
        for i, string in enumerate(result_lines):
            if string.strip():
                parsed_lines.append(re.split(r',\s', string))
        
        if not parsed_lines:
             return [], pd.DataFrame(columns=self.columnheadings), result

        recent_values = pd.DataFrame(parsed_lines, columns=self.columnheadings)
        sample_ids = recent_values['Sample ID'].tolist() if 'Sample ID' in recent_values else []
    
        return sample_ids, recent_values, result

    def get_entries_by_criteria(self, sample_ids_list=None, date_from_str=None, date_to_str=None, num_samples_to_fetch=None):
        """
        Retrieve sample data records based on specified criteria.
        - sample_ids_list: A list of sample IDs (crf3d_sampleid) to filter by.
        - date_from_str: Start date string in 'YYYY-MM-DD' format.
        - date_to_str: End date string in 'YYYY-MM-DD' format.
        - num_samples_to_fetch: Maximum number of records to return (OData $top).
        """
        filter_conditions = []

        if sample_ids_list:
            if isinstance(sample_ids_list, str): # Single ID
                sample_ids_list = [sample_ids_list]
            if sample_ids_list: # If list is not empty after ensuring it's a list
                sample_filter_parts = [f"crf3d_sampleid eq '{sample_id}'" for sample_id in sample_ids_list]
                if len(sample_filter_parts) == 1:
                    filter_conditions.append(sample_filter_parts[0])
                else:
                    filter_conditions.append(f"({ ' or '.join(sample_filter_parts) })")

        if date_from_str:
            filter_conditions.append(f"createdon ge {date_from_str}T00:00:00Z")
        
        if date_to_str:
            filter_conditions.append(f"createdon le {date_to_str}T23:59:59Z")

        select_clause = ','.join(self.colstrs)
        
        param_prefix = "\\" if not sys.platform.startswith('win32') else "" # For PowerShell escaping on non-Windows

        querystring = f'?{param_prefix}$select={select_clause}'
        if filter_conditions:
            querystring += f"&{param_prefix}$filter={' and '.join(filter_conditions)}"
        
        # Order by 'createdon desc'
        orderby_clause = 'createdon desc'
        if sys.platform.startswith('win32'):
            querystring += f'&$orderby={orderby_clause}'
        else: # MacOS/Linux
            querystring += f'&"{param_prefix}$orderby"={orderby_clause}'


        if num_samples_to_fetch is not None and num_samples_to_fetch > 0:
            querystring += f'&{param_prefix}$top={num_samples_to_fetch}'
            ps_num_samples_arg = num_samples_to_fetch # Pass to PS script for -Top if script uses it
        else:
            ps_num_samples_arg = 0 # Or a high number if your script expects a value for -Top
                                   # and you prefer to not use OData $top for some reason.
                                   # If recent-entries_V2.ps1 is modified to prefer OData $top, this becomes less critical.

        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}" "{querystring}" {ps_num_samples_arg} "{self.crm_url}" -cols "{",".join(self.colstrs)}"'
        
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script (get_entries_by_criteria for SampleData).")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None, None, result

        if len(self.columnheadings) != len(self.colstrs):
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.colstrs)} values.')
            return None, None, result
    
        result_lines = result.stdout.strip().splitlines()
        if not result_lines or (len(result_lines) == 1 and not result_lines[0].strip()):
            print("No sample entries found matching the criteria.")
            return [], pd.DataFrame(columns=self.columnheadings), result

        parsed_lines = []
        for i, string in enumerate(result_lines):
            if string.strip(): 
                parsed_lines.append(re.split(r',\s', string))
            
        if not parsed_lines:
             print("No valid sample entries found after parsing.")
             return [], pd.DataFrame(columns=self.columnheadings), result

        filtered_values = pd.DataFrame(parsed_lines, columns=self.columnheadings)
        sample_ids = filtered_values['Sample ID'].tolist() if 'Sample ID' in filtered_values else []
    
        return sample_ids, filtered_values, result

    def check_existence(self, sample_id, require_confirmation=True):
        """
        Return True if 'sample_id' is in the table.
        """
        if sys.platform.startswith('win32'):
            querystring = '?$select=' + ','.join(self.checkstrs)
        else:
            querystring = '?\\$select=' + ','.join(self.checkstrs)

        allow_entry = False
        confirmation = ''
        sample_data_id = None
        
        while not allow_entry:
            cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.check_path}" "{sample_id}" "{self.table_name}" "{querystring}" "{self.crm_url}" -cols "{",".join(self.checkstrs)}" -headings "{",".join(self.checkheadings)}"'
            result = subprocess.run(cli1, shell=True, capture_output=True, text=True)
            print("\n", result.stdout.strip(), "\n")

            if "invalid" in result.stdout.lower(): # This check might need to be more robust based on actual PS script output
                print(f"Sample ID '{sample_id}' was not found in the table '{self.table_name}', please enter a new sample ID.")
                sample_id = input(str("\nEnter the ID of the sample to be measured : "))
            else:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if self.sample_data_heading in line:
                        try:
                            sample_data_id_part = line.split(",")[1].strip()
                            sample_data_id = sample_data_id_part.split(":")[1].strip()
                        except IndexError:
                            print(f"Warning: Could not parse {self.sample_data_heading} from line: {line}")
                            sample_data_id = None # Ensure it's None if parsing fails
                        break
                
                if(require_confirmation):
                    confirmation = input(str("Please confirm this is the sample to be measured (Y/N): "))
                    if confirmation.upper() == "Y":
                        print(f"\nConfirmed the Sample ID: '{sample_id}'.")
                        allow_entry = True
                    else:
                        sample_id = input(str("\nEnter the sample id you wish to add data for: "))
                else:
                    allow_entry = True

        return allow_entry, sample_data_id, result
    
#######################################
#####  class JVScanDataverseTable #####
#######################################
class JVScanDataverseTable:
    def __init__(self, crm_url, table_name, col_logical_names=None, image_column_name=None):
        self.search_path = os.getcwd() + "/scripts/recent-entries_V2.ps1"
        self.crm_url = crm_url
        self.table_name = table_name
        self.image_column_name = image_column_name
        self.col_logical_names = col_logical_names
        if col_logical_names is None:
            cols = ('cr69a_sample_id', 'cr69a_elapsed_time_min', 'cr69a_base_time_sec', 'cr69a_test_id', 'cr69a_iph_suns', 'cr69a_voc_v',
                    'cr69a_mpp_v', 'cr69a_jsc_macm2', 'cr69a_rsh', 'cr69a_rser', 'cr69a_ff_', 'cr69a_pce_',
                    'cr69a_operator_name', 'cr69a_scan_type', 'cr69a_location', 'cr69a_cell_number', 'cr69a_module',
                    'cr69a_masked', 'cr69a_mask_area_cm2', 'cr69a_temperature_c', 'cr69a_humidity_pct', 'cr69a_wire_mode',
                    'cr69a_path')
            self.cols_unjoined = cols 
            self.cols = ','.join(cols) 
            self.columnheadings = list(cols) # Assuming logical names are used as headings if not provided
        else:
            self.cols_unjoined = list(col_logical_names.values())
            self.cols = ','.join(tuple(col_logical_names.values()))
            self.columnheadings = list(col_logical_names.keys())

    def insert_data(self, row, image_path=None, sample_lookup_id=None, sample_lookup_name=None):
        cli = (f'pwsh -ExecutionPolicy Bypass -File "scripts/insert-data_v4.ps1" '
               f'-crm_url "{self.crm_url}" '
               f'-table_name "{self.table_name}" '
               f'-cols "{self.cols}" '
               f'-sample_id "{row["sample_id"]}" '
               f'-elapsed_time "{row["elapsed_time"]}" '
               f'-base_time "{row["base_time"]}" '
               f'-test_id "{row["test_id"]}" '
               f'-i_ph_suns "{row["i_ph_suns"]}" '
               f'-voc_v "{row["voc_v"]}" '
               f'-mpp_v "{row["mpp_v"]}" '
               f'-jsc_ma "{row["jsc_ma"]}" '
               f'-rsh "{row["rsh"]}" '
               f'-rser "{row["rser"]}" '
               f'-ff "{row["ff"]}" '
               f'-pce "{row["pce"]}" '
               f'-operator "{row["operator"]}" '
               f'-scan_type "{row["scan_type"]}" '
               f'-lab_location "{row["lab_location"]}" '
               f'-cell_number "{row["cell_number"]}" '
               f'-module "{row["module"]}" '
               f'-masked "{row["masked"]}" '
               f'-mask_area "{row["mask_area"]}" '
               f'-temp_c "{row["temp_c"]}" '
               f'-hum_pct "{row["hum_pct"]}" '
               f'-four_wire_mode "{row["four_wire_mode"]}" '
               f'-scan_data_path "{row["scan_data_path"]}"')
        
        if image_path and self.image_column_name:
            cli += f' -image_path "{image_path}" -image_column_name "{self.image_column_name}"'

        if sample_lookup_id and sample_lookup_name:
            print("Warning: inserting lookup data is not currently functioning.")
            cli += f' -sampleDataRecordId "{sample_lookup_id}" -sampleDataLookupColumn "{sample_lookup_name}"'

        print(f'four_wire_mode = {row["four_wire_mode"]}')
        
        result =  subprocess.run(cli, shell=True)
        return result

    def calc_and_save_parameters_db(self, df, form_responses_dictionary, keithley, image_data_path, scan_data_path, 
                            path = r'C:\\Users\\Public', temp_c=0, hum_pct=0, four_wire=False, 
                            samplename = "testymctest", cellnumber = "0", scantype="R", timenowstr = "", 
                            datetodaystr = "", saveparameters = "yes", verbose = 1, timeseries = False,
                            base_t = 0, I_ph = 1.0, mpp_tracking_mode="False"):
        
        par_df_new = keithley.calc_and_save_parameters(df,  path, samplename, cellnumber, scantype, timenowstr, 
                datetodaystr, saveparameters, verbose, timeseries, base_t, I_ph, mpp_tracking_mode)

        row = {
            'sample_id':     f'{str(par_df_new["Sample_Name"].iloc[0])}',
            'elapsed_time':  f'{str(par_df_new["Elapsed_Time"].iloc[0])}',
            'base_time':     f'{base_t}',
            'test_id':       f'{str(par_df_new["Test_ID"].iloc[0])}',
            'i_ph_suns':     f'{str(par_df_new["I_ph(suns)"].iloc[0])}',
            'voc_v':         f'{str(par_df_new["V_oc(V)"].iloc[0])}',
            'mpp_v':         f'{str(par_df_new["mpp(V)"].iloc[0])}',
            'jsc_ma':        f'{str(par_df_new["J_sc(mA)"].iloc[0])}',
            'rsh':           f'{str(par_df_new["R_sh(Ω-cm^2)"].iloc[0])}',
            'rser':          f'{str(par_df_new["R_ser(Ω-cm^2)"].iloc[0])}',
            'ff':            f'{str(par_df_new["FF(%)"].iloc[0])}',
            'pce':           f'{str(par_df_new["PCE(%)"].iloc[0])}',
            'operator':      f'{form_responses_dictionary["Operator"]}',
            'scan_type':     f'{str(par_df_new["Scan_Type"].iloc[0])}',
            'lab_location':  f'{form_responses_dictionary["Where were the measurements done?"]}',
            'cell_number':   f'{form_responses_dictionary["Cell number?"]}',
            'module':        f'{form_responses_dictionary["Is this a module measurement?"]}',
            'masked':        f'{form_responses_dictionary["Is the sample masked?"]}',
            'mask_area':     f'{form_responses_dictionary["Mask area (cm^2)?"]}',
            'temp_c': temp_c,
            'hum_pct': hum_pct,
            'four_wire_mode': four_wire,
            'scan_data_path': scan_data_path
        }

        # Ensure row keys match the logical names expected by col_logical_names
        # This assumes self.col_logical_names is a dict where keys are user-friendly and values are DB logical names
        # For insert_data, the 'row' dictionary should ideally use keys that directly match what insert-data_v4.ps1 expects
        # or be transformed. The current script passes values positionally based on 'self.cols'.
        # For clarity, the 'row' dictionary keys here should ideally be the user-friendly ones used in self.columnheadings
        # and then mapped to the logical names in self.cols_unjoined when constructing the PowerShell command string.
        # However, the original insert_data_v4.ps1 call in the provided code relies on specific parameter names
        # like -sample_id, -elapsed_time. The check below is a bit mismatched with how insert_data is called.
        # For now, I'll keep it as is, reflecting the original structure, but this is an area for future refinement.
        
        # if self.col_logical_names: # Only check if col_logical_names was provided
        #     if not all(key in self.col_logical_names for key in row): # Check if all row keys are in col_logical_names
        #         print("Error: 'row' keys and 'col_logical_names' keys don't align properly for insertion.")
        #         # Find differing keys for debugging:
        #         # print(f"Row keys: {set(row.keys())}")
        #         # print(f"Col_logical_names keys: {set(self.col_logical_names.keys())}")
        #         return None # Indicate error

        returnval = self.insert_data(row, image_path=image_data_path)
        print(f'Inserting data into table "{self.table_name}".' )
            
        return par_df_new
    
    def recent_entries(self, num_samples):
        """
        Retrieve 'num_samples' records of JV scan data, most recent first.
        """
        if sys.platform.startswith('win32'):
            querystring = '?$select=' + self.cols
            orderbystring = '&$orderby=createdon desc'
        else:
            querystring = '?\\$select=' + self.cols
            orderbystring = '&"\\$orderby"=createdon desc'
        querystring += orderbystring

        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}" "{querystring}" {num_samples} "{self.crm_url}" -cols "{self.cols}"'
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script (recent_entries for JVScan).")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None, None, result

        if len(self.columnheadings) != len(self.cols_unjoined):
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.cols_unjoined)} values.')
            return None, None, result

        result_lines = result.stdout.strip().splitlines()
        if not result_lines or (len(result_lines) == 1 and not result_lines[0].strip()):
            return [], pd.DataFrame(columns=self.columnheadings), result
            
        parsed_lines = []
        for i, string in enumerate(result_lines):
            if string.strip():
                 parsed_lines.append(re.split(r',\s', string))
        
        if not parsed_lines:
            return [], pd.DataFrame(columns=self.columnheadings), result
            
        recent_values = pd.DataFrame(parsed_lines, columns=self.columnheadings)
        jv_test_ids = recent_values['sample_id'].tolist() if 'sample_id' in recent_values else []

        return jv_test_ids, recent_values, result

    def get_entries_by_criteria(self, sample_ids_list=None, date_from_str=None, date_to_str=None, num_samples_to_fetch=None):
        """
        Retrieve JV scan data records based on specified criteria.
        - sample_ids_list: A list of sample IDs (crf3d_sample_id) to filter by.
        - date_from_str: Start date string in 'YYYY-MM-DD' format.
        - date_to_str: End date string in 'YYYY-MM-DD' format.
        - num_samples_to_fetch: Maximum number of records to return (OData $top).
        """
        filter_conditions = []

        if sample_ids_list:
            if isinstance(sample_ids_list, str): # Single ID
                sample_ids_list = [sample_ids_list]
            if sample_ids_list:
                # Ensure the field name crf3d_sample_id is correct for JVScanDataverseTable
                # The notebook uses 'sample_id': 'crf3d_sample_id' in col_logical_names
                sample_id_field_logical_name = self.col_logical_names.get('sample_id', 'crf3d_sample_id') # Default if not in map
                sample_filter_parts = [f"{sample_id_field_logical_name} eq '{sample_id}'" for sample_id in sample_ids_list]
                if len(sample_filter_parts) == 1:
                    filter_conditions.append(sample_filter_parts[0])
                else:
                    filter_conditions.append(f"({ ' or '.join(sample_filter_parts) })")
        
        if date_from_str:
            filter_conditions.append(f"createdon ge {date_from_str}T00:00:00Z")
        
        if date_to_str:
            filter_conditions.append(f"createdon le {date_to_str}T23:59:59Z")

        select_clause = self.cols # Already a comma-separated string of logical names
        
        param_prefix = "\\" if not sys.platform.startswith('win32') else ""

        querystring = f'?{param_prefix}$select={select_clause}'
        if filter_conditions:
            querystring += f"&{param_prefix}$filter={' and '.join(filter_conditions)}"
        
        orderby_clause = 'createdon desc'
        if sys.platform.startswith('win32'):
            querystring += f'&$orderby={orderby_clause}'
        else: # MacOS/Linux
            querystring += f'&"{param_prefix}$orderby"={orderby_clause}'
            
        if num_samples_to_fetch is not None and num_samples_to_fetch > 0:
            querystring += f'&{param_prefix}$top={num_samples_to_fetch}'
            ps_num_samples_arg = num_samples_to_fetch 
        else:
            ps_num_samples_arg = 0 # Or adjust if recent-entries_V2.ps1 has a different expectation for 'all'

        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}" "{querystring}" {ps_num_samples_arg} "{self.crm_url}" -cols "{self.cols}"'
        
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script (get_entries_by_criteria for JVScan).")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None, None, result

        if len(self.columnheadings) != len(self.cols_unjoined): # Ensure this check is valid
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.cols_unjoined)} values.')
            return None, None, result
            
        result_lines = result.stdout.strip().splitlines()
        if not result_lines or (len(result_lines) == 1 and not result_lines[0].strip()):
            print("No JV entries found matching the criteria.")
            return [], pd.DataFrame(columns=self.columnheadings), result

        parsed_lines = []
        for i, string in enumerate(result_lines):
            if string.strip():
                 parsed_lines.append(re.split(r',\s', string))

        if not parsed_lines:
            print("No valid JV entries found after parsing.")
            return [], pd.DataFrame(columns=self.columnheadings), result
            
        filtered_values = pd.DataFrame(parsed_lines, columns=self.columnheadings)
        # The key for sample ID in self.columnheadings is 'sample_id'
        jv_test_ids = filtered_values['sample_id'].tolist() if 'sample_id' in filtered_values else []
            
        return jv_test_ids, filtered_values, result