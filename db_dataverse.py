'''
This file contains classes `SampleDataverseTable` and `JVScanDataverseTable`.
    *`SampleDataverseTable` has functions `recent_entries(num_samples)` 
        and check_existence(sample_id).
    * `JVScanDataverseTable` has function 
        `insert_data(row, image_path, image_column_name, sample_lookup_id, sample_column_name)`.
    
    RLH 051224.
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

        # the following two lists are only used by the recent_entries function.
        if col_logical_names is None:
            # Default values are for the table Sample_data_V2 in Randall Headrick's environment on UVM.
            self.columnheadings = [
                "Sample ID", "Operator", "Perovskite Composition", "HTL Material", "ETL Material",
                "Top Capping Material", "Bottom Capping Material", "Bulk Passivation Materials", "Is Encapsulated"
            ]
            self.colstrs = [
                "cr69a_sample_id", "cr69a_operator", "cr69a_perovskite_composition", "cr69a_htl_material",
                "cr69a_etl_material", "cr69a_top_capping_material", "cr69a_bottom_capping_material",
                "cr69a_bulk_passivation_materials", "cr69a_is_encapsulated"
            ]
        else:
            #  'col_logical_names' is assumed to be a dictionary.
            self.columnheadings = list(col_logical_names.keys())  # Convert to a list of strings.
            self.colstrs = list(col_logical_names.values())  


        # # The following lists are only used by the check_existence function. These are for the table on UVM.
        # self.checkheadings = ["Sample ID", "Sample_data_V2_ID" , "Operator","Entry Date","Cell_active_area"]
        # self.sample_data_heading = "Sample_data_V2_ID"  # This column contains the GUID.
        # self.checkstrs = ["cr69a_sample_id", "cr69a_sampledatav2id", "cr69a_operator", "cr69a_entry_date_and_time", "cr69a_cell_active_area"]

        # The following lists are only used by the check_existence function.
        self.checkheadings = ["Sample_ID", "Sample_data_ID" , "Operator_name","Created_on","Cell_active_area"]
        self.sample_data_heading = "Sample_data_ID"  # This column contains the GUID.
        self.checkstrs = ["crf3d_sampleid", "crf3d_sample_dataid", "crf3d_operatorname", "createdon", "crf3d_cellactiveareacm2"]
   
    def recent_entries(self, num_samples):
        """
        Retrieve 'num_samples' records of sample data, most recent first.
        """
 
        if sys.platform.startswith('win32'):
            querystring = '?$select=' + ','.join(self.colstrs)                 # These work on Windows.
            orderbystring = '&"$orderby"=' + 'createdon desc'                  # 'cr69a_entry_date_and_time desc' <--forUVM.
        else:
            querystring = '?\$select=' + ','.join(self.colstrs)                # These work on MacOS.
            orderbystring = '&"\$orderby"=' + 'createdon desc'                 # 'cr69a_entry_date_and_time desc' <--forUVM.
        querystring += orderbystring
       
        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}"\
                "{querystring}" {num_samples} "{self.crm_url}" -cols "{",".join(self.colstrs)}"'
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        # Check for some common errors.
        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script.")
            return None, None, result

        if len(self.columnheadings) != len(self.colstrs):
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.colstrs)} values.')
            return None, None, result
    
        result_lines = result.stdout.strip().splitlines()
        for i, string in enumerate(result_lines):
            result_lines[i] = re.split(r',\s', string)
    
        recent_values = pd.DataFrame(result_lines, columns=self.columnheadings)
        sample_ids = recent_values['Sample ID'].tolist()
    
        return sample_ids, recent_values, result


    def check_existence(self, sample_id, require_confirmation=True):
        """
        Return True if 'sample_id' is in the table.
        """
        if sys.platform.startswith('win32'):
            querystring = '?$select=' + ','.join(self.checkstrs)   # This works on Windows.
        else:
            querystring = '?\$select=' + ','.join(self.checkstrs)  # This works on MacOS.

        allow_entry = False
        confirmation = ''
        sample_data_id = None # Primary id column of the table, typically a code like '1afd05c4-eef9-ee11-a1fe-002248243549'.
        
        while not allow_entry:
            cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.check_path}" "{sample_id}" "{self.table_name}" "{querystring}"\
            "{self.crm_url}" -cols "{",".join(self.checkstrs)}" -headings "{",".join(self.checkheadings)}"'
            result = subprocess.run(cli1, shell=True, capture_output=True, text=True)
            print("\n", result.stdout.strip(), "\n")
            # Check if entry was not found
            if "invalid" in result.stdout.lower():
                print(f"Sample ID '{sample_id}' was not found in the table '{self.table_name}', please enter a new sample ID.")
                sample_id = input(str("\nEnter the ID of the sample to be measured : "))
            else:
                # Parse the result to extract the Sample_data_ID (this is the primary id column of the table).
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if self.sample_data_heading in line:
                        sample_data_id = line.split(",")[1].strip()             # take the part after the first comma.
                        sample_data_id = sample_data_id.split(":")[1].strip()   # then take the part after the first colon.
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
        """
        Set up environment-specific parameters for the JVScanDataverseTable instance.
        """
        self.search_path = os.getcwd() + "/scripts/recent-entries_V2.ps1" #PowerShell script to list recent database entries.
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
            self.cols_unjoined = cols   # Kludge to make it work.
            self.cols = ','.join(cols)  # Convert cols list to a comma-separated string.
        else:
            #  'col_logical_names' is assumed to be a dictionary.
            self.cols_unjoined = list(col_logical_names.values())    # Kludge to make it work.
            self.cols = ','.join(tuple(col_logical_names.values()))  # Convert to a comma-separated string.
            self.columnheadings = list(col_logical_names.keys())     # Convert to a list of strings.

    def insert_data(self, row, image_path=None, sample_lookup_id=None, sample_lookup_name=None):
        # Constructing PowerShell command
        cli = (f'pwsh -ExecutionPolicy Bypass -File "scripts/insert-data_v4.ps1" '
               f'-crm_url "{self.crm_url}" '
               f'-table_name "{self.table_name}" '
               f'-cols "{self.cols}" '
               f'-sample_id "{row["sample_id"]}" '
               f'-elapsed_time "{row["elapsed_time"]}" '
               f'-base_time "{row["base_time"]}" '  # New base_time parameter.
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
        
        # Add image parameters if provided.
        if image_path and self.image_column_name:
            cli += f' -image_path "{image_path}" -image_column_name "{self.image_column_name}"'

        # Add Lookup parameters if provided.
        if sample_lookup_id and sample_lookup_name:
            print("Warning:  inserting lookup data is not currrently functioning.")
            cli += f' -sampleDataRecordId "{sample_lookup_id}" -sampleDataLookupColumn "{sample_lookup_name}"'

        print(f'four_wire_mode = {row["four_wire_mode"]}')
        

        # Execute the PowerShell command.
        result =  subprocess.run(cli, shell=True)
        return result

    def calc_and_save_parameters_db(self, df, form_responses_dictionary,  keithley, image_data_path, scan_data_path, 
                            path = r'C:\\Users\Public', temp_c=0, hum_pct=0, four_wire=False, 
                            samplename = "testymctest", cellnumber = "0", scantype="R", timenowstr = "", 
                            datetodaystr = "", saveparameters = "yes", verbose = 1, timeseries = False,
                            base_t = 0, I_ph = 1.0, mpp_tracking_mode="False"):
        """
        Calculate PV parameters and save to a file. Append if the file already exists.
        Note: AM1.5 has 80.5 mW/cm2 between 400-1100 nm. LSH-7320 calibration is 100 mW/cm2 at 1 sun.
        The _db version also saves a superset of the parameters to the Dataverse table. 
        """
        
        # Call the 'super' of this function, the version that calculates and 
        par_df_new = keithley.calc_and_save_parameters(df,  path, samplename, cellnumber, scantype, timenowstr, 
                datetodaystr, saveparameters, verbose, timeseries, base_t, I_ph, mpp_tracking_mode)
            # Data row as a dictionary.
        row = {
            'sample_id':     f'{str(par_df_new["Sample_Name"].iloc[0])}',
            'elapsed_time':  f'{str(par_df_new["Elapsed_Time"].iloc[0])}',
            'base_time':     f'{base_t}',                                                   # New base_time value.
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
            'scan_type':     f'{str(par_df_new["Scan_Type"].iloc[0])}',                     # Will always be 'F' for the forward sweep.
            'lab_location':  f'{form_responses_dictionary["Where were the measurements done?"]}',
            'cell_number':   f'{form_responses_dictionary["Cell number?"]}',
            'module':        f'{form_responses_dictionary["Is this a module measurement?"]}',
            'masked':        f'{form_responses_dictionary["Is the sample masked?"]}',
            'mask_area':     f'{form_responses_dictionary["Mask area (cm^2)?"]}',
            'temp_c': temp_c,                                                                 # New.
            'hum_pct': hum_pct,                                                               # New.
            'four_wire_mode': four_wire,                                                      # New.
            'scan_data_path': scan_data_path                                                  # New.
        }

        # Check for consistency between the row data and the column logical names.
        res = all(self.col_logical_names.get(key) != None for key in row)
        if res != True:
            print("Error:  'row' and 'col_logical_names' keys don't match.")
            par_df_new = None
            return None

        # Insert data into the table.  Make sure the 'row' dictionary is consistent with the 'self.cols' parameter of JVScanDataverseTable.
        # To do:  we might one day also pass 'cols' as a dictionary with the same keys. Then the class constructor can make sure they are consistent.
        #returnval  =  self.insert_data(row)  # Simple version with only the required rows. 
        returnval = self.insert_data(row, image_path= image_data_path)
        # Inserting the Lookup always produces and error.  RLH 051224.
        #returnval = self.insert_data(row, sample_lookup_id='01e2d5d8-2007-ef11-9f89-000d3a55d192', sample_lookup_name='cr69a_sample_data_lookup')
        print(f'Inserting data into table "{self.table_name}".' )
            
        return par_df_new
    
    def recent_entries(self, num_samples):
        """
        Retrieve 'num_samples' records of sample data, most recent first.
        """

        if sys.platform.startswith('win32'):
            querystring = '?$select=' + self.cols                           # These work on Windows.
            orderbystring = '&"$orderby"=' + 'createdon desc'                  # 'cr69a_entry_date_and_time desc' <--forUVM.
        else:
            querystring = '?\$select=' + self.cols                          # These work on MacOS.
            orderbystring = '&"\$orderby"=' + 'createdon desc'                 # 'cr69a_entry_date_and_time desc' <--forUVM.
        querystring += orderbystring

        cli1 = f'pwsh -ExecutionPolicy Bypass -File "{self.search_path}" "{self.table_name}"\
                "{querystring}" {num_samples} "{self.crm_url}" -cols "{self.cols}"'
        result = subprocess.run(cli1, shell=True, capture_output=True, text=True)

        # Check for some common errors.
        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            print("An error occurred during the execution of the PowerShell script.")
            return None, None, result

        if len(self.columnheadings) != len(self.cols_unjoined):
            print(f'An error occurred: {len(self.columnheadings)} column headings does not match {len(self.cols_unjoined)} values.')
            return None, None, result

        result_lines = result.stdout.strip().splitlines()
        for i, string in enumerate(result_lines):
            result_lines[i] = re.split(r',\s', string)

        recent_values = pd.DataFrame(result_lines, columns=self.columnheadings)
        jv_test_ids = recent_values['sample_id'].tolist()

        return jv_test_ids, recent_values, result
    
