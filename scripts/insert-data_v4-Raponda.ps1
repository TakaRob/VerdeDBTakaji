#  Based on code by Felix Walberg, Spring 2024.  RLH 050624.
#  This script requires mandatory column data for the JV_test_data table.
#
param (
    [Parameter(Mandatory=$true)]  [string] $crm_url,          # The base URL of the Microsoft Dataverse environment.
    [Parameter(Mandatory=$true)]  [string] $table_name,       # Logical name of the Dataverse table.
    [Parameter(Mandatory=$true)]  [string] $cols,             # The columns to insert data into, embedded in a String separated by `,`
    [Parameter(Mandatory=$true)]  [string] $sample_id,        # 0 Mandatory parameters describing the measurement and results.
    [Parameter(Mandatory=$true)]  [string] $elapsed_time,     # 1
    [Parameter(Mandatory=$true)]  [string] $base_time,        # 2
    [Parameter(Mandatory=$true)]  [string] $test_id,          # 3
    [Parameter(Mandatory=$true)]  [string] $i_ph_suns,        # 4
    [Parameter(Mandatory=$true)]  [string] $voc_v,            # 5
    [Parameter(Mandatory=$true)]  [string] $mpp_v,            # 6
    [Parameter(Mandatory=$true)]  [string] $jsc_ma,           # 7
    [Parameter(Mandatory=$true)]  [string] $rsh,              # 8
    [Parameter(Mandatory=$true)]  [string] $rser,             # 9
    [Parameter(Mandatory=$true)]  [string] $ff,               # 10  
    [Parameter(Mandatory=$true)]  [string] $pce,              # 11
    [Parameter(Mandatory=$true)]  [string] $operator,         # 12
    [Parameter(Mandatory=$true)]  [string] $scan_type,        # 13
    [Parameter(Mandatory=$true)]  [string] $lab_location,     # 14
    [Parameter(Mandatory=$true)]  [string] $cell_number,      # 15
    [Parameter(Mandatory=$true)]  [string] $module,           # 16
    [Parameter(Mandatory=$true)]  [string] $masked,           # 17
    [Parameter(Mandatory=$true)]  [string] $mask_area,        # 18
    [Parameter(Mandatory=$true)]  [string] $temp_c,           # 19
    [Parameter(Mandatory=$true)]  [string] $hum_pct,          # 20
    [Parameter(Mandatory=$true)]  [string] $four_wire_mode,   # 21
    [Parameter(Mandatory=$true)]  [string] $folder_path,      # 22
    [Parameter(Mandatory=$false)] [string] $sampleDataRecordId, # Optional parameter for the sample data record ID.
    [Parameter(Mandatory=$false)] [string] $sampleDataLookupColumn, # Optional parameter for the lookup column name.
    [Parameter(Mandatory=$false)] [string] $image_path,       # Path to the optional image file to upload.
    [Parameter(Mandatory=$false)] [string] $image_column_name # Logical name of the column that contains the images.
)

$colArray = $cols -split ','                                  # Unpack the column logical names. They have names like 'cr69a_sample_id'.

# Import the Set-FileColumn function script and other helper scripts.
. .\scripts\Set-FileColumn.ps1
. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

Connect $crm_url

Invoke-DataverseCommands {
    Write-Host "Sample ID: '$($sample_id)'"
    Write-Host "Test ID: '$($test_id)'"
    Write-Host "crm url: '$($crm_url)'"
    Write-Host "Table name: '$($table_name)'"
    # Change Yes/No to boolean values.
    if ($masked -like 'No') { $masked = $false  }
    else { $masked = $true }
    if ($module -like 'No') {$module = $false }
    else {$module = $true}
    if ($four_wire_mode -like 'No') {$four_wire_mode = $false }
    else {$four_wire_mode = $true}

    # Create a cell object to hold the value to be entered into the dataverse database.
    $cell = @{
            $colArray[0] =     $sample_id
            $colArray[1] =     [decimal]$elapsed_time
            $colArray[2] =     [decimal]$base_time
            $colArray[3] =     $test_id 
            $colArray[4] =     [decimal]$i_phsuns
            $colArray[5] =     [decimal]$voc_v
            $colArray[6] =     $mppv
            $colArray[7] =     [decimal]$js_ma
            $colArray[8] =     [decimal]$rsh
            $colArray[9] =     [decimal]$rser
            $colArray[10] =     [decimal]$ff
            $colArray[11] =    [decimal]$pce
            $colArray[12] =    $operator
            $colArray[13] =    $scan_type
            $colArray[14] =    $lab_location
            $colArray[15] =    [int]$cell_number
            $colArray[16] =    [bool]$module   
            $colArray[17] =    [bool]$masked
            $colArray[18] =    [decimal]$mask_area
            $colArray[19] =    $temp_c
            $colArray[20] =    $hum_pct
            $colArray[21] =    [bool]$four_wire_mode
            $colArray[22] =    $folder_path
           }

#######  The code below gives an error, so I have commented it out for now.  #######
    # # Check if the sampleDataRecordId and sampleDataLookupColumn are provided.
    # if ('' -ne $sampleDataRecordId -and '' -ne $sampleDataLookupColumn) {
    #     # Construct the JSON payload for the lookup value.
    #     $lookupPayload = @{
    #         "@odata.type" = "#Microsoft.Dynamics.CRM.Lookup"
    #         "entityType" = "cr69a_jv_test_data_v2"
    #         "id" = "01e2d5d8-2007-ef11-9f89-000d3a55d192"
    #     } | ConvertTo-Json -Compress

    #     # Set the lookup column in the cell object.
    #     $cell['cr69a_sample_data_lookup'] = $lookupPayload
    #     Write-Host "cell: '$($cell['cr69a_sample_data_lookup'])'"
    # }
   
   # Insert the entry into table with name stored in -setName, with the content stored in -body.
    $cellID = New-Record `
        -setName $table_name `
        -body $cell
    Write-Host "CellID: $($cellID)"
    Write-Host "Type of `$cellID: $($cellID.GetType())"
    $cellIDString = $cellID.ToString()
    Write-Host "CellIDString: $($cellIDString)"
    
    # Check if image_path and image_column_name are provided.
    if ('' -ne $image_path -and '' -ne $image_column_name) {
        # Upload image file using the obtained record ID.
        Set-FileColumn `
            -setName $table_name `
            -id $cellIDString     # Use double quotes around $cellID
            -columnName $image_column_name `
            -file $image_path 
    } 



    #     # Check if image_path and image_column_name are provided.
    
    # # Retrieve the most recent record ID.
    # $record = Get-Records `
    #     -setName $table_name `
    #     -query '?$orderby=createdon desc&$top=1'  # Order by createdon in descending order and get the top 1 record.
    # $recordId = $record.value[0].cr69a_jv_test_data_v2id
    # Write-Host "Record ID: '$($recordId)'"
    
    # if ('' -ne $image_path -and '' -ne $image_column_name) {
    #     # Upload image file using the obtained record ID.
    #     Set-FileColumn `
    #         -setName $table_name `
    #         -id $recordId `
    #         -columnName $image_column_name `
    #         -file $image_path 
    # }
}