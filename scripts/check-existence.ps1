# Paramters will be input from a python script. Based on code by Felix Walberg.  RLH 050524.
param(
    [Parameter(Mandatory=$true)] [string]$record_id,      # The Record ID we are searching for.
    [Parameter(Mandatory=$true)] [string]$table_name,     # Logical name of the Dataverse table.
    [Parameter(Mandatory=$true)] [string]$query_string,   # Query to retrieve specific columns and order them.
    [Parameter(Mandatory=$true)] [string]$crm_url,        # The base URL of the Microsoft Dataverse environment.
    [Parameter(Mandatory=$true)] [string]$cols,           # The columns to retrieve embedded in a String, separated by `,`
    [Parameter(Mandatory=$true)] [string]$headings        # The column headings embedded in a String, separated by `,`
)

$colArray = $cols -split ','                              # Unpack the column names.
$headingArray = $headings -split ','                      # Unpack the column heading names.

# Load required scripts
. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Connect to CRM
Connect $crm_url

Invoke-DataverseCommands {
    # Retrieve existing records
    $retrieveExistingRecords = Get-Records `
        -setName $table_name `
        -query $query_string
    # Check if the record id is in the returned records; return identifying information to the host if it is found. 
    $found = $false
    foreach ($record in $retrieveExistingRecords.value) {  
        # The Record ID should always be the first column in the list.
        if ($record.$($colArray[0]) -eq $record_id) {        
            $found = $true
            $info = "PowerShell: Found the $($headingArray[0]): '$($record.$($colArray[0]))'"
            for ($i=1; $i -lt $colArray.Length; $i++) {
                $info += ", '$($headingArray[$i])': $($record.$($colArray[$i]))"
            }
            Write-Host $info
        }
    }
    # Notify the host if the record id was not found.
    if ($found -eq $false) {
        Write-Host "PowerShell: Invalid: $($headingArray[0]) was not found."
    }

}
