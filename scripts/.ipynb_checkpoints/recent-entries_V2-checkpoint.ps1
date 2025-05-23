# Paramters will be input from a python script. Based on code by Felix Walberg.  RLH 050424.
param (
    [Parameter(Mandatory=$true)] [string]$table_name,     # Logical name of the Dataverse table.
    [Parameter(Mandatory=$true)] [string]$query_string,   # Query to retrieve specific columns and order then.
    [Parameter(Mandatory=$true)] [int]$num_records,       # Number of records to pass back via `Write-Host`.
    [Parameter(Mandatory=$true)] [string]$crm_url,        # The base URL of the Microsoft Dataverse environment.
    [Parameter(Mandatory=$true)] [string]$cols            # The columns to retrieve embedded in a String, separated by `,`.
)

$colArray = $cols -split ','                              # Unpack the column names.

# Collections of scripts to communicate with Dataverse.
. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Connect to the Dataverse environment.
Connect $crm_url

# Build the command, send it, and receive the result. Also, pass the requested number of records back to the Host.
Invoke-DataverseCommands {

    $retrieveExistingRecords = Get-Records `
      -setName $table_name `
      -query $query_string

    $count = 0
    foreach ($record in $retrieveExistingRecords.value) {
        if($count -lt $num_records)
        {
            $output = ""
            foreach ($col in $colArray) {
                $output += "$($record.$col), "
            }
            Write-Host $output.TrimEnd(", ")
        }
        $count++
     }
    
}



