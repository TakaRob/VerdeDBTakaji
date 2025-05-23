# recent-entries_V3.ps1
# Parameters will be input from a python script.
# V3: Prioritizes OData $top if present in query_string for API-side limiting.
#     Otherwise, uses num_records_arg for PowerShell-side loop limiting.
param (
    [Parameter(Mandatory=$true)] [string]$table_name,     # Logical name of the Dataverse table.
    [Parameter(Mandatory=$true)] [string]$query_string,   # Query to retrieve specific columns, order, filter, and potentially $top.
    [Parameter(Mandatory=$true)] [int]$num_records_arg,   # Fallback number of records to output if $top is not in query_string.
                                                        # Python should pass 0 if OData $top is used and we want all API results.
    [Parameter(Mandatory=$true)] [string]$crm_url,        # The base URL of the Microsoft Dataverse environment.
    [Parameter(Mandatory=$true)] [string]$cols            # The columns to retrieve embedded in a String, separated by `,`.
)

$colArray = $cols -split ','

# Collections of scripts to communicate with Dataverse.
# Use $PSScriptRoot for robust relative pathing if these scripts are in the same directory as recent-entries_V3.ps1
try {
    . "$PSScriptRoot/Core.ps1"
    . "$PSScriptRoot/TableOperations.ps1"
    . "$PSScriptRoot/CommonFunctions.ps1"
}
catch {
    Write-Error "Failed to load Core.ps1, TableOperations.ps1, or CommonFunctions.ps1. Ensure they are in the same directory as recent-entries_V3.ps1 or in the specified relative path ./scripts/"
    # Fallback for compatibility if scripts are in a subdirectory named 'scripts' relative to where this script is called from
    . "./scripts/Core.ps1"
    . "./scripts/TableOperations.ps1"
    . "./scripts/CommonFunctions.ps1"
}


# Connect to the Dataverse environment.
Connect $crm_url

# Build the command, send it, and receive the result.
Invoke-DataverseCommands {
    # The Get-Records cmdlet should ideally just execute the query string as is.
    # Dataverse will handle the $top if it's in $query_string.
    $retrieveExistingRecords = Get-Records `
      -setName $table_name `
      -query $query_string

    # Check if OData $top was likely used in the query string
    $odataTopSpecified = $query_string -match '\$top=\d+'

    $count = 0
    $recordsToOutput = $retrieveExistingRecords.value

    if ($recordsToOutput) {
        foreach ($record in $recordsToOutput) {
            $shouldOutput = $false
            if ($odataTopSpecified) {
                # If OData $top was in the query, Dataverse already limited the results.
                # Python's get_entries_by_criteria passes num_samples_to_fetch as ps_num_samples_arg.
                # If num_samples_to_fetch was specified (and >0), it's also in $query_string as $top.
                # If num_samples_to_fetch was None/0, ps_num_samples_arg is 0, and $top is not in $query_string.
                # This script's loop limiting via $num_records_arg is only truly a fallback.
                # If $odataTopSpecified is true, we assume API handled it.
                $shouldOutput = $true 
            } elseif ($num_records_arg -gt 0 -and $count -lt $num_records_arg) {
                # If OData $top was NOT specified AND num_records_arg is positive, then limit here.
                $shouldOutput = $true
            } elseif ($num_records_arg -le 0 -and !$odataTopSpecified) {
                # If OData $top was NOT specified AND num_records_arg is 0 or negative, output all (no client-side limit).
                $shouldOutput = $true
            }

            if ($shouldOutput) {
                $outputLine = ""
                foreach ($colName in $colArray) {
                    # Ensure property exists before trying to access it to avoid errors on sparse data
                    if ($record.PSObject.Properties[$colName] -ne $null) {
                        $outputLine += "$($record.$colName), "
                    } else {
                        $outputLine += ", " # Add a comma for placeholder to maintain CSV structure
                    }
                }
                Write-Host $outputLine.TrimEnd(", ") # Output the formatted string for current record
            }
            $count++
        }
    } else {
        # Write-Host "No records returned from Dataverse for query: $query_string" # Optional: for debugging
    }
}