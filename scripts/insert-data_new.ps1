. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Input from python script, if there are additional fields added, add to this list of arguments.
$test_id = $args[0]
$test_value = $args[1]

Connect 'https://orgc25b23b3.api.crm.dynamics.com/'

Invoke-DataverseCommands {
    Write-Host "Test ID: '$($test_id)'"
    Write-Host "Test value: '$($test_value)'"

    # Create a cell object to hold the value to be entered into the dataverse database.
    $cell = @{
            'cr69a_testid' = $test_id
            'cr69a_testvalue' = $test_value
           }
    Write-Host "Cell: '$($cell.'cr69a_testid')'"
    Write-Host "Cell: '$($cell.'cr69a_testvalue')'"


    # # Construct the JSON payload for the lookup value.
    # $lookupPayload = @{
    #     "@odata.type" = "#Microsoft.Dynamics.CRM.Lookup"
    #     "entityType" = "cr69a_jv_test_v2"
    #     "id" = "1fa9cb02-1b10-ef11-9f8a-0003a55d192"
    # } | ConvertTo-Json -Compress

    # # Set the lookup column in the cell object.
    # $cell['cr69a_sample_data_lookup'] = $lookupPayload
    # Write-Host "cell: '$($cell.'cr69a_sample_data_lookup')'"
   
    # Insert the entry into table with name stored in -setName, with the content stored in -body.
    $cellID = New-Record `
        -setName 'cr69a_jv_test_v2s' `
        -body $cell
    Write-Host "cellID: '$($cellID)'"
    
    
}