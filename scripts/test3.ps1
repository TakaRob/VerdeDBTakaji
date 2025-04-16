. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Input from python script
$cell_name = $args[0]
$pce = $args[1]
$file = $args[2]

Connect 'https://orgc25b23b3.api.crm.dynamics.com/'
Invoke-DataverseCommands {
    # $sampleCell = @{
    #     'cr69a_sampleCell' = $args[0]
    #  }

    #  $cellID = New-Record `
    #     -setName 'cr69a_testSolars' `
    #     -body $sampleCell

        # $cell = @{
        #     'cr69a_petname' = $value
        #     'cr69a_species' = 1
        #     'cr69a_breed' = 'Lab'
        #    }
      
        #    $cellID2 = New-Record `
        #     -setName 'cr69a_pets' `
        #     -body $cell
        
    Write-Host "Here is the entry: '$($cell_name)'"
    $cell = @{
            'cr69a_petname' = $cell_name
            'cr69a_species' = 1
            'cr69a_file' = $file
            'cr69a_breed' = $pce
           }
    $cellID2 = New-Record `
        -setName 'cr69a_pets' `
        -body $cell

    # Retrieve all values with a pce over 12.5.

   $cellsInDatabase = Get-Records `
   -setName 'cr69a_pets' `
   -query ('({0})/cr69a_pets?$select=cr69a_petname,cr69a_breed &$filter=cr69a_breed ge 12.5' `
      -f $accountFourthCoffeeId)

    # Find 'Rafel Shillo' in the list of 'Fourth Coffee' account contacts
    Write-Host "Contact list for account '$($accountFourthCoffee.name)':"
    foreach ($contact in $fourthCoffeeContacts.value) {
    Write-Host "`tName: $($contact.fullname)," `
        "Job title: $($contact.jobtitle)"
    }

    # Remove 'Rafel Shillo' from the contact list of 'Fourth Coffee'

    Remove-FromCollection `
    -targetSetName 'accounts' `
    -targetId $accountFourthCoffeeId `
    -collectionName 'contact_customer_accounts' `
    -id $rafelShilloId | Out-Null

    # $sampleCell = @{
    #     'cr69a_sampleCell' = $cell_name
    #     'cr69a_pce' = $pce
    #     'cr69a_file' = $file
    # }
    # $cellID = New-Record `
    #     -setName 'cr69a_testSolar' `
    #     -body $sampleCell
}