. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1

Connect 'https://orgc25b23b3.api.crm.dynamics.com/' # change this




# Retrieve Records
Write-Host 'Retrieve first three account records:'
(Get-Records -setName accounts -query '?$select=name&$top=3').value | 
Format-Table -Property name, accountid

# # Create a record
Write-Host 'Create an account record:'
$newAccountID = New-Record -setName accounts -body @{
      name                = 'Example Account'; 
      accountcategorycode = 1 # Preferred
   }
Write-Host "Account with ID $newAccountID created"

# # Retrieve a record
# Write-Host 'Retrieve the created record:'
# Get-Record -setName  accounts -id $newAccountID.Guid '?$select=name,accountcategorycode' |
# Format-List -Property name,
# accountid,
# accountcategorycode,
# accountcategorycode@OData.Community.Display.V1.FormattedValue

# # Update a record
# Write-Host 'Update the record:'
# $updateAccountData = @{
#    name                = 'Updated Example account';
#    accountcategorycode = 2; #Standard
# }
# Update-Record -setName accounts -id $newAccountID.Guid -body $updateAccountData
# Write-Host 'Retrieve the updated the record:'
# Get-Record -setName accounts -id  $newAccountID.Guid -query '?$select=name,accountcategorycode' |
# Format-List -Property name,
# accountid,
# accountcategorycode,
# accountcategorycode@OData.Community.Display.V1.FormattedValue

# # Delete a record
Write-Host 'Delete the record:'
Remove-Record -setName accounts -id $newAccountID.Guid
Write-Host "The account with ID $newAccountID was deleted"



# # Function call with invalid setName input
# Invoke-DataverseCommands {

#     # Retrieve Records
#     Write-Host 'Retrieve first three account records:'
#        (Get-Records `
#        -setName account `
#        -query '?$select=name&$top=3').value | 
#     Format-Table -Property name, accountid
 
#  }

