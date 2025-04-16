function Connect {
    param (
       [Parameter(Mandatory)] 
       [String] 
       $uri
    )

 ## Login interactively if not already logged in
    if ($null -eq (Get-AzTenant -ErrorAction SilentlyContinue)) {
       Connect-AzAccount | Out-Null
    }
 
 # Get an access token
    $token = (Get-AzAccessToken -ResourceUrl $uri).Token
 
 # Define common set of headers
    $global:baseHeaders = @{
       'Authorization'    = 'Bearer ' + $token
       'Accept'           = 'application/json'
       'OData-MaxVersion' = '4.0'
       'OData-Version'    = '4.0'
    }
 
 # Set baseURI
    $global:baseURI = $uri + 'api/data/v9.2/'
 }

 # The Invoke-DataverseCommands tries everything in a try-catch block so that error messages are more descriptive
 function Invoke-DataverseCommands {
    param (
       [Parameter(Mandatory)] 
       $commands
    )
    try {
       Invoke-Command $commands -NoNewScope
    }
    catch [Microsoft.PowerShell.Commands.HttpResponseException] {
       Write-Host "An error occurred calling Dataverse:" -ForegroundColor Red
       $statuscode = [int]$_.Exception.StatusCode;
       $statusText = $_.Exception.StatusCode
       Write-Host "StatusCode: $statuscode ($statusText)"
       # Replaces escaped characters in the JSON
       [Regex]::Replace($_.ErrorDetails.Message, "\\[Uu]([0-9A-Fa-f]{4})", 
          {[char]::ToString([Convert]::ToInt32($args[0].Groups[1].Value, 16))} )
 
    }
    catch {
       Write-Host "An error occurred in the script:" -ForegroundColor Red
       $_
    }
 }