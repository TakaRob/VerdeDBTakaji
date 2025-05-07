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
 
 # --- Start Modification ---

 # Get an access token object (now includes -AsSecureString)
    $tokenInfo = Get-AzAccessToken -ResourceUrl $uri -AsSecureString # <<< Added -AsSecureString

 # Extract the SecureString token
    $secureToken = $tokenInfo.Token

 # Convert the SecureString to plain text *immediately* before use (Use with caution!)
 # This is necessary because the 'Authorization' header requires a plain text string.
    $plainTextToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    )

 # Define common set of headers using the plain text token
    $global:baseHeaders = @{
       'Authorization'    = 'Bearer ' + $plainTextToken # <<< Use the converted plain text token here
       'Accept'           = 'application/json'
       'OData-MaxVersion' = '4.0'
       'OData-Version'    = '4.0'
    }

 # Optionally, clear the plain text variable from memory if possible (though it's used in $global:baseHeaders)
 # Clear-Variable plainTextToken # Consider if/where this makes sense in your flow

 # --- End Modification ---

 # Set baseURI
    $global:baseURI = $uri + 'api/data/v9.2/'
 } # End of Connect function

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
 } # End of Invoke-DataverseCommands function