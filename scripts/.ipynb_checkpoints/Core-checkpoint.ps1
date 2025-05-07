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

 #edits here ~~~~~~~~
 # Get an access token object
    #$token = (Get-AzAccessToken -ResourceUrl $uri).Token
    $tokenInfo = Get -AzAccessToken -ResourceUrl $uri -AsSecureString

# extract token
    $secureToken = $tokenInfo.Token

    #convert to plaintext
    $plainTextToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    )
 
 # Define common set of headers #turn the token to plaintexttoken
    $global:baseHeaders = @{
       'Authorization'    = 'Bearer ' + $plainTextToken
       'Accept'           = 'application/json'
       'OData-MaxVersion' = '4.0'
       'OData-Version'    = '4.0'
    }
# end edit ~~~~~~~~~
 
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