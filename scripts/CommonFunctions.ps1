function Get-WhoAmI{

    $WhoAmIRequest = @{
          Uri = $baseURI + 'WhoAmI'
          Method = 'Get'
          Headers = $baseHeaders
       }
    
    Invoke-RestMethod @WhoAmIRequest
    }

# This function has a parameter of the limit for how many entries/requests you can make with the API at a time
# if it exceeds the limit it will wait a few seconds and try again.
    function Invoke-ResilientRestMethod {
        param (
           [Parameter(Mandatory)] 
           $request,
           [bool]
           $returnHeader
        )
        try {
           Invoke-RestMethod @request -ResponseHeadersVariable rhv
           if ($returnHeader) {
              return $rhv
           }
        }
        catch [Microsoft.PowerShell.Commands.HttpResponseException] {
           $statuscode = $_.Exception.Response.StatusCode
           # 429 errors only
           if ($statuscode -eq 'TooManyRequests') {
              if (!$request.ContainsKey('MaximumRetryCount')) {
                 $request.Add('MaximumRetryCount', 3)
                 # Don't need - RetryIntervalSec
                 # When the failure code is 429 and the response includes the Retry-After property in its headers, 
                 # the cmdlet uses that value for the retry interval, even if RetryIntervalSec is specified
              }
              # Will attempt retry up to 3 times
              Invoke-RestMethod @request -ResponseHeadersVariable rhv
              if ($returnHeader) {
                 return $rhv
              }
           }
           else {
              throw $_
           }
        }
        catch {
           throw $_
        }
     }