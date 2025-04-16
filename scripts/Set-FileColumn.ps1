function Set-FileColumn {
   param (
      [Parameter(Mandatory=$true)]
      [string]
      $setName,

      [Parameter(Mandatory=$true)]
      [System.Guid]
      $id,

      [Parameter(Mandatory=$true)]
      [string]
      $columnName,

      [Parameter(Mandatory=$true, ValueFromPipeline, ValueFromPipelineByPropertyName)]
      [System.IO.FileInfo]$file
   )

   $uri = '{0}{1}({2})/{3}' -f $baseURI, $setName, $id, $columnName

   $patchHeaders = $baseHeaders.Clone()
   $patchHeaders.Add('Content-Type', 'application/octet-stream')
   $patchHeaders.Add('x-ms-file-name', $file.Name)

   $body = [System.IO.File]::ReadAllBytes($file.FullName)

   $FileUploadRequest = @{
      Uri     = $uri
      Method  = 'Patch'
      Headers = $patchHeaders
      Body    = $body
   }

   Invoke-RestMethod @FileUploadRequest
}