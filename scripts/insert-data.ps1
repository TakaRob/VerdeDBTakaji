. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Input from python script, if there are additional fields added, add to this list of arguments.
$elapsed_time = $args[0]
$cell_name = $args[1]
$phsuns = $args[2]
$vocv = $args[3]
$mppv = $args[4]
$jscma = $args[5]
$rsh = $args[6]
$rser = $args[7]
$ff = $args[8]
$pce = $args[9]
Connect 'https://orgc25b23b3.api.crm.dynamics.com/'

Invoke-DataverseCommands {
    Write-Host "Here is the entry: '$($cell_name)'"

    # Create a cell object to hold the value to be entered into the dataverse database.
    $cell = @{
            'cr69a_sampleid' = $cell_name
            'cr69a_elapsedtime' = $elapsed_time
            'cr69a_i_phsuns' = $phsuns
            'cr69a_v_ocv' = $vocv
            'cr69a_mppv' = $mppv
            'cr69a_j_scma' = $jscma
            'cr69a_r_sh' = $rsh
            'cr69a_r_ser' = $rser
            'cr69a_ff' = $ff
            'cr69a_pce' = $pce
           }
   
    # Insert the entry into table with name stored in -setName, with the content stored in -body.
    $cellID = New-Record `
        -setName 'cr69a_sampledatas' `
        -body $cell
    
    
}