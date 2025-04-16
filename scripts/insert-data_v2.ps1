. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Input from python script, if there are additional fields added, add to this list of arguments.
$sample_id =    $args[0]
$elapsed_time = $args[1]
$test_id =      $args[2]
$i_phsuns =     $args[3]
$voc_v =        $args[4]
$mpp_v =        $args[5]
$jsc_ma =       $args[6]
$rsh =          $args[7]
$rser  =        $args[8]
$ff =           $args[9]
$pce =          $args[10]
$operator =     $args[11]
$scan_type =    $args[12]
$lab_location = $args[13]
$cell_number =  $args[14]
$module =       $args[15]
$masked =       $args[16]
$mask_area =    $args[17]


Connect 'https://orgc25b23b3.api.crm.dynamics.com/'

Invoke-DataverseCommands {
    Write-Host "Sample ID: '$($sample_id)'"
    Write-Host "Test ID: '$($test_id)'"
    if ($masked -like 'No') { $masked = $false  }
    else { $masked = $true }
    Write-Host "Masked? : $masked"
    if ($module -like 'No') {$module = $false }
    else {$module = $true}
    Write-Host "Module? : $module"

    # Create a cell object to hold the value to be entered into the dataverse database.
    $cell = @{
            'cr69a_sample_id' =        $sample_id
            'cr69a_elapsed_time_min' = [decimal]$elapsed_time
            'cr69a_test_id' =          $test_id
            'cr69a_iph_suns' =         [decimal]$i_phsuns
            'cr69a_voc_v' =            [decimal]$voc_v
            'cr69a_mpp_v' =            $mppv
            'cr69a_jsc_macm2' =        [decimal]$js_ma
            'cr69a_rsh' =              [decimal]$rsh
            'cr69a_rser' =             [decimal]$rser
            'cr69a_ff_' =              [decimal]$ff
            'cr69a_pce_' =             [decimal]$pce
            'cr69a_operator_name' =    $operator
            'cr69a_scan_type' =        $scan_type
            'cr69a_location' =         $lab_location
            'cr69a_cell_number'  =     [int]$cell_number
            'cr69a_module' =           $module   
            'cr69a_masked' =           $masked
            'cr69a_mask_area_cm2' =    [decimal]$mask_area
           }
   
    # Insert the entry into table with name stored in -setName, with the content stored in -body.
    $cellID = New-Record `
        -setName 'cr69a_jv_test_data_v2s' `
        -body $cell
    
    
}