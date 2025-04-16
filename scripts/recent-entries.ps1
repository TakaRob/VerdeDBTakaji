. .\scripts\Core.ps1
. .\scripts\TableOperations.ps1
. .\scripts\CommonFunctions.ps1

# Input from python script, if there are additional fields added, add to this list of arguments.
# $table_name = "cr69a_sampledatav2s"
$table_name = $args[0]
$num_samples = $args[1]

Connect 'https://orgc25b23b3.api.crm.dynamics.com/'

Invoke-DataverseCommands {

    $retrieveExistingSamples = Get-Records `
      -setName $table_name `
      -query '?$select=cr69a_sample_id,cr69a_operator,cr69a_perovskite_composition,cr69a_htl_material,cr69a_etl_material,cr69a_top_capping_material,cr69a_bottom_capping_material,cr69a_bulk_passivation_materials,cr69a_is_encapsulated,cr69a_entry_date_and_time&$orderby=cr69a_entry_date_and_time desc'
    $count = 0
    Write-Host "Sample ID, Operator, Perovskite Composition, HTL Material, ETL Material, Top Capping Material, Bottom Capping Material, Bulk Passivation Materials, Is Encapsulated"
    foreach ($sample in $retrieveExistingSamples.value) {
        if($count -le $num_samples)
        {
            Write-Host "$($sample.cr69a_sample_id), $($sample.cr69a_operator), $($sample.cr69a_perovskite_composition), $($sample.cr69a_htl_material), $($sample.cr69a_etl_material), $($sample.cr69a_top_capping_material), $($sample.cr69a_bottom_capping_material), $($sample.cr69a_bulk_passivation_materials), $($sample.cr69a_is_encapsulated)" `
        }
        $count = $count + 1
     }
    
   
    
}
