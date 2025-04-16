# Verde_database
 Scripts for accessing Microsoft Dataverse tables specific to Verde Technologies Inc.

 Files:

 db_dataverse.py
 Package with functions needed to access Microsoft Dataverse tables.  It is intended to work with the packages in the Solar-Simulator_IV_Sweep repository. This version is specific to Verde Technologies Inc.

db_form.py and main_questions.json
Package to create a simple form within a Jupyter notebook.  It has a few quirks, such as a limitation of only one checkbox question on the form.  To do: check to see whether there are multiple inconsistent versions of this package.

Solar_Simulator_IV_sweep_with_lamp_control_V4s_verde.ipynb
A version of the solar simulator sofware that works with the database packages desceribed above.  


Solar_Simulator_timeseries_report_V1.ipynb
Unfinished script to load data from a Dataverse table with J-V test data and display it as a timeseries or as a box-and-whisker plot.


Folders:

/scripts

Powershell scripts that interact with Microsoft Dataverse tables.  These scripts should be fairly general.