# Verde_database
 ## Scripts for accessing Microsoft Dataverse tables specific to Verde Technologies Inc.

 ### Files:

 #### db_dataverse.py
 Package with functions needed to access Microsoft Dataverse tables.  It is intended to work with the packages in the Solar-Simulator_IV_Sweep repository. This version is specific to Verde Technologies Inc.

#### db_form.py and main_questions.json
Package to create a simple form within a Jupyter notebook.  It has a few quirks, such as a limitation of only one checkbox question on the form.  To do: check to see whether there are multiple inconsistent versions of this package.

#### Solar_Simulator_IV_sweep_with_lamp_control_V4s_verde.ipynb
A version of the solar simulator sofware that works with the database packages desceribed above.  


#### Solar_Simulator_timeseries_report_V1.ipynb
Unfinished script to load data from a Dataverse table with J-V test data and display it as a timeseries or as a box-and-whisker plot.


### Folders:

#### /scripts

Powershell scripts that interact with Microsoft Dataverse tables.  These scripts should be fairly general.

 

## Setting Up PowerShell 7.5 and Azure Az Module

This guide details the steps required to install PowerShell 7.5 and configure the Azure Az module for managing Azure resources.

---

### Step 1: Install PowerShell 7.5

First, ensure you have PowerShell version 7.5 installed.

1.  **Download:** Obtain the installer from the official release page:
    *   [https://github.com/PowerShell/PowerShell/releases/tag/v7.5.0](https://github.com/PowerShell/PowerShell/releases/tag/v7.5.0)
2.  **Install:** Select the package appropriate for your Operating System and Chip Architecture (e.g., Windows x64, Linux Arm64). Run the installer and proceed through the default installation steps.
3.  **Launch:** Open your terminal and start PowerShell 7.5 by executing:
    ```powershell
    pwsh
    ```
    *   **Note:** On Windows systems, running `powershell` typically launches the older, built-in Windows PowerShell (version 5.1). Use `pwsh` explicitly to run version 7.5.
4.  **Verify Version:** Confirm the correct version is running using this command within the `pwsh` terminal:
    ```powershell
    $PSVersionTable.PSVersion
    ```
    You should see output indicating the Major version is 7 and the Minor version is 5 (e.g., 7.5.0).

---

### Step 2: Configure Execution Policy

Adjusting the execution policy allows for easier script execution and module installation from trusted sources without requiring elevated privileges for the policy change itself.

1.  **Check Policies:** View the current execution policies applied:
    ```powershell
    Get-ExecutionPolicy -List
    ```
    This will display the policies set for different scopes (like MachinePolicy, UserPolicy, CurrentUser, etc.).
2.  **Set Policy for Current User:** Execute the following command to set the policy to `RemoteSigned` specifically for your user account:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
    *   Using `-Scope CurrentUser` means this change applies only to your user profile and typically does not require administrator rights ('Run as administrator').
    *   You will likely be prompted to confirm this change; enter `Y` and press Enter.

---

### Step 3: Install the Azure Az Module

Install the necessary PowerShell module to interact with Azure services.

1.  **Install Command:** Run the following in your `pwsh` terminal. You can find more details on the installation process here: [Microsoft Docs](https://learn.microsoft.com/bs-latn-ba/powershell/azure/install-azps-windows?view=azps-13.4.0&tabs=powershell&pivots=windows-psgallery).
    ```powershell
    Install-Module -Name Az -Repository PSGallery -Scope CurrentUser -Force -Verbose
    ```
    *   `-Scope CurrentUser`: Installs the module in your user directory, avoiding the need for administrator privileges.
    *   `-Force`: Ensures installation even if an older version exists or if confirmation prompts would otherwise appear.
    *   `-Verbose`: Provides detailed output during installation. **I personally always use the `-Verbose` tag** as it lets you see exactly what is happening during potentially long operations.
2.  **Expected Behavior:** This command downloads and installs numerous Az sub-modules. The process can take approximately **10 minutes or more**. The `-Verbose` output will show ongoing activity.

---

### Step 4: Verify and Update Az Module

Confirm the module installation and ensure it is up-to-date.

1.  **Verify Installation:** Check if the Az module is now available to your PowerShell session:
    ```powershell
    Get-Module -Name Az -ListAvailable
    ```
    You should see output listing the `Az` module directory and its version.
2.  **Update Module:** It's good practice to ensure you have the latest version. Run the update command:
    ```powershell
    Update-Module -Name Az -Force -Verbose
    ```
    *   This operation can also take **around 10 minutes**.
    *   Again, `-Verbose` is useful here to monitor the update progress.

---

### Step 5: Connect to Azure

With the environment prepared, connect your PowerShell session to your Azure account.

1.  **Connect Command:** Execute the following:
    ```powershell
    Connect-AzAccount
    ```
2.  **Authentication:** Upon running this command, you should expect a browser window to open prompting you to log in to your Azure account, or you might receive a device code to authenticate via `https://microsoft.com/devicelogin`. Follow the on-screen instructions.

---

**Setup Complete.** Your PowerShell 7.5 environment is now configured with the Azure Az module, and you should be able to execute commands against your Azure subscription.