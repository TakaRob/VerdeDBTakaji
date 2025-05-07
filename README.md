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

---
## Setting Up Python Environment for Verde Database Scripts and Notebooks

This section guides you through setting up a Conda environment to run the Python scripts (`db_dataverse.py`, `db_form.py`) and Jupyter notebooks (`.ipynb` files such as `Solar_Simulator_IV_sweep_with_lamp_control_V4s_verde.ipynb`) described in the "Files" section of this document.

### Prerequisites for Python Setup

1.  **Git:** (If not already installed for previous PowerShell steps). You need Git installed on your system to clone the repository. You can download it from https://git-scm.com/.
2.  **`environment.yml` file:** This file should be present in the root of the `Verde_database` project repository. It defines the Conda environment and its dependencies. An example structure relevant to this project might be:

    ```yaml
    name: verde_env # Suggested environment name for this project
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python=3.9 # Or your project's Python version
      - jupyterlab
      - pandas # Likely needed for data handling and notebooks
      # Consider adding packages for Dataverse interaction if db_dataverse.py needs them, e.g.:
      # - msal # For authentication if using MSAL directly
      # - requests
      # - azure-identity # If using Azure SDK for authentication
      # Add other specific packages required by your Python scripts and notebooks
      - pip
      # - pip:
      #   - some-pip-package # If any pip-specific packages are needed
    ```

---

### Step 1: Clone the Project Repository (If you haven't already)

If you don't have the `Verde_database` project files on your system yet, open your terminal or Git Bash. Use the `git clone` command. Replace `<VERDE_DATABASE_REPOSITORY_URL>` with the actual URL for the project.

```bash
git clone <VERDE_DATABASE_REPOSITORY_URL>
```

### Step 2: Install Miniconda (If Conda is not yet installed)

Conda will manage our Python environment and its packages. We recommend installing Miniconda, which is a minimal installer for Conda. This is a separate step from installing Python for PowerShell, if applicable.

1. Go to the Miniconda download page: Miniconda — conda documentation
2. Download the installer appropriate for your operating system.
3. Run the installer and follow the on-screen instructions. For consistency, it's often easier to use Anaconda Prompt on Windows or ensure your terminal is correctly initialized for Conda on macOS/Linux.

---

### Step 3: Open Anaconda Prompt (or Your Terminal)

- **Windows:** Search for "Anaconda Prompt" in your Start Menu and open it. This prompt is pre-configured to use Conda commands.
- **macOS/Linux:** Open your standard terminal window. (Ensure Conda has been initialized for your shell by the Miniconda installer, which might require restarting your terminal).

---

### Step 4: Navigate to the Cloned Project Directory

In the Anaconda Prompt or terminal you opened, change your current directory to the Verde_database folder (or whatever your cloned project directory is named).

```bash
cd path/to/Verde_database
```

---

### Step 5: Create the Conda Environment

Use the environment.yml file from the project to create the dedicated Conda environment. We suggest naming the environment **verde_env** for this project, or use the name specified within the environment.yml if it has one.

```bash
conda env create -f environment.yml -n verde_env
```

---

### Step 6: Activate the Conda Environment

Once the environment is created, you need to activate it. This makes the environment's Python interpreter and packages available in your current shell session.

```bash
conda activate verde_env
```

Your terminal prompt should now change to indicate that the environment is active (e.g., `(verde_env) C:\Users\YourUser\path\to\Verde_database>`).

---

### Step 7: Launch JupyterLab

With the **verde_env** (or your chosen environment name) activated, you can now launch JupyterLab. This will allow you to work with the project's Jupyter notebooks, such as `Solar_Simulator_IV_sweep_with_lamp_control_V4s_verde.ipynb` and `Solar_Simulator_timeseries_report_V1.ipynb`.

```bash
jupyter lab
```

This command will start a JupyterLab server and typically open it in your default web browser.

---

This concludes the setup for the Python/Conda environment for the Verde_database project. You should now be able to run the project's Python scripts and Jupyter notebooks.
