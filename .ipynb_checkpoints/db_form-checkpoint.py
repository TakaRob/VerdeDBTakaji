# tkinter does not work well with Jupyter notebooks, so I switched to ipywidgets
# based on the recommendation here: 
# https://swan-community.web.cern.ch/t/creating-user-interface-with-tkinter/321
# Based on code by Yoni Masha.
# RLH 4/18/2024.  V2  4/25/2024 RLH.
# RLH 2/11/25.  Add the option to provide a custom list of checkbox values that is not in the json file.

import ipywidgets as widgets
import json
import os
import sys

class parameters_form():

    def __init__(self, questions_file_main = "./main_questions.json", dir_path_local = "./", persistent = False, drop_down_values=[], checkbox_values=[]):
        """
        Constructor.
        """
        #  
        self.vars = []             # List of widgets with answer choices and defaults.
        self.questions = []        # Questions loaded from the json file. 
        self.questions_file_main= questions_file_main # Location of the main json questions file.
        self.dir_path_local = dir_path_local          # Local directory.
        self.questions_file_local = ''                # Name of the local json questions file. It is automatically generated.
        self.persistent = persistent                  # 'True'loads the local json file when the form is refreshed.
        self.responses = []                           # Populated after the 'Submit' button is clicked.
        self.local_file_save_result = ''              # Used by get_responses to see if the local file was saved. 
        self.dropdown_values = drop_down_values       # In this version we can have at most one dropdown list. Set this variable before calling display_question_canvas.
        self.checkbox_values = checkbox_values        # Option to load a custom list of answers. This can only be used once per form.

    def create_question_frame(self, question_info):
        """
        Create a question frame and display it.
        """
        frame = widgets.VBox()

        label = widgets.Label(value=question_info['question'], style={'font_weight': 'bold'})
        frame.children += (label,)

        question_type = question_info['question_type']
        default_value = question_info['default']
        if question_type == 'checkbox':
            check_vars = []
            check_widgets = []
            if self.checkbox_values == []:
                for answer, value in zip(question_info['answers'], default_value):
                    check_var = widgets.Checkbox(value=value, description=answer, indent=False)
                    check_vars.append(check_var)
                    frame.children += (check_var,)   
            else:
                for answer in self.checkbox_values:
                    check_var = widgets.Checkbox(value=False, description=answer, indent=False)
                    check_vars.append(check_var)
                    frame.children += (check_var,)   
            self.vars.append(check_vars)
        elif question_type == 'radio':
            var = widgets.RadioButtons(options=question_info['answers'], value=default_value)
            self.vars.append(var)
            frame.children += (var,)
        elif question_type == 'text':
            text_widget = widgets.Text(value=default_value, description="")
            self.vars.append(text_widget)
            frame.children += (text_widget,)
        elif question_type == 'dropdown':
            dd_widget = widgets.Dropdown(options=self.dropdown_values, disabled=False)
            self.vars.append(dd_widget)
            frame.children += (dd_widget,)
        return frame

    def get_responses(self, button):
        """
        Get responses from form when button is clicked.
        """
        self.local_file_save_result =  False
        # The responses are contained in the 'value' field of each widget.
        self.responses = []
        for var in self.vars:
            if isinstance(var, list):
                check_vars = []
                for checkvar in var:
                    check_vars.append(checkvar.value)
                self.responses.append(check_vars)
            elif isinstance(var, widgets.Text):
                self.responses.append(var.value)
            else:
                self.responses.append(var.value)
        #  
        # Insert the current responses as the defaults in the "questions" variable.
        for myint in range(len(self.questions)):
            self.questions[myint]['default'] =  self.responses[myint]
        # Create the local folder if it doesn't already exist.
        if(not os.path.isdir(self.dir_path_local)):
           os.makedirs(self.dir_path_local)
           print(f'Created the folder: {self.dir_path_local}')
        # Save a local copy of the json file with the latest update; overwrite each time the 'Submit' button is pressed.
        with open(self.dir_path_local + '/' + self.questions_file_local, 'w') as op_questions:
            json.dump(self.questions, op_questions)
        # Print the responses after the 'Submit' button is clicked.
        print(f'Form responses: "{self.responses}"')
        # Check to see if it worked.  This step proved to be harder than it should have been.
        # The function fails without producing an error if the file cannot be opened.
        # In that case, the line below is never reached.
        self.local_file_save_result = True

    def display_question_canvas(self): 
        """
        Display the form. This is the top level function.
        """
        canvas = widgets.Output(layout={'border': '1px solid black', 'height': '400px'})
        display(canvas)
        # Check if the local json file exists, and optionally open it. Otherwise open the main file.
        # This will make the local selections persistent even if we re-run the cell or restart the kernel.
        # The first step is to build the local file name from the dir_path_local global variable.
        # 'operator_name' is extracted from the first part of the directory name i.e. 'yoni_041524'.
        dir_path = os.path.abspath(self.dir_path_local)
        if sys.platform.startswith('win32'):
            directory_name = dir_path.split('\\')[-1]  # This works on Windows.
        else:
            directory_name = dir_path.split('/')[-1]  # This works on MacOS.
        operator_name = directory_name.split('_')[0]
        self.questions_file_local = f"{operator_name}_questions_latest"+".json"
        # Decide which file we are going to read.  The local file updates when new responses are entered.
        if(os.path.isfile(self.dir_path_local + "/"  + self.questions_file_local) and self.persistent):
            file_to_read = self.dir_path_local + "/" + self.questions_file_local
        else:
            file_to_read = self.questions_file_main
        # Open and read the file. 
        with open(file_to_read, 'r') as json_file:
            self.questions = json.load(json_file)
            operators = self.questions[0]["answers"]
            if operator_name.lower() not in [op.lower() for op in operators]: 
                self.questions[0]["answers"].append(operator_name.capitalize())
            self.questions[0]["default"] = operator_name.capitalize()
        # The helper function 'create_question_frame' builds the frame from the list of questions. 
        question_frames = []
        for question_info in self.questions:
            # Widgets can't handle "" as a default value, so we need to work around that problem.
            if question_info["default"] == "": 
                question_info["default"] = None
            # Make sure the default answers are boolean or None for checkbox questions.
            if question_info["question_type"] == 'checkbox':
                question_info["default"] = [(None if x == "" else x) for x in question_info["default"]]
                # This is a temporary version of the dropdown capability.  
            frame = self.create_question_frame(question_info)
            question_frames.append(frame)
        # Display the vbox.
        vbox = widgets.VBox(question_frames)
        display(vbox)
        # Display and activate the 'Submit' button.
        response_button = widgets.Button(description="Submit", button_style='success')
        response_button.on_click(self.get_responses)
        display(response_button)
        