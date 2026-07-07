# =============================================================
# src/exception.py
# ------------------------------------------------------------
# This module defines a custom exception class that provides detailed error messages.
# It includes a function to extract error details such as the filename and line number
# where the exception occurred.
# =============================================================


# =============================================
# Importing the Necessary Libraries
# --------------------------------------------
# Sys Library
## Used to handle exceptions and system-specific parameters
## Manupulate Python runtime environment
# -------------------------------------------
# Logging Module
## Used for logging error messages
## Helps in tracking events that happen when some software runs
# =============================================

import sys

# =============================================
# Function to Generate Detailed Error Messages
# --------------------------------------------
# This function takes an error and its details, extracts the filename and line number
# where the error occurred, and formats a detailed error message.
# =============================================

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script: [{0}] at line number: [{1}] with message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


# =============================================
# Custom Exception Class
# --------------------------------------------
# This class extends the built-in Exception class to provide more detailed error messages.
# It utilizes the error_message_detail function to format the error message.
# =============================================

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message   
    
  
