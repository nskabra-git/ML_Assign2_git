"""
# Makes 'model' a package; no code needed here.

# What __init__.py is for
###############################
# In Python, any folder becomes a Python package only if it contains a file named __init__.py.
# This is part of Python’s import system.

# Why do we need a package?
################################
# Because Streamlit app uses imports like: 
# # from model.data import load_dataset;
# # from model.logistic_regression import build_model;
# These imports only work if model/ is recognized as a package.
# The file __init__.py tells Python:
# “This directory is a module/package; allow imports from it.”
# Without it, Python may give errors like:
# ModuleNotFoundError: No module named 'model'
# or Streamlit deployment failures because the structure isn’t treated as a package.

# What goes inside __init__.py?
########################################
# For your assignment, nothing is required inside it.
# An empty file is enough.
# That's why we write: # Makes 'model' a package; no code needed here.

# Why Streamlit especially needs this
############################################
# When Streamlit runs on Community Cloud:
# It copies your repo
# Runs your app as a module
# It must resolve imports inside your folder
# If there is no __init__.py, Streamlit cannot import your model code correctly.
# Adding this file ensures all these work:
# # from model.knn import build_model
# # from model.evaluate import compute_metrics
# # from model.naive_bayes import build_model

# Summary
###############
#############################################################################################################
# #      File                  #                          Purpose
#############################################################################################################
# # __init__.py                #   Tells Python that model/ is a package; required for imports.
# # Contents needed?           #   No — can be empty.
# # Why needed in your repo?   #   So Streamlit, Python, and Jupyter can import the model files correctly.
#############################################################################################################
"""
