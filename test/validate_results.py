#!/usr/bin/env python

"""Validates the test results
"""

import os
import sys
import re
import math
import cv2
import numpy as np

# Maximum difference in size of generated file vs. master before there's an error
FILE_SIZE_MAX_DIFF_FRACTION = 0.10

# How much difference is allowed before we start looking at values
# For example, a 10% allowance means we don't start looking at histogram values until bin 25
PERCENT_DIFF_ALLOWED = (5.0 / 100.0)

# Calculate the starting histogram index value
HIST_START_INDEX = int(math.ceil(256 * PERCENT_DIFF_ALLOWED))

# Number elements in a bucket that are "OK". Any bucket value above this is considered failure
# Note that for 3 channel images, a value of 100 means that about 33 pixels in any bin exceeding
# this value would cause failure
HIST_BIN_MAX = 100

# Expected folders
datasets_folder = "./datasets"
compare_folder = "./compare"

argc = len(sys.argv)
if argc <= 1:
    raise RuntimeError("Missing filename match strings parameter")
if ',' in sys.argv[1]:
    file_endings = []
    for ending in sys.argv[1].split(','):
        file_endings.append(ending.strip())
else:
    file_endings = [sys.argv[1].strip()]

# Check for a dataset filter and ensure it's not an empty string (white space counts)
dataset_filter = None
if argc > 2:
    dataset_filter = sys.argv[2]
    dataset_filter_len = len(dataset_filter)
    if dataset_filter_len <= 0:
        dataset_filter = None

def find_file_match(folder, end):
    """Locates a file in the specified folder that has the matching ending.
    Args:
        folder(str): path to the folder to look in
        end(str): the file name ending to look for
    Return:
        The path of the first file that matches the end parameter
    Notes: This function will work recursively to find the file. Each folder is fully scanned
           before its sub-folders are examined. Each subfolder is examined in depth before
           moving on to the next subfolder
    """
    if not os.path.exists(folder):
        return None
    if not os.path.isdir(folder):
        return None

    dir_list = os.listdir(folder)
    subdirs = []

    # First try to find the file. Save any sub folders for later
    for entry in dir_list:
        # Skip over hidden files
        if entry[0] == '.':
            continue

        # Check the name to see if it's a file and if it first the descrioption
        test_path = os.path.join(folder, entry)
        if os.path.isfile(test_path):
            if test_path.endswith(end):
                return test_path
        else:
            subdirs.append(entry)

    # Loop through sub folders
    subdirs_len = len(subdirs)
    if subdirs_len > 0:
        for one_dir in subdirs:
            found = find_file_match(os.path.join(folder, one_dir), end)
            if not found is None:
                return found

    return None

def find_filtered_folders(folder, regex_filter=None):
    """Finds subfolders that match the filter
    Args:
        folder(str): the folder to iterate over
        regex_filter(str): optional regular expression used to filter out subfolders
    Return:
        A list of matched folders or None if no folders were found and/or matched
    Notes:
        If regex_filter is None then all subfolders are considered a match
    """
    found = []

    for name in os.listdir(folder):
        # Skip over special folders and hidden names
        name_len = len(name)
        if name_len >= 1 and name[0] == '.':
            continue

        full_name = os.path.join(folder, name)
        if os.path.isdir(full_name):
            if not regex_filter is None:
                match = re.search(regex_filter, full_name)
                if not match is None:
                    found.append(name)
            else:
                found.append(name)

    found_len = len(found)
    return found if not found_len <= 0 else None

# Find subfolders if they're specified. Having no subfolders, with a filter specified, is not considered an error
if not dataset_filter is None:
    filtered_folders = find_filtered_folders(compare_folder, dataset_filter)
    if filtered_folders is None:
        filtered_folders = find_filtered_folders(datasets_folder, dataset_filter)
    if filtered_folders is None:
        filtered_folders = [None]
else:
    filtered_folders = [None]

# Loop through everything
filtered_folder_range = range(0, len(filtered_folders))
for one_end in file_endings:
    # If we have subfolders, we loop through those
    for folder_idx in filtered_folder_range:
        # Find the file with the correct name
        sub_folder = filtered_folders[folder_idx]

        match_folder = compare_folder if sub_folder is None else os.path.join(compare_folder, sub_folder)
        master = find_file_match(match_folder, one_end)

        match_folder = datasets_folder if sub_folder is None else os.path.join(datasets_folder, sub_folder)
        source = find_file_match(match_folder, one_end)

        print("Master file: " + str(master))
        print("Source file: " + str(source))

        if master is None:
            raise RuntimeError("Missing the comparison files used to validate results: " + str(one_end))
        if source is None:
            raise RuntimeError("Missing the resulting files from the dataset: " + str(one_end))

        # Check file sizes
        master_size = os.path.getsize(master)
        source_size = os.path.getsize(source)
        if master_size <= 0 and not source_size <= 0:
            raise RuntimeError("Generated file is not empty like comparison file: " + source + " vs " + master)
        if not master_size == 0:
            diff = abs(master_size - source_size)
            if not diff == 0 and float(diff)/float(master_size) > FILE_SIZE_MAX_DIFF_FRACTION:
                raise RuntimeError("File size difference exceeds limit of " + FILE_SIZE_MAX_DIFF_FRACTION + ": " + source + " vs " + master)
        if master_size == 0 or source_size == 0:
            print("Success compare empty files (" + one_end + "): " + source + " vs " + master)
            continue

        # Check file types
        _, ext = os.path.splitext(master)
        if not ext:
            print("Success compare extension-less files (" + one_end + "): " + source + " vs " + master)
            continue

        if not (ext == ".tif" or ext == "png"):
            print("Success. No futher tests for files (" + one_end + "): " + source + " vs " + master)
            continue

        im_mas = cv2.imread(master)
        im_src = cv2.imread(source)

        if im_mas is None:
            print("Master image was not loaded: '" + master + "'")
            exit(1)
        if im_src is None:
            print("Source image was not loaded: '" + source + "'")
            exit(1)

        # We use a dict so that we can add better error handling later if desired
        failures = {}

        # Check the image attributes
        if not im_mas.shape == im_src.shape:
            failures['image dimensions'] = True

        if 'image dimensions' not in failures:
            # calculate the differences between the images and check that
            diff = np.absolute(np.subtract(im_mas, im_src))
            hist, _ = np.histogram(diff, 256, (0, 255))

            start_idx = HIST_START_INDEX if HIST_START_INDEX < hist.size else 0
            for idx in range(start_idx, hist.size):
                if hist[idx] > HIST_BIN_MAX:
                    failures['image differences'] = True
                    break

        # Report any errors back
        failures_len = len(failures)
        if failures_len > 0:
            print("We have " + str(failures_len) + "errors detected for files (" + one_end + "): " + source + " vs " + master)
            errs = ', '.join(str(k) for k in failures.keys())
            raise RuntimeError("Errors found: %s" % errs)

print("Test has run successfully")
