# Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research

This repository contains two scripts that convert raw DICOM data acquired on the Hyperfine Swoop 64mT low-field scanner and Philips 3T high-field scanner to the BIDS standard (https://bids.neuroimaging.io/). The dataset comprises healthy volunteers scanned at both field strengths.

## Contents

- `lowfield_DICOM_to_BIDS.py` — for 64mT Hyperfine data
- `highfield3T_DICOM_to_BIDS.py` — for 3T Philips data
- `requirements.txt` — Python dependencies
- Example input/output structure described below

## Features

- Converts raw DICOM files to compressed NIfTI format using dicom2nifti
- Assigns BIDS-compliant filenames and folder structure
- Generates bval/bvec files from DICOM metadata or hardcoded vectors
- Anonymizes sensitive metadata and defaces anatomical scans using PyDeface
- Automatically creates dataset_description.json
- Compatible with BIDS Validator

## Requirements

Install dependencies:

pip install -r requirements.txt

## Low-Field Conversion (64mT)

### Script: lowfield_DICOM_to_BIDS.py

This script handles DICOM data from the Hyperfine Swoop 64 mT scanner.

### Usage

Update the variables at the end of the script:

all_volunteer_directory = r"path_to_all_dicom_folders"
base_output_path = r"path_to_bids_output"

Run:

python lowfield_DICOM_to_BIDS.py

### Input

Your all_volunteer_directory/ should contain one folder per participant, each with:

ABCD0_0001/
└── DICOM/
    ├── session1/
    └── session2/

You also need:
- File_used_for_ID_shuffling.json — maps original folder names to anonymized BIDS IDs
- Optional: File_to_exclude_volunteers.json

### Output

- BIDS-compliant folder structure
- .bidsignore and dataset_description.json
- bval/bvec files for diffusion scans

## High-Field Conversion (3T)

### Script: highfield3T_DICOM_to_BIDS.py

This script converts Philips 3T DICOM data, including both low-resolution and high-resolution protocols.

### Usage

Set variables inside the script:

- base_dir — path containing participant folders
- output_bids_dir — path to save BIDS-formatted output
- id_mapping_file — path to a JSON mapping raw folder names to BIDS IDs
- participant_folders — list of folders to include (e.g. ["ABCD0_0001"])

Run:

python highfield3T_DICOM_to_BIDS.py

### Input

Each participant folder must include a DICOM/ directory containing scan subfolders.

### Output

- BIDS-compliant folder structure
- Anonymized metadata
- bval/bvec files (with hardcoded vectors for b=0 and b=1000)
- dataset_description.json

Note: run-1 = b=0, run-2 = b=1000, both with fixed gradient vectors.

## Notes

- This pipeline was developed for a healthy volunteer study at LUMC.
- The dataset supports validation and benchmarking of low-field MRI.
- DICOM tags containing identifiers and dates are removed.
- Anatomical scans are defaced using PyDeface.

## Citation

If you use this code or dataset, please cite:

Ruben van den Broek, Andrew Webb, Beatrice Lena. (2025). Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research. Zenodo. DOI: [to be inserted]

## License

MIT License
