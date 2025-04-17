import os
import json
import pydicom
import numpy as np
import nibabel as nib
from nibabel.arrayproxy import reshape_dataobj
from pathlib import Path
from tqdm import tqdm
import random
import re
# import orientation_correction
import subprocess
import dicom2nifti
import tempfile
import shutil

def create_dataset_description(
        output_path,
        metadata: dict
    ):
    """
    Create a dataset_description.json file for the BIDS dataset.
    """
    description = {
        "Name": metadata.get("Name"),
        "BIDSVersion": "1.7.0",
        "License": metadata.get("License", "N/A"),
        "Authors": metadata.get("Authors", []),
        "Acknowledgements": metadata.get("Acknowledgements", ""),
        "HowToAcknowledge": metadata.get("HowToAcknowledge", ""),
        "Funding": metadata.get("Funding", []),
        "ReferencesAndLinks": metadata.get("ReferencesAndLinks", []),
        "DatasetDescription": metadata.get("DatasetDescription", "No description provided."),
        "Software version": metadata.get("SoftwareVersion")
    }

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    description_file_path = output_path / 'dataset_description.json'
    with open(description_file_path, 'w') as f:
        json.dump(description, f, indent=4)

    print(f"Dataset description file created at: {description_file_path}")

def create_bidsignore(base_output_path):
    """Create a .bidsignore file in the BIDS root directory to ignore non-standard files."""
    bidsignore_path = os.path.join(base_output_path, '.bidsignore')
    
    # Add the patterns to ignore to the .bidsignore file
    bidsignore_content = """
# Ignore temporary or intermediate files
*.log
*.txt
*.tmp
*.json.tmp

# Ignore derivatives and sourcedata directories
derivatives/
sourcedata/

# Ignore localizer scans if not needed for validation
*acq-localizer*
"""
    # Write the .bidsignore file
    with open(bidsignore_path, 'w') as f:
        f.write(bidsignore_content.strip())
    
    # print(f'.bidsignore file created at {bidsignore_path}')

def is_dicom(file_path):
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False

def extract_volunteer_id(volunteer_dir, json_file_path):
    """Extract volunteer ID from directory name, convert it using the inclusion dictionary from a JSON file."""
    
    # Load the inclusion dictionary from the provided JSON file
    with open(json_file_path, 'r') as file:
        inclusion_dict = json.load(file)

    # Use regex to find the NPMR0_XXXX format
    match = re.search(r'NPMR0_\d{4}', volunteer_dir)
    if match:
        # Get the full NPMR0_XXXX string
        npmr_id = match.group(0)  # e.g., 'NPMR0_1234'
        
        # Convert to the corresponding value in the inclusion dictionary
        online_id = inclusion_dict.get(npmr_id)
        if online_id is not None:
            return str(online_id)  # Return the converted ID
        else:
            raise ValueError(f"NPMR ID '{npmr_id}' not found in inclusion dictionary.")
    else:
        raise ValueError(f"Volunteer ID not found in directory name: {volunteer_dir}")
    
def create_bids_structure(base_path, volunteer_id, session_id, scan_type):
    """Create the BIDS directory structure for a given volunteer, session, and scan type."""
    bids_path = Path(base_path) / f'sub-{volunteer_id}' / f'ses-{session_id}'
    
    if scan_type in ['DWI', 'ADC'] or scan_type.startswith('DWI_'):
        scan_path = bids_path / 'dwi'
    else:
        scan_path = bids_path / 'anat'
    
    scan_path.mkdir(parents=True, exist_ok=True)
    return scan_path

def dicom_to_nifti(volunteer_id, dicom_files, output_nifti_path):
    """
    Convert a list of DICOM files to a NIfTI file using dicom2nifti.

    Parameters:
        volunteer_id (str): Identifier for the volunteer, added to the output file name.
        dicom_files (list): List of paths to the DICOM files to be converted.
        output_nifti_path (str): Path to save the resulting NIfTI file, including the file name.
    """
    # Ensure the output directory exists
    output_folder = os.path.dirname(output_nifti_path)
    os.makedirs(output_folder, exist_ok=True)

    # Use a temporary directory to collect DICOM files for dicom2nifti
    with tempfile.TemporaryDirectory() as temp_dir:
        for dicom_file in dicom_files:
            shutil.copy(dicom_file, temp_dir)  # Copy DICOM files to the temporary directory

        # Convert the temporary DICOM folder to NIfTI
        try:
            temp_output_folder = os.path.dirname(output_nifti_path)
            dicom2nifti.convert_directory(temp_dir, temp_output_folder, compression=True)

            # Move the resulting NIfTI file to the exact output_nifti_path
            for file in os.listdir(temp_output_folder):
                if file.endswith(".nii.gz"):
                    shutil.move(os.path.join(temp_output_folder, file), output_nifti_path)
                    break

            # print(f"Converted DICOM files for {volunteer_id} to NIfTI at {output_nifti_path}")
        except Exception as e:
            raise RuntimeError(f"Error converting DICOM files for {volunteer_id}: {e}")

def generate_json_metadata(volunteer_id, dicom_files, output_json_path):
    """Generate JSON metadata from the first DICOM file, converting specific tags to volunteer ID."""
    if not dicom_files:
        raise ValueError("No DICOM files provided for metadata generation.")
    
    dicom_data = pydicom.dcmread(dicom_files[0])
    metadata = {}
    
    dicom_data.PatientName = volunteer_id  # Change Patient's Name to volunteer_id
    dicom_data.PatientID = volunteer_id    # Change Patient ID to volunteer_id

    exclude_tags = [
        (0x0008, 0x0020),  # Study Date
        (0x0008, 0x0021),  # Series Date
        (0x0008, 0x0023),  # Content Date
        (0x0008, 0x002A),  # Acquisition DateTime
        (0x0008, 0x1070),  # Operators' Name
        # (0x0010, 0x0010),  # Patient's Name
        # (0x0010, 0x0020),  # Patient ID        
        (0x0351, 1007),    # Unknown tag that contained pt ID
    ]

    for elem in dicom_data:
        if elem.tag not in exclude_tags and elem.VR != 'SQ':
            metadata[str(elem.tag)] = str(elem.value)
    
    with open(output_json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def extract_bvals_bvecs(dicom_files):
    """Extract unique b-values and b-vectors from a list of DICOM files for each volume."""
    bvals = []
    bvecs = []

    # Track unique b-value/b-vector pairs
    seen_volumes = set()

    for file in dicom_files:
        dicom_data = pydicom.dcmread(file)

        # Extract b-value and b-vector for each file
        try:
            b_value = dicom_data[0x5200, 0x9229][0][0x0018, 0x9117][0][0x0018, 0x9087].value
            mr_diff_seq = dicom_data[0x5200, 0x9229][0][0x0018, 0x9117][0]
            b_vector = mr_diff_seq[0x0018, 0x9076][0][0x0018, 0x9089].value

            # Only add unique (b_value, b_vector) pairs, one per volume
            volume_key = (b_value, tuple(b_vector))
            if volume_key not in seen_volumes:
                seen_volumes.add(volume_key)
                bvals.append(b_value if b_value else 0)
                bvecs.append(list(b_vector) if b_vector else [0, 0, 0])

        except KeyError:
            print(f"Warning: Missing diffusion data in {file}")

    return bvals, bvecs

def save_bval_bvec(bvals, bvecs, output_bval_path, output_bvec_path):
    """Save b-values and b-vectors to .bval and .bvec files."""
    # Write bvals in a single line with a newline at the end
    with open(output_bval_path, 'w') as bval_file:
        bval_file.write(' '.join(map(str, bvals)) + '\n')

    # Write bvecs as three rows (x, y, z components) with each row on a new line
    bvecs = np.array(bvecs).T
    with open(output_bvec_path, 'w') as bvec_file:
        for row in bvecs:
            bvec_file.write(' '.join(map(str, row)) + '\n')

def convert_dicom_to_bids(volunteer_id, dicom_files, output_nifti_path, output_json_path, output_bval_path=None, output_bvec_path=None):
    """Convert DICOM files to BIDS format by creating NIfTI, JSON, and optionally bval/bvec files."""

    dicom_to_nifti(volunteer_id, dicom_files, output_nifti_path)

    generate_json_metadata(volunteer_id, dicom_files, output_json_path)
    
    if output_bval_path and output_bvec_path:
        bvals, bvecs = extract_bvals_bvecs(dicom_files)
        save_bval_bvec(bvals, bvecs, output_bval_path, output_bvec_path)

def determine_scan_type(dicom_file):
    """Determine the type of scan from the DICOM file based on relevant tags."""
    dicom_data = pydicom.dcmread(dicom_file)

    # series_description = dicom_data.get((0x0018, 0x1030), None)  ### these do not always contain the useful series descrpiption values, sometimes 'Hyperfine stability scan - in Vivo - Not for Diagnostic Use'
    series_description = dicom_data.get((0x0008, 0x103E), None)

    if series_description:
        series_description = series_description.value
        if 'ADC' in series_description:
            return 'ADC'
    
    acquisition_type = dicom_data.get((0x0018, 0x0024), None)
    if acquisition_type:
        acquisition_type = acquisition_type.value
        if 'Localizer' in acquisition_type:
            return 'Localizer'
        elif 'DWI' in acquisition_type or 'DIFFUSION' in series_description or 'DWI' in series_description:
            b_value = 'unknown'
            parts = acquisition_type.split('b=')
            if len(parts) > 1:
                b_value = parts[1].split()[0].strip()
                if b_value == '0':
                    b_value = '0'
            return f'DWI'
    
    acquisition_contrast = dicom_data.get((0x0008, 0x9209), None)
    if acquisition_contrast:
        acquisition_contrast = acquisition_contrast.value
        if any(term in acquisition_contrast for term in ['T1', 'T1W', 'T1-weighted']):
            return 'T1w'
        elif any(term in acquisition_contrast for term in ['T2', 'T2W', 'T2-weighted']):
            return 'T2w'
        elif any(term in acquisition_contrast for term in ['FLAIR', 'FLUID_ATTENUATED']):
            return 'FLAIR'
    
    return 'unknown'

def list_directories(parent_directory):
    """List directories within the parent directory."""
    return next(os.walk(parent_directory))[1]

def process_scan_directory(scan_path, volunteer_id, session_id, base_output_path, run_counters):
    """Process each scan directory, keeping track of run counters for different modalities."""
    dicom_files = list(Path(scan_path).glob('*'))
    dicom_files = [f for f in dicom_files if is_dicom(f)]
    
    if not dicom_files:
        print(f"No DICOM files found in {scan_path}. Skipping...")
        return
    
    scan_type = determine_scan_type(dicom_files[0])

    if scan_type == 'unknown':
        print(f"Unknown scan type in {scan_path}. Skipping...")
        return

    # Ensure the run counter for each modality, including Localizer, increments correctly
    if scan_type in run_counters:
        run_counters[scan_type] += 1
    else:
        run_counters[scan_type] = 1
    
    run_counter = run_counters[scan_type]
    
    # Use proper capitalization for modality names
    if scan_type == 'FLAIR' or scan_type == 'ADC':
        scan_type_cap = scan_type.upper()  # Ensure both FLAIR and ADC are capitalized
    else:
        scan_type_cap = scan_type.capitalize() if scan_type in ['T1w', 'T2w'] else scan_type.lower()

    # Handle Localizer scan naming with the 'acq-localizer' suffix
    if scan_type == 'Localizer':
        scan_type_cap = 'T1w' 
        acq_suffix = 'acq-localizer'
        nifti_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_{scan_type_cap}_{acq_suffix}.nii.gz'
        json_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_{scan_type_cap}_{acq_suffix}.json'
    else:
        nifti_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_{scan_type_cap}.nii.gz'
        json_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_{scan_type_cap}.json'
    
    if scan_type.startswith('DWI'):
        bval_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_dwi.bval'
        bvec_filename = f'sub-{volunteer_id}_ses-{session_id}_run-{run_counter}_dwi.bvec'        
    else:
        bval_filename = None
        bvec_filename = None

    # Create BIDS structure for the scan
    bids_scan_path = create_bids_structure(base_output_path, volunteer_id, session_id, scan_type)
    nifti_path = bids_scan_path / nifti_filename
    json_path = bids_scan_path / json_filename
    bval_path = bids_scan_path / bval_filename if bval_filename else None
    bvec_path = bids_scan_path / bvec_filename if bvec_filename else None

    try:
        if scan_type.startswith('DWI'):
            convert_dicom_to_bids(volunteer_id, dicom_files, nifti_path, json_path, bval_path, bvec_path)
        else:
            convert_dicom_to_bids(volunteer_id, dicom_files, nifti_path, json_path)
    except Exception as e:
        print(f"Error processing {scan_path}: {e}")

def process_sessions(session_path, volunteer_id, session_id, base_output_path):
    """Process each session directory, resetting run counters for each modality."""
    scan_dirs = list_directories(session_path)

    # Initialize run counters for each modality
    run_counters = {
        'T1w': 0,
        'T2w': 0,
        'FLAIR': 0,
        'ADC': 0,
        'Localizer': 0,
        'DWI': 0,
    }
    
    with tqdm(scan_dirs, desc=f'Processing Scans for {volunteer_id} - Session {session_id}') as scan_progress:
        for scan_dir in scan_progress:
            scan_path = os.path.join(session_path, scan_dir)
            if os.path.isdir(scan_path):
                process_scan_directory(scan_path, volunteer_id, session_id, base_output_path, run_counters)

def process_volunteers(all_volunteer_directory, base_output_path):
    """Process each volunteer directory to handle scans."""

    # List all volunteer directories
    volunteer_dirs = list_directories(all_volunteer_directory)

    NPMR_key_file = r'File_used_for_ID_shuffling'
    exclude_volunteers_file = r'File_to_exclude_volunteers'

    # Load excluded volunteers if the file exists
    if os.path.isfile(exclude_volunteers_file):
        with open(exclude_volunteers_file) as exclude_file:
            exclude_volunteers = json.load(exclude_file)
        volunteer_dirs = [vd for vd in volunteer_dirs if vd not in exclude_volunteers]

    # Load included volunteers if the file exists
    if os.path.isfile(NPMR_key_file):
        with open(NPMR_key_file) as json_file:
            included_volunteers = json.load(json_file)
        volunteer_dirs = [vd for vd in volunteer_dirs if any(npmr in vd for npmr in included_volunteers)]

    exit()
    # volunteer_dirs = [item for item in volunteer_dirs if '0041' in item] ### if you want you handle just one volunteer

    metadata = {
        "Name": "In Vivo MR Neuroimaging at 64 mT: An Open Access DICOM Dataset of 65 Volunteers",
        "License": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Authors": ["Dr. Beatrice Lena", "Ruben van den Broek", "Prof. Andrew Webb"],
        "Acknowledgements": "We wish to thank all the volunteers that participated in this study.",
        "HowToAcknowledge": (
            "Ruben van den Broek, Andrew Webb, Beatrice Lena. (2025). In Vivo MR Neuroimaging at 64 mT: "
            "An Open Access DICOM Dataset of 65 Volunteers. [Dataset]. Zenodo. XX as link with DOI"
        ),
        "Funding": [
            "Partial funding for this project came from: Horizon 2020 ERC Advanced Grant PASMAR (670629), "
            "the Dutch Science Foundation Open Technology Grant number 18981, and 22HLT02 A4IM from the European Partnership "
            "on Metrology, co-financed by the European Union’s Horizon Europe Research."
        ],
        "ReferencesAndLinks": [],
        "DatasetDescription": (
            "This dataset comprises MRI scans from 65 healthy adult volunteers acquired using the Hyperfine Swoop scanner at the "
            "Leiden University Medical Center (LUMC) between November 2023 and October 2024. Some participants underwent multiple "
            "scanning sessions, resulting in a total of 85 successful sessions. The scanner operates at 64 mT and is equipped with a "
            "single-transmit, eight-receive channel coil. All scans followed the standard clinical head imaging protocol, including "
            "the following sequences: Localizer, T1-weighted, T2-weighted, Fluid Attenuated Inversion Recovery (FLAIR), and Diffusion "
            "Weighted Imaging (DWI). Details of the scans can be found in Table 1. Additionally, 11 volunteers underwent MRI scans at "
            "3T (Philips, Best) using similar sequences, following the standard clinical brain imaging protocol. These scans were acquired "
            "at two resolutions: high resolution (HR), as used in clinical practice, and low resolution (LR), matching the resolution of "
            "the low-field scans. Detailed scan parameters for the 3T acquisitions are provided in Table 2. For privacy reasons, all MRI "
            "scans have been defaced to remove facial features while preserving brain structures. No personally identifiable information "
            "is included in the dataset."
        ),
        "SoftwareVersion": ["8.8.8"]
    }

    create_dataset_description(base_output_path, metadata)
        # Create dataset_description.json in the root of the BIDS dataset

    random.shuffle(volunteer_dirs)

    with tqdm(volunteer_dirs, desc='Processing Volunteers') as volunteer_progress:
        for volunteer_dir in volunteer_progress:
            try:
                volunteer_id = extract_volunteer_id(volunteer_dir, NPMR_key_file)

                # volunteer_id = volunteer_id.replace('NPMR0_', '')  # Removes non-essential part of ID
                volunteer_output_path = os.path.join(base_output_path, f'sub-{volunteer_id}')
                os.makedirs(volunteer_output_path, exist_ok=True)

                # List existing session folders in the output path
                existing_sessions = [
                    d for d in os.listdir(volunteer_output_path)
                    if os.path.isdir(os.path.join(volunteer_output_path, d)) and d.startswith('ses-')
                ]
            
                # Extract the highest existing session number
                existing_session_numbers = [int(s.split('-')[1]) for s in existing_sessions]
                max_session_num = max(existing_session_numbers, default=0)
                
                session_dirs = list_directories(os.path.join(all_volunteer_directory, volunteer_dir, 'DICOM'))
                
                for session_idx, session_dir in enumerate(sorted(session_dirs)):
                    # Generate the next session ID based on existing sessions
                    session_id = f'{max_session_num + session_idx + 1:02d}'  # Create session ID like '01', '02', etc.
                    session_path = os.path.join(all_volunteer_directory, volunteer_dir, 'DICOM', session_dir)
                    
                    if os.path.exists(session_path):
                        process_sessions(session_path, volunteer_id, session_id, base_output_path)
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    all_volunteer_directory = r"folder_path_to_each_volunteers_DICOM"
    base_output_path = r"output_path"
    process_volunteers(all_volunteer_directory, base_output_path)
