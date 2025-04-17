#%%
import os
import glob
import json
import pydicom
import numpy as np
# import nibabel anib
import dicom2nifti
from pathlib import Path
import subprocess
import traceback
import shutil
import tempfile

#%%

class DicomToBidsConverter:
    def __init__(self):
        # Directories
        self.base_dir = Path(r"path_containing all dicom data")

        self.output_bids_dir = Path(r"output_dir")
        
        # ID mapping file path
        self.id_mapping_file = Path(r"path_for_mapping_file")

        # Tags to clear in DICOM files
        self.tags_to_clear = [
            (0x0040, 0x0244),  # PerformedProcedureStepStartDate
            (0x0040, 0x0245),  # PerformedProcedureStepStartTime
            (0x0040, 0x0250),  # PerformedProcedureStepEndDate
            (0x0040, 0x0251),  # PerformedProcedureStepEndTime
            (0x0040, 0x2004),  # IssueDateOfImagingServiceRequest
            (0x0040, 0x2005),  # IssueTimeOfImagingServiceRequest
            (0x0008, 0x0032),  # AcquisitionTime
            (0x0008, 0x0020),  # StudyDate
            (0x0008, 0x0021),  # SeriesDate
            (0x0008, 0x0022),  # AcquisitionDate
            (0x0008, 0x0023),  # ContentDate
            (0x0008, 0x0012),  # InstanceCreationDate
            (0x0008, 0x0013),  # InstanceCreationTime
            (0x0008, 0x0033),  # PresentationCreationTime
            (0x2005, 0x140F),  # AcquisitionDateTime
            (0x0010, 0x1030),  # PatientWeight
            (0x0010, 0x0010),  # PatientName
            (0x0020, 0x000E),  # SeriesInstanceUID
            (0x0020, 0x0052),  # FrameOfReferenceUID
            (0x0008, 0x0018),  # SOPInstanceUID
            (0x0040, 0x0006),  # ScheduledPerformingPhysicianName
            (0x0008, 0x1050),  # PerformingPhysicianName
            (0x0008, 0x002A),  # AcquisitionDateTime
            (0x0008, 0x1140),  # Referenced SOP Instance UID
            (0x0008, 0x0080),  # InstitutionName
            (0x0040, 0x0275),  # ReferencedPerformedProcedureStepSequence
        ]

        # Participant folders to process. commented oud is already correct so will be skipped
        self.participant_folders = [
            'xxx',
            'yyy',
        ]

        # Load the ID mapping
        self.id_mapping = self.load_id_mapping()

    def load_id_mapping(self):
        """Load the ID mapping from the JSON file."""
        try:
            with open(self.id_mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f" !!! Error loading ID mapping: {e}")
            return {}

    def get_mapped_id(self, npmr_id):
        """Get the mapped ID for an NPMR ID from the mapping."""
        return self.id_mapping.get(npmr_id, npmr_id)

    def anonymize_dicom_data(self):
        """Clear specified DICOM tag attributes for anonymization."""
        for tag in self.tags_to_clear:
            try:
                if tag in self.dicom_data:
                    self.dicom_data[tag].value = ""
                else:
                    # print(f"Tag {tag} not found in DICOM data.")
                    pass
            except Exception as e:
                print(f"!!! Error processing tag {tag}: {e}")
        return self.dicom_data

    def find_tag_recursive(self, dataset, tag):
        """Recursively search for a tag in a DICOM dataset, including sequences."""
        if tag in dataset:
            return dataset[tag].value

        for elem in dataset.iterall():
            if elem.VR == "SQ":  # sequence
                for item in elem.value:
                    result = self.find_tag_recursive(item, tag)
                    if result is not None:
                        return result

        return None

    def determine_scan_type(self):  ### updated version, checks if slice thickness and pixel spacing exist
        """
        Determine the scan type based on SeriesNumber and ProtocolName if needed.
        """        
        
        ProtocolName = self.dicom_data.ProtocolName.lower()
        print(f'\n self.dicom_data.ProtocolName = {self.dicom_data.ProtocolName}')
        
        bval = getattr(self.dicom_data, 'DiffusionBValue', None)
        if bval is not None:
            print(f"self.DiffusionBValue = {bval}")
        else: 
            print(f"self.DiffusionBValue = does not exist")
            # print(f'self.dicom_data = {self.dicom_data}')

        if 'survey' in ProtocolName:
            return 'survey'
        
        if 'lr' in ProtocolName:
            if 'dwi' in ProtocolName:
                return 'b0_LR'
            if 'flair' in ProtocolName:
                return 'FLAIR_LR'
            if 't1' in ProtocolName:
                return 'T1W_LR'
        else:
            if 'dwi' in ProtocolName:
                return 'b0_HR'
            if 'flair' in ProtocolName:
                return 'FLAIR_HR'

            slice_thickness = getattr(self.dicom_data, 'SliceThickness', None)
            print(f'slice_thickness = {slice_thickness}')

            if 't2' in ProtocolName and slice_thickness:
                if slice_thickness > 4:
                    return 'T2W_LR'
                if slice_thickness < 4:
                    return 'T2W_HR'

            if 't1' in ProtocolName:
                return 'T1W_HR'

            pixel_spacing = getattr(self.dicom_data, 'PixelSpacing', None)
            print(f'pixel_spacing = {pixel_spacing}')            
            if 'adc' in ProtocolName and pixel_spacing:
                if pixel_spacing[0] > 1.5:
                    return 'ADC_LR'
                if pixel_spacing[0] < 1.5:
                    return 'ADC_HR'
                
        return "unknown_scan"

    def create_bids_structure(self):
        """Create the BIDS-compliant directory structure based on the scan type."""
        # Determine the folder based on the scan type
        if "adc" in self.scan_type.lower() or self.scan_type.lower().startswith("dwi") or 'b0' in self.scan_type.lower():
            scan_type_folder = "dwi"  # Place ADC and DWI scans in /dwi
        else:
            scan_type_folder = "anat"  # Place other anatomical scans in /anat
        
        subject_dir = self.output_bids_dir / f"sub-{self.mapped_id}" / scan_type_folder
        subject_dir.mkdir(parents=True, exist_ok=True)
        return subject_dir

    def generate_filenames(self):
        """Generate BIDS-compliant filenames with standard suffixes for anatomical and DWI scans."""
        scan_suffix_map = {
            "T1W_HR": "T1w", "T1W_LR": "T1w",
            "T2W_HR": "T2w", "T2W_LR": "T2w",
            "FLAIR_HR": "FLAIR", "FLAIR_LR": "FLAIR",
            "ADC_HR": "adc", "ADC_LR": "adc",
            "b0_LR": "dwi", "b1000_LR": "dwi",
            "b0_HR": "dwi", "b1000_HR": "dwi",
            "Survey": "Survey"
        }

        bids_suffix = scan_suffix_map.get(self.scan_type, "unknown")

        acq_label = ""
        if "HR" in self.scan_type:
            acq_label = "acq-highres"
        elif "LR" in self.scan_type:
            acq_label = "acq-lowres"

        self.bval_filename = None
        self.bvec_filename = None

        if 'b0' in self.scan_type or 'ADC' in self.scan_type:
            self.DiffusionBValue = self.find_tag_recursive(self.dicom_data, (0x0018, 0x9087))
            self.DiffusionGradientOrientation = self.find_tag_recursive(self.dicom_data, (0x0018, 0x9089))

            run_label = "run-1" if self.DiffusionBValue < 1 else "run-2"
            nifti_filename = f"sub-{self.mapped_id}_{acq_label}_{run_label}_{bids_suffix}.nii.gz"
            json_filename = f"sub-{self.mapped_id}_{acq_label}_{run_label}_{bids_suffix}.json"

            self.create_bval_bvec_file_paths(acq_label, run_label, bids_suffix)
        else:
            nifti_filename = f"sub-{self.mapped_id}_{acq_label}_{bids_suffix}.nii.gz"
            json_filename = f"sub-{self.mapped_id}_{acq_label}_{bids_suffix}.json"

        return nifti_filename, json_filename
    
    # def create_bval_bvec_file_paths(self, acq_label,run_label, bids_suffix):
    #     print(f'self.DiffusionBValue = {self.DiffusionBValue}')

    #     self.bval_filename = f'sub-{self.mapped_id}_{acq_label}_{run_label}_{bids_suffix}.bval'
    #     self.bvec_filename = f'sub-{self.mapped_id}_{acq_label}_{run_label}_{bids_suffix}.bvec'          

    # def create_bval_bvec_files(self, bval_path, bvec_path):
    #     with open(bval_path, 'w') as bval_file:
    #         bval_file.write(f"{self.DiffusionBValue}\n")  # Ensure it's a single value

    #     self.DiffusionGradientOrientation = [1, 1, 1]  # Keep the hardcoded vector
    #     with open(bvec_path, 'w') as bvec_file:
    #         for row in self.DiffusionGradientOrientation:  # Iterate over list
    #             bvec_file.write(f"{row}\n")  # Write each value on a new line

    def create_bval_bvec_files(self, bval_path, bvec_path):
        """
        Create .bval and .bvec files for diffusion data.
        Uses b=0 (bvec [0 0 0]) for run-1 and b=1000 (bvec [1 1 1]) for run-2,
        based on file naming convention.
        """
        ### Since the 3T metadata is not always stored consistently, the .bvec and .bval files are created like this. 
        if 'run-1' in os.path.basename(bvec_path):
            bval_value = 0
            bvec_array = np.array([[0], [0], [0]])
        elif 'run-2' in os.path.basename(bvec_path):
            bval_value = 1000
            bvec_array = np.array([[1], [1], [1]])
        else:
            bval_value = self.DiffusionBValue  # fallback
            bvec_array = np.array([[1], [1], [1]])

        with open(bval_path, 'w') as bval_file:
            bval_file.write(f"{bval_value}\n")

        np.savetxt(bvec_path, bvec_array, fmt='%d', delimiter=' ')

        print(f"Created bval = {bval_value} at {bval_path}")
        print(f"Created bvec = {bvec_array.T.tolist()[0]} at {bvec_path}")    


    def find_dicom_files(self, dicom_dir):
        """Recursively find all DICOM files in the specified directory."""
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Attempt to read the file to confirm it is a DICOM
                    self.dicom_data = pydicom.dcmread(file_path, stop_before_pixels=True)
                    dicom_files.append(file_path)
                except pydicom.errors.InvalidDicomError:
                    continue  # Skip files that are not DICOM
        return dicom_files

    def clean_value(self, value):
        """Convert DICOM element value into a JSON-friendly format."""
        if isinstance(value, pydicom.sequence.Sequence):
            return [self.clean_value(item) for item in value]
        elif isinstance(value, pydicom.dataset.Dataset):
            return {element.keyword if element.keyword else f"({element.tag.group:04X},{element.tag.element:04X})": self.clean_value(element.value)
                    for element in value}
        elif isinstance(value, (list, tuple)):
            return [self.clean_value(item) for item in value]
        elif isinstance(value, bytes):
            try:
                return value.decode()
            except Exception:
                return str(value)
        elif isinstance(value, (float, int, str)):
            return value
        else:
            return str(value)

    def dicom_to_nifti(self, dicom_files, output_nifti_path, bval_path, bvec_path):  ### old method, works on 9 of 11 sets
        """
        Convert DICOM files to NIfTI format, forcing overwrite and avoiding temp files.

        Parameters:
            dicom_files (list): List of paths to the DICOM files.
            output_nifti_path (str): Path to save the resulting NIfTI file.
        """
        output_folder = os.path.dirname(output_nifti_path)
        os.makedirs(output_folder, exist_ok=True)

        if bval_path and bvec_path:
            ### b-values exist so the files should be made
            self.create_bval_bvec_files(bval_path, bvec_path)

        try:
            # Convert DICOM directly to the output folder
            dicom2nifti.convert_directory(os.path.commonpath(dicom_files), output_folder, compression=True)

            # Locate the generated file
            nifti_files = [f for f in os.listdir(output_folder) if f.endswith(".nii.gz")]
            if not nifti_files:
                raise RuntimeError("No NIfTI file was created.")

            generated_nifti_path = os.path.join(output_folder, nifti_files[0])

            os.rename(generated_nifti_path, output_nifti_path)
        except Exception as e:
            print("Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Error converting DICOM files: {e}")

    def save_metadata(self, json_path):
        """Save all DICOM metadata to a JSON file."""
        metadata = {}
        for element in self.dicom_data:
            name = element.keyword if element.keyword else f"({element.tag.group:04X},{element.tag.element:04X})"
            try:
                metadata[name] = self.clean_value(element.value)
            except Exception as e:
                metadata[name] = f"Error processing value: {e}"
                print("Full traceback:")
                traceback.print_exc()                

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def process_scan_folder(self, scan_dir):
        """Process a single scan folder within a participant's DICOM directory."""

        dicom_files = self.find_dicom_files(scan_dir)
        if not dicom_files:
            print(f"No valid DICOM files found in {scan_dir}. Skipping this scan.")
            return

        # Read the first DICOM file in the directory
        self.dicom_data = pydicom.dcmread(dicom_files[0])

        # Print a confirmation for debugging
        print(f'Processing DICOM data for scan directory: {scan_dir}')

        self.scan_type = self.determine_scan_type()

        print(f'self.scan_type = {self.scan_type }, scan_dir = {scan_dir}')

        if self.scan_type in ['Survey', 'unknown_scan', 'survey', 'Survey MST']:
            print(f"Skipping scan folder {scan_dir} due to scan type: {self.scan_type}")
            return  # Exit function and move to the next iteration in process_participant_folder

        self.dicom_data = self.anonymize_dicom_data()

        # Create output paths and filenames
        bids_path = self.create_bids_structure()
        nifti_filename, json_filename = self.generate_filenames()

        nifti_path = bids_path / nifti_filename
        json_path = bids_path / json_filename
        bval_path = bids_path / self.bval_filename if self.bval_filename else None
        bvec_path = bids_path / self.bvec_filename if self.bvec_filename else None        

        self.dicom_to_nifti(dicom_files, nifti_path, bval_path, bvec_path)
        
        self.save_metadata(json_path)

    def process_participant_folder(self, npmr_id):
        """Process each scan series folder in a participant's DICOM directory."""
        self.mapped_id = self.get_mapped_id(npmr_id)
        dicom_base_dir = os.path.join(self.base_dir, npmr_id, "DICOM")
        
        print(f" \n Processing participant {npmr_id} (mapped ID: {self.mapped_id}) in {dicom_base_dir}...")

        for root, dirs, _ in os.walk(dicom_base_dir):
            # Only process folders with no subdirectories, assumed to contain DICOM files for a single scan
            if not dirs:
                try:
                    self.process_scan_folder(root)

                except Exception as e:
                    print(f"!!!! Error processing scan folder {root} for participant {npmr_id} (mapped ID: {self.mapped_id}): {e}")
                    print("Full traceback:")
                    traceback.print_exc()
            
    def create_and_save_dataset_description(self, output_path):
        """
        Create and save a dataset_description.json file for a BIDS dataset.

        Parameters:
            output_path (str or Path): Directory where the dataset_description.json file will be saved.
        """
        description = {
            "Name": "LUMC 3T Healthy Volunteers Dataset",
            "BIDSVersion": "1.7.0",
            "License": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
            "Authors": [
    "Dr. Beatrice Lena",
    "Ruben van den Broek",
    "Prof. Andrew Webb"
            ],
            "Acknowledgements": "We wish to thank all the volunteers that participated in this study.",
            "HowToAcknowledge": "Instructions for acknowledging the dataset",
            "Funding": [
    "Partial funding for this project came from: Horizon 2020 ERC Advanced Grant PASMAR (670629), the Dutch Science Foundation Open Technology Grant number 18981, and 22HLT02 A4IM  from the European Partnership on Metrology, co-financed by the European Union’s Horizon Europe Research."
            ],
            "ReferencesAndLinks": [],
            "DatasetDescription": "This dataset contains 3T MRI scans of healthy volunteers scanned at LUMC between March 2024 and October 2024. Scans include Localizer, T1w, T2w, FLAIR, and DWI. All are scanned once in high resolution and once in low resolution.",
            # "SoftwareVersion": "8.8.8"
        }

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        description_file_path = output_path / 'dataset_description.json'

        with open(description_file_path, 'w') as f:
            json.dump(description, f, indent=4)

        print(f"Dataset description file created at: {description_file_path}")

    def run(self):
        """Process each original `NPMR` ID in the participant folder list."""

        for npmr_id in self.participant_folders:
            self.process_participant_folder(npmr_id)

        self.create_and_save_dataset_description( output_path=self.output_bids_dir)        
        print("Processing complete for all participants.")

#%%

# Entry point for script execution
if __name__ == "__main__":
    converter = DicomToBidsConverter()
    converter.run()

