"""
Link to Pydicom documentation: https://pydicom.github.io/pydicom/stable/index.html
Link to useful information on the DICOM standard: https://www.leadtools.com/help/sdk/v21/dicom/api/default-value-representation-table.html
"""

import os
import filecmp
import difflib
import pydicom
import shutil
import tempfile
import numpy as np
import nibabel as nib
import dicom2nifti as d2n
from pydicom.util.codify import code_file_from_dataset
from pydicom.tag import Tag
from datetime import datetime, timedelta
from .utils import SortedDict, JsonDict
from .nifti import reorient_nifti


class DicomData(np.ndarray):
    def __init__(self):
        super(DicomData, self).__init__()


class DicomMeta(JsonDict):
    def __init__(self):
        super(DicomMeta, self).__init__()


class DicomFile(object):
    """A wrapper around a Pydicom Dataset object with additional functionalities.

    Can be instantiated from a file path or a Pydicom Dataset directly.
    """
    def __init__(self, file_path_or_pydicom_dataset, **dcmread_kwargs):
        if isinstance(file_path_or_pydicom_dataset, DicomFile):
            self.file_path = file_path_or_pydicom_dataset.file_path
            self.dicom = file_path_or_pydicom_dataset.dicom

        elif isinstance(file_path_or_pydicom_dataset, pydicom.dataset.FileDataset):
            self.file_path = None
            self.dicom = file_path_or_pydicom_dataset

        elif os.path.isfile(file_path_or_pydicom_dataset):
            self.file_path = str(file_path_or_pydicom_dataset)
            self.dicom = self.read(**dcmread_kwargs)

        else:
            assert file_path_or_pydicom_dataset is None, "Please instantiate a DicomFile with an existing file path, a Pydicom Dataset or None."
            self.file_path = None
            self.dicom = None

        self.data, self.meta = None, None

    def read(self, **dcmread_kwargs):
        assert os.path.isfile(self.file_path), "The given file path does not exist or still needs to be set."
        self.dicom = pydicom.dcmread(self.file_path, **dcmread_kwargs)
        return self.dicom

    def write(self, file_path=None, overwrite=False, copy=False, copy_as_link=False):
        if file_path is None:
            assert not copy, "For a copy, file_path must be given."
            file_path = self.file_path

        assert not os.path.isfile(file_path) or overwrite, "By default an overwrite is not allowed. If wanted specify overwrite=True."
        file_dir = os.path.dirname(file_path)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        if copy:
            assert self.file_path is not None, "A copy can only be performed if the dicomfile's file_path attribute is set."
            if copy_as_link:
                os.symlink(self.file_path, file_path)
            
            else:
                shutil.copy(self.file_path, file_path)

        else:
            self.dicom.save_as(file_path)

        new_dicom_file = DicomFile(self)  
        new_dicom_file.file_path = file_path          
        return new_dicom_file

    def get_datetime_file(self, tags=["Series", "Acquisition", "InstanceCreation", "Content"]):
        for tag in tags:
            try:
                date = str(self.dicom[tag + "Date"].value)
                time = str(self.dicom[tag + "Time"].value)
                format = "%Y%m%d_%H%M%S.%f" if "." in time else "%Y%m%d_%H%M%S"
                return datetime.strptime(f"{date}_{time}", format)
            
            except:
                continue

    def to_code(self, exclude_size=100, include_private=True, text_path_out=None):
        code = code_file_from_dataset(self.dicom, exclude_size=exclude_size, include_private=include_private)
        if text_path_out is not None:
            with open(text_path_out, "w", encoding="utf8") as f:
                f.write(code)
        
        return code
    
    def compare_with(self, other_dicom_file, **to_code_kwargs):
        code_0 = self.to_code(**to_code_kwargs).splitlines()
        code_1 = other_dicom_file.to_code(**to_code_kwargs).splitlines()
        diff = difflib.Differ().compare(code_0, code_1)
        return "\n".join([line for line in diff if line[0] in ["+", "-"]])

    def anonymize(self, new_patient_id=None, remove_private_tags=True, file_path=None):
        for data_element in self.dicom:
            if data_element.VR == "PN":
                data_element.value = "anonymized"

        if new_patient_id is not None:
            self.dicom.PatientID = new_patient_id
            self.dicom.PatientName = new_patient_id

        if remove_private_tags:
            self.dicom.remove_private_tags()

        if file_path is not None:
            self.write_dicom_file(file_path)

    def get_orientation(self):  # https://stackoverflow.com/questions/70645577/translate-image-orientation-into-axial-sagittal-or-coronal-plane
        return DicomFile.image_ori_to_str(self.dicom.ImageOrientationPatient)

    @staticmethod
    def image_ori_to_str(image_ori):
        if image_ori is None:
            return 
        
        else:
            image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
            image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])
            image_z = np.cross(image_x, image_y)
            abs_image_z = abs(image_z)
            main_index = list(abs_image_z).index(max(abs_image_z))
            if main_index == 0:
                return "sagittal"
            
            elif main_index == 1:
                return "coronal"

            else:
                return "axial"


class DicomSeries(list):
    """A list of the underlying dicom files belonging to a single series.
    
    This class adds functionalities such as dicom 2 nifti conversion, etc, to a collection of dicom files that form a series.
    """
    def __init__(self, dicom_files):
        super(DicomSeries, self).__init__()
        assert DicomSeries.is_dicom_series(dicom_files), "The given list of dicom files does not pass the 'DicomSeries.is_dicom_series' test. Please check consistency!"
        for dicom_file in dicom_files:
            self.append(DicomFile(dicom_file))
        
    def to_nifti(self, output_path=None, **dicom_series_to_nifti_kwargs):
        return DicomSeries.dicom_series_to_nifti(self, output_path=output_path, **dicom_series_to_nifti_kwargs)

    @ staticmethod
    def dicom_series_to_nifti(dicom_series, output_path=None, resample=True, validate_orthogonality=False, validate_slice_increment=False, resample_padding=0, output_orientation="LPS"):
        dicom_series = DicomSeries(dicom_series)
        if resample:
            d2n.enable_resampling()
            d2n.settings.set_resample_padding(resample_padding)
        
        if not validate_orthogonality:
            d2n.disable_validate_orthogonal()
        
        if not validate_slice_increment:
            d2n.disable_validate_slice_increment()

        tmp_dir = tempfile.TemporaryDirectory()
        if output_path is None:
            output_path = os.path.join(tmp_dir.name, "tmp_file.nii.gz")
        
        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        dicom_files = [dicom_file.read() for dicom_file in dicom_series]
        try:
            img = d2n.convert_dicom.dicom_array_to_nifti(dicom_files, output_file=output_path)["NII"]
        
        except:
            for dicom_file in dicom_files:
                # workaround due to some CBCT imaging could not be loaded since values were None and in common.py only ValueError is catched for float(value) but with None giving TypeError
                if Tag(0x0018, 0x0080) in dicom_file and Tag(0x0018, 0x0081) in dicom_file:
                    if dicom_file.RepetitionTime is None:
                        dicom_file.RepetitionTime = ""
                    
                    if dicom_file.EchoTime is None:
                        dicom_file.EchoTime = ""

            img = d2n.convert_dicom.dicom_array_to_nifti(dicom_files, output_file=output_path)["NII"]

        if output_orientation is not None:
            img = reorient_nifti(img, output_orientation=output_orientation)

        assert d2n.common.is_orthogonal_nifti(img), "There is something wrong with the orthogonality!"
        if output_path == os.path.join(tmp_dir.name, "tmp_file.nii.gz"):
            tmp_dir.cleanup()
        
        else:
            nib.save(img, output_path)

        d2n.disable_resampling()  # back to default
        d2n.settings.set_resample_padding(0)  # back to default
        d2n.enable_validate_orthogonal()  # back to default
        d2n.enable_validate_slice_increment()  # back to default
        return img

    @staticmethod
    def is_dicom_series(dicom_series):
        if len(dicom_series) > 0:
            series_instanceuid = DicomFile(dicom_series[0]).dicom.SeriesInstanceUID
            series_description = DicomFile(dicom_series[0]).dicom.get("SeriesDescription")
            series_number = DicomFile(dicom_series[0]).dicom.SeriesNumber
            for i, dicom_file in enumerate(dicom_series):
                if DicomFile(dicom_file).dicom.SeriesInstanceUID != series_instanceuid:
                    print(f"The SeriesInstanceUID of dicom file {i} does not match the one of dicom file 0.")
                    return False
                
                if DicomFile(dicom_file).dicom.get("SeriesDescription") != series_description:
                    print(f"The SeriesDescription of dicom file {i} does not match the one of dicom file 0.")
                    return False
                
                if DicomFile(dicom_file).dicom.SeriesNumber != series_number:
                    print(f"The SeriesNumber of dicom file {i} does not match the one of dicom file 0.")
                    return False

            return True


class DicomDirectory(SortedDict):
    """Recursively find all dicom files in a given root directory. 

    This class has cool funcionalities, such as sorting into series and writing to a new directory in a sorted way.
    """
    def __init__(self, root_dir=None, **dicom_file_kwargs):
        super(DicomDirectory).__init__()
        self.root_dir = root_dir
        if self.root_dir is not None:
            self.read(**dicom_file_kwargs)

    def read(self, **dicom_file_kwargs):
        assert self.root_dir is not None and os.path.isdir(self.root_dir), "The given root directory is inexistent or not a directory."
        for subdir, _, files in os.walk(self.root_dir):
            dicom_files = []
            for file in files:
                if file != "DICOMDIR":
                    file_path = os.path.join(subdir, file)
                    try:
                        dicom_files.append(DicomFile(file_path, **dicom_file_kwargs))
                    
                    except:
                        print(f"Skipped due to invalid DICOM file: {file_path}")
                        continue
            
            if len(dicom_files) > 0:
                self[subdir] = dicom_files
        
        if not self.is_sorted_into_series():
            print("WARNING: This dicom directory is not sorted into series!")

    def sort(self, tags, remove_duplicates=False):
        sorted_directory = DicomDirectory()
        for subdir in self:
            for dicom_file in self[subdir]:
                sorted_subdir = os.sep.join([str(dicom_file.dicom.get(tag, None)) for tag in tags])
                sorted_directory[sorted_subdir] = sorted_directory.get(sorted_subdir, []) + [dicom_file]
        
        if remove_duplicates:
            for subdir in sorted_directory:
                filtered_subdir = []
                for dicom_file in sorted_directory[subdir]:
                    if not any([filecmp.cmp(dicom_file.file_path, dicom_file_.file_path, shallow=False) for dicom_file_ in filtered_subdir]):
                        filtered_subdir.append(dicom_file)

                sorted_directory[subdir] = filtered_subdir

        sorted_directory.root_dir = self.root_dir
        return sorted_directory

    def sort_into_series(self, tags=["PatientName", "StudyDate", "SeriesNumber"], remove_duplicates=True):
        sorted_directory = self.sort(tags, remove_duplicates=remove_duplicates)
        assert sorted_directory.is_sorted_into_series(), "Please carefully choose the sorting tags such that the returned dicom directory is sorted into series!"
        return sorted_directory
    
    def is_sorted_into_series(self):
        for subdir in self:
            if not DicomSeries.is_dicom_series(self[subdir]):
                return False

        return True
    
    def iter_level(self, level=-1):
        dict_list, prev_iter_key = [], None
        for subdir in self:
            iter_key = os.sep.join(["*" if -i > level + 1 else k for i, k in enumerate(subdir.split(os.sep)[::-1])][::-1])
            if iter_key != prev_iter_key:
                prev_iter_key = iter_key
                dict_list.append(DicomDirectory())
                dict_list[-1].root_dir = self.root_dir

            dict_list[-1][subdir] = self[subdir]
        
        print(f"The directory was grouped into {len(dict_list)} at level {level}.")
        return dict_list

    def filter_subdirs(self, filter_fn):
        filtered_directory = DicomDirectory()
        for subdir in self:
            if filter_fn(self[subdir]):
                filtered_directory[subdir] = self[subdir]

        filtered_directory.root_dir = self.root_dir
        return filtered_directory
    
    def retain_largest(self):
        filtered_directory = DicomDirectory()
        largest_subdirs = []
        for subdir in self:
            if len(largest_subdirs) == 0 or len(self[subdir]) > len(self[largest_subdirs[0]]):
                largest_subdirs = [subdir]
            
            elif len(self[subdir]) == len(self[largest_subdirs[0]]):
                largest_subdirs.append(subdir)

        for subdir in largest_subdirs:
            filtered_directory[subdir] = self[subdir]

        filtered_directory.root_dir = self.root_dir
        return filtered_directory
    
    def reroot(self, root_dir):
        assert self.root_dir is not None and self.root_dir != "", "To reroot a DicomDirectory the root_dir attribute must be specified."
        rerooted_directory = DicomDirectory()
        for subdir in self:
            rerooted_directory[subdir.replace(self.root_dir, root_dir)] = self[subdir]

        rerooted_directory.root_dir = root_dir
        return rerooted_directory

    def write(self, **dicomfile_write_kwargs):
        assert self.root_dir is not None, "To write a DicomDirectory the root_dir attribute must be specified."
        for subdir in self:
            for i, dicom_file in enumerate(self[subdir]):
                if dicom_file.file_path is not None:
                    file_name = os.path.split(dicom_file.file_path)[1]

                else:
                    file_name = f"dicom_{i}.dcm"

                dicom_file.write(file_path=os.path.join(self.root_dir, file_name), **dicomfile_write_kwargs)

    def print_structure(self):
        for key in sorted(self.keys()):
            print(key)
