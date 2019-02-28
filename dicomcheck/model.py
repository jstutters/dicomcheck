import datetime
import logging
from collections import defaultdict

import numpy as np
from dataclasses import dataclass

from dicomcheck import dicomattributes


def _parse_time(time_attr):
    if "." in time_attr:
        parsed_val = datetime.datetime.strptime(time_attr, "%H%M%S.%f")
    else:
        parsed_val = datetime.datetime.strptime(time_attr, "%H%M%S")
    return parsed_val.time()


def _read_attr(dcm, attr, converter):
    if isinstance(attr, str):
        try:
            val = converter(getattr(dcm, attr))
        except Exception:
            val = None
    else:
        try:
            val = converter(dcm.get(attr).value)
        except Exception:
            val = None
    return val


class DicomCollection:
    patients = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def add(self, dcm_path, dcm):
        patient = Patient.from_dicom(dcm)
        study = Study.from_dicom(dcm)
        series = Series.from_dicom(dcm)
        logging.info("Adding %s, %s, %s", patient, study, series)
        self.patients[patient][study][series].append((dcm_path, dcm))

    def identify(self, nifti):
        patient = Patient.from_nifti(nifti)
        study = Study.from_nifti(nifti)
        series = Series.from_nifti(nifti)
        if patient not in self.patients or study not in self.patients[patient] or series not in self.patients[patient][study]:
            # TODO: Add a real exception
            raise Exception
        else:
            return patient, study, series

    def get_session_date(self):
        for p in self.patients:
            for s in self.patients[p]:
                d = s.date
                break
        return d

    def get_series(self, series):
        patient = list(self.patients.keys())[0]
        study = list(self.patients[patient].keys())[0]
        for my_series in self.patients[patient][study]:
            if my_series.fuzzy_match(series):
                return my_series
        else:
            return None

    def __str__(self):
        s = ""
        for patient in self.patients:
            s += str(patient) + "\n"
            for study in self.patients[patient]:
                s += "\t" + str(study) + "\n"
                for series in self.patients[patient][study]:
                    s += "\t\t" + str(series) + "\n"
                    s += "\t\t\t{0:d} files\n".format(len(self.patients[patient][study][series]))
        return s


@dataclass(frozen=True)
class Patient:
    name: str
    id_: str
    birth_date: datetime.date
    sex: str

    @classmethod
    def from_dicom(cls, dcm):
        return cls(*Patient._read_attributes(dcm))

    @classmethod
    def from_nifti(cls, nifti):
        return cls(*Patient._read_attributes(nifti))

    @staticmethod
    def _read_attributes(attributes):
        bd = datetime.datetime.strptime(attributes[dicomattributes.PatientBirthDate].value, "%Y%m%d").date()
        return (
            _read_attr(attributes, dicomattributes.PatientsName, str),
            _read_attr(attributes, dicomattributes.PatientID, str),
            bd,
            _read_attr(attributes, dicomattributes.PatientsSex, str)
        )



@dataclass(frozen=True)
class Study:
    uid: str
    date: datetime.date
    time: datetime.time
    id_: str
    description: str
    referring_physician: str
    accession_number: str
    scanner_manufacturer: str
    scanner_model_name: str
    field_strength: str
    device_serial_number: str

    @classmethod
    def from_dicom(cls, dcm):
        date = datetime.datetime.strptime(dcm.StudyDate, "%Y%m%d").date()
        time = _parse_time(dcm.StudyTime)
        return cls(
            _read_attr(dcm, "StudyInstanceUID", str),
            date,
            time,
            _read_attr(dcm, "StudyID", str),
            _read_attr(dcm, "StudyDescription", str),
            _read_attr(dcm, "ReferringPhysicianName", str),
            _read_attr(dcm, "AccessionNumber", str),
            _read_attr(dcm, "Manufacturer", str),
            _read_attr(dcm, "ManufacturerModelName", str),
            _read_attr(dcm, "MagneticFieldStrength", float),
            _read_attr(dcm, "DeviceSerialNumber", str)
        )

    @classmethod
    def from_nifti(cls, nifti):
        uid = nifti[dicomattributes.StudyInstanceUID].split()[0]
        date = datetime.datetime.strptime(nifti[dicomattributes.StudyDate], "%Y%m%d").date()
        time = _parse_time(nifti[dicomattributes.StudyTime])
        return cls(
            uid,
            date,
            time,
            nifti[dicomattributes.StudyID] or "",
            nifti[dicomattributes.StudyDescription] or "",
            nifti[dicomattributes.ReferringPhysiciansName] or "",
            nifti[dicomattributes.AccessionNumber] or ""
        )


@dataclass(frozen=True)
class Series:
    uid: str
    number: int
    description: str
    modality_type: str
    slice_plane: str
    tr: float
    te: float
    ti: float
    slice_thickness: float
    slice_gap: float
    echo_train_length: float
    field_of_view: float
    date: datetime.date
    time: datetime.time

    def fuzzy_match(self, other_series):
        return self.description == other_series.description

    @classmethod
    def from_dicom(cls, dcm):
        date = datetime.datetime.strptime(dcm.SeriesDate, "%Y%m%d").date()
        time = _parse_time(dcm.SeriesTime)
        return cls(
            _read_attr(dcm, "SeriesInstanceUID", str),
            _read_attr(dcm, "SeriesNumber", int),
            _read_attr(dcm, "SeriesDescription", str),
            _read_attr(dcm, "Modality", str),
            Series.identify_slice_plane(dcm),
            _read_attr(dcm, dicomattributes.TR, float),
            _read_attr(dcm, dicomattributes.TE, float),
            _read_attr(dcm, dicomattributes.TI, float),
            _read_attr(dcm, dicomattributes.SliceThickness, float),
            _read_attr(dcm, dicomattributes.SliceGap, float),
            _read_attr(dcm, dicomattributes.EchoTrainLength, float),
            _read_attr(dcm, dicomattributes.FieldOfView, float),
            date,
            time
        )

    @classmethod
    def from_nifti(cls, nifti):
        uid = nifti[dicomattributes.SeriesInstanceUID].split()[0]
        date = datetime.datetime.strptime(nifti[dicomattributes.SeriesDate], "%Y%m%d").date()
        time = _parse_time(nifti[dicomattributes.SeriesTime])
        return cls(
            uid,
            int(nifti[dicomattributes.SeriesNumber]),
            date,
            time,
            nifti[dicomattributes.SeriesDescription],
            nifti[dicomattributes.Modality]
        )

    @staticmethod
    def identify_slice_plane(dcm):
        try:
            slice_dir = Series.slice_direction(dcm)
        except TypeError:
            return ''
        max_el = np.argmax(np.abs(slice_dir))
        if max_el == 0:
            return 'sagittal'
        elif max_el == 1:
            return 'coronal'
        elif max_el == 2:
            return 'axial'
        else:
            raise ValueError('Maximum element not in 0-2')

    @staticmethod
    def slice_direction(dcm):
        """Calculate the slice direction from a DICOM header.

        Use the image orientation patient field to calculate the slice direction
        (as described in
        http://www.cs.ucl.ac.uk/fileadmin/cmic/Documents/DavidAtkinson/DICOM.pdf).
        If component one of the scan direction vector is high the scan is in the
        sagittal plane; high component 2 indicates the coronal plane; high
        component 3 indicates that slices are in the axial plane.
        """
        iop = [float(x) for x in dcm.get((0x0020, 0x0037))]
        iop1 = np.array(iop[0:3])
        iop2 = np.array(iop[3:])
        return np.cross(iop1, iop2)
