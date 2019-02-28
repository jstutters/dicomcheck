import argparse
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pydicom

from dicomcheck import model

SCAN_PARAMS = OrderedDict([
    ((0x0018, 0x0081), 'First TE'),
    ((0x0018, 0x0081), 'Second TE'),
    ((0x0018, 0x0080), 'TR'),
    ((0x0018, 0x0082), 'TI'),
    ((0x0018, 0x0050), 'Slice thickness'),
    ((0x0018, 0x0088), 'Slice gap'),
    ((0x0018, 0x0091), 'Echo train length'),
    ((0x0018, 0x0094), 'Field of view'),
    ((0x0028, 0x0010), 'Rows'),
    ((0x0028, 0x0011), 'Columns'),
    ((0x0028, 0x0030), 'Reconstruction pixel size'),
    ((0x0018, 0x9058), 'Frequency encoding'),
    ((0x0018, 0x9231), 'Phase encoding'),
    ((0x0018, 0x0083), 'Number of averages'),
    ((0x0018, 0x1314), 'Flip angle (degress)'),
    ((0x0018, 0x1250), 'Receive Coil Name'),
    ((0x0018, 0x1251), 'Transmit Coil Name'),
    ((0x0019, 0x100a), 'Number of B0'),
    ((0x0019, 0x100c), 'B value'),
    ((0x0019, 0x1018), 'Real dwell time'),
    ((0x0018, 0x1060), 'Trigger time'),
])

PATIENT_PARAMS = OrderedDict([
    ((0x0010, 0x0010), 'Patient name'),
    ((0x0010, 0x0020), 'Patient ID'),
    ((0x0010, 0x0030), 'Patient DoB'),
    ((0x0008, 0x0022), 'Scan date'),
])

SCANNER_PARAMS = OrderedDict([
    ((0x0008, 0x0070), 'Scanner manufacturer'),
    ((0x0008, 0x1090), 'Scanner model name'),
    ((0x0018, 0x0087), 'Field strength'),
    ((0x0018, 0x1000), 'Device serial number'),
])


def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    if not args.reference_dicom_path or not args.new_dicom_path:
        parser.print_help()
        sys.exit(1)
    compare(Path(args.reference_dicom_path), Path(args.new_dicom_path))


def index_dicoms(dicom_dir):
    logging.info('Reading DICOMs in %s', dicom_dir)
    dc = model.DicomCollection()
    for f in dicom_dir.rglob("*"):
        if f.is_dir():
            continue
        logging.info('Reading %s', f)
        try:
            dcm = pydicom.read_file(str(f), stop_before_pixels=True)
            dc.add(f, dcm)
        except Exception as e:
            logging.exception(e)
            continue
    return dc


def compare(reference_dicom_dir, new_dicom_dir):
    for d in (reference_dicom_dir, new_dicom_dir):
        if not d.exists():
            print(d, " not found")
            sys.exit(1)
    first = True
    ref_dc = index_dicoms(reference_dicom_dir)
    new_dc = index_dicoms(new_dicom_dir)
    for new_patient in new_dc.patients:
        for new_study in new_dc.patients[new_patient]:
            for new_series in new_dc.patients[new_patient][new_study]:
                if new_series.modality_type != "MR" or new_series.tr is None:
                    continue
                if first:
                    output_patient(new_patient)
                    output_scanner(new_study)
                    first = False
                ref_series = ref_dc.get_series(new_series)
                output_scan(ref_series, new_series)


def output_patient(patient):
    """Print patient information."""
    print("=== Patient ===")
    print_line("Name", patient.name)
    print_line("ID", patient.id_)
    print_line("Date of birth", patient.birth_date)
    print_line("Sex", patient.sex)


def output_scanner(study):
    """Print scanner information."""
    print('=== Scanner ===')
    print_line("Scanner manufacturer", study.scanner_manufacturer)
    print_line("Scanner model", study.scanner_model_name)
    print_line("Field strength", study.field_strength)
    print_line("Device serial number", study.device_serial_number)


def print_line(description, x):
    line = '%25s: %50s'
    # use_colour = sys.stdout.isatty()
    # if use_colour:
    #     if x != y:
    #         line = '\033[1;31m' + line + '\033[1;m'
    print(line % (description, x))


def print_comparison_line(description, attr, a, b):
    line = '%25s: %50s %50s'
    x = getattr(a, attr)
    y = getattr(b, attr)
    use_colour = sys.stdout.isatty()
    if use_colour:
        if x != y:
            line = '\033[1;31m' + line + '\033[1;m'
    print(line % (description, x, y))


def slice_count(dicom_path):
    try:
        count = str(len(os.listdir(dicom_path)))
    except FileNotFoundError:
        count = ''
    return count


def deep_get(dcm, target, do_deep_search):
    if not dcm:
        return None
    el = dcm.get(target)
    try:
        val = el.value
    except AttributeError:
        val = None
    if val is not None:
        return str(val)
    if do_deep_search:
        vals = []
        recurse_tree(dcm, target, vals)
        uniques = list(set([str(e.value) for e in vals]))
        if uniques:
            return ', '.join(uniques)
        else:
            return None
    else:
        return None


def recurse_tree(dataset, target, vals):
    for data_element in dataset:
        if data_element.tag == target:
            vals.append(data_element)
        if data_element.VR == "SQ":
            for dataset in data_element.value:
                recurse_tree(dataset, target, vals)
    return vals


def make_param_set(k, dcms, do_deep_search=False):
    params = []
    for d in dcms:
        try:
            v = deep_get(d, k, do_deep_search)
        except Exception:
            continue
        else:
            if v:
                params.append(v)
    if not params:
        raise(AttributeError)
    return set(params)


def output_scan(ref_series, new_series):
    print("\n===", new_series.description, "===")
    print_comparison_line("Number", 'number', ref_series, new_series)
    print_comparison_line("Date", 'date', ref_series, new_series)
    print_comparison_line("Time", 'time', ref_series, new_series)
    print_comparison_line("Modality", 'modality_type', ref_series, new_series)
    print_comparison_line("Slice plane", 'slice_plane', ref_series, new_series)
    print_comparison_line("TR", 'tr', ref_series, new_series)
    print_comparison_line("TE", 'te', ref_series, new_series)
    print_comparison_line("TI", 'ti', ref_series, new_series)
    print_comparison_line("Slice thickness", 'slice_thickness', ref_series, new_series)
    print_comparison_line("Slice gap", 'slice_gap', ref_series, new_series)
    print_comparison_line("Echo train length", 'echo_train_length', ref_series, new_series)
    print_comparison_line("Field of view", 'field_of_view', ref_series, new_series)


def make_argument_parser():
    parser = argparse.ArgumentParser(description='MRI sequence comparison tool')
    parser.add_argument(
        'reference_dicom_path',
        metavar='REFERENCE_DICOM_PATH',
        help='Path to directory containing reference DICOMs'
    )
    parser.add_argument(
        'new_dicom_path',
        metavar='NEW_DICOM_PATH',
        help='Path to directory containing new DICOMs'
    )
    return parser


if __name__ == "__main__":
    main()
