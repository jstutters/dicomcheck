import argparse
from collections import defaultdict, OrderedDict
import os
import sys
import pydicom
import numpy as np

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
    if not args.dicom_path:
        parser.print_help()
        sys.exit(1)
    compare(args.dicom_path, do_deep_search=args.do_deep_search)


def compare(dicom_dir, do_deep_search=False):
    if not os.path.exists(dicom_dir):
        print('DICOM directory not found')
        sys.exit(1)

    first = True
    sequences = sorted(os.listdir(dicom_dir))
    for sequence in sequences:
        sequence_dir = os.path.join(dicom_dir, sequence)
        if not os.path.isdir(sequence_dir):
            continue
        dcms = try_to_load_dicom(sequence_dir)
        if not dcms:
            continue
        if first:
            output_patient(dcms[0])
            output_scanner(dcms[0])
            first = False
        output_scan(sequence, sequence_dir, dcms, do_deep_search=do_deep_search)


def try_to_load_dicom(sequence_dir):
    dcms = []
    try:
        files = os.listdir(sequence_dir)
    except FileNotFoundError:
        return [defaultdict(lambda: None)]
    for f in sorted(files):
        path = os.path.join(sequence_dir, f)
        try:
            dcm = pydicom.read_file(path)
        except FileNotFoundError:
            dcm = defaultdict(lambda: None)
        except pydicom.errors.InvalidDicomError:
            continue
        except IsADirectoryError:
            continue
        dcms.append(dcm)
    return dcms


def first_slice_path(sequence_dir):
    return os.path.join(sequence_dir, sorted(os.listdir(sequence_dir))[0])


def output_patient(dcm):
    print("=== Patient ===")

    for k, desc in PATIENT_PARAMS.items():
        try:
            val = dcm.get(k).value
        except AttributeError:
            val = ''
        print_line(desc, val)


def output_scanner(dcm):
    """Print scanner information."""
    print('=== Scanner ===')

    for k, desc in SCANNER_PARAMS.items():
        try:
            val = dcm.get(k).value
        except AttributeError:
            val = ''
        print_line(desc, val)


def print_line(description, x):
    line = '%25s: %50s'
    # use_colour = sys.stdout.isatty()
    # if use_colour:
    #     if x != y:
    #         line = '\033[1;31m' + line + '\033[1;m'
    print(line % (description, x))


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


def output_scan(scan_name, dcm_dir, dcm, do_deep_search=False):
    print("\n===", scan_name, "===")

    slices = slice_count(dcm_dir)
    print_line('Number of slices', slices)

    orientation = slice_plane(dcm[0])
    print_line('Slice orientation', orientation)

    for k in SCAN_PARAMS:
        try:
            val = make_param_set(k, dcm, do_deep_search=False)
        except AttributeError:
            val = ''
        if len(val) == 1:
            val = list(val)[0]
        elif len(val) > 1:
            val = ', '.join(list(val))
        print_line(SCAN_PARAMS[k], val)


def slice_plane(dcm):
    try:
        slice_dir = slice_direction(dcm)
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


def make_argument_parser():
    parser = argparse.ArgumentParser(description='MRI sequence comparison tool')
    parser.add_argument(
        'dicom_path',
        metavar='DICOM_PATH',
        help='Path to directory containing DICOMs'
    )
    parser.add_argument(
        '--deep',
        action='store_true',
        dest='do_deep_search',
        help='Do deep search for keys'
    )
    return parser


if __name__ == "__main__":
    main()
