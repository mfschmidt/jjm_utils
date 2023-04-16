#!/usr/bin/env python3

# extract_schaefer_roi.py
#
# This script will input a subject id, and combined with other options,
# create binary masks and cropped timeseries of masked data.

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from pathlib import Path
from templateflow import api as tflow

# Trigger printing in red to highlight problems
red_on = '\033[91m'
green_on = '\033[92m'
color_off = '\033[0m'
err = f"{red_on}ERROR: {color_off}"

regions = {
    "AntTemp": "anterior temporal",
    "Aud": "auditory",
    "Cent": "central",
    "Cinga": "cingulate anterior",
    "Cingm": "mid-cingulate",
    "Cingp": "cingulate posterior",
    "ExStr": "extrastriate cortex",
    "ExStrInf": "extra-striate inferior",
    "ExStrSup": "extra-striate superior",
    "FEF": "frontal eye fields",
    "FPole": "frontal pole",
    "FrMed": "frontal medial",
    "FrOper": "frontal operculum",
    "IFG": "inferior frontal gyrus",
    "Ins": "insula",
    "IPL": "inferior parietal lobule",
    "IPS": "intraparietal sulcus",
    "OFC": "orbital frontal cortex",
    "ParMed": "parietal medial",
    "ParOcc": "parietal occipital",
    "ParOper": "parietal operculum",
    "pCun": "precuneus",
    "pCunPCC": "precuneus posterior cingulate cortex",
    "PFCd": "dorsal prefrontal cortex",
    "PFCl": "lateral prefrontal cortex",
    "PFCld": "lateral dorsal prefrontal cortex",
    "PFClv": "lateral ventral prefrontal cortex",
    "PFCm": "medial prefrontal cortex",
    "PFCmp": "medial posterior prefrontal cortex",
    "PFCv": "ventral prefrontal cortex",
    "PHC": "parahippocampal cortex",
    "PostC": "post central",
    "PrC": "precentral",
    "PrCd": "precentral dorsal",
    "PrCv": "precentral ventral",
    "RSC": "retrosplenial cortex",
    "Rsp": "retrosplenial",
    "S2": "S2",
    "SPL": "superior parietal lobule",
    "ST": "superior temporal",
    "Striate": "striate cortex",
    "StriCal": "striate calcarine",
    "Temp": "temporal",
    "TempOcc": "temporal occipital",
    "TempPar": "temporal parietal",
    "TempPole": "temporal pole",
}


def get_arguments():
    """ Parse command line arguments """

    usage_str = """
        Typical usage:

        extract_schaefer_roi.py U03280 \\
        --subjects-dir /data/BI/human/derivatives/new_conte/freesurfer7 \\
        --fmriprep-dir /data/BI/human/derivatives/new_conte/fmriprep \\
        --output-dir /data/export/home/christinam/conte_new \\
        --atlas /path/to/atlas/tpl-STUFF_dseg.nii.gz \\
        --region-names ld \\
        --verbose

        That command would use the templateflow version of the Schaefer 2018
        atlas, at 1mm resolution, downloading it if necessary. It would mask out
        the data from each ROI in the fMRIPrep BOLD and crop the first 7 TRs
        from those timeseries and the fMRIPrep confounds. It then smooths at
        5mm fwhm, hi-pass filters at 0.01Hz, and residualizes the timeseries
        with different confounds regressor collections. Finally, it saves the
        results in the output-dir.
    """

    parser = argparse.ArgumentParser(
        description="Create binary masks for a region in the Schaefer atlas.",
        usage=usage_str,
    )
    parser.add_argument(
        "subject",
        help="subject id, with or without the 'sub-'",
    )
    parser.add_argument(
        "--subjects-dir", default=None,
        help="Specify FreeSurfer SUBJECTS_DIR to override ENV variable",
    )
    parser.add_argument(
        "--fmriprep-dir", default=None,
        help="Specify fMRIPrep path, above the subject dir.",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Specify where to write output, defaults to current path.",
    )
    # This looks like it might support any atlas, but it doesn't.
    # Templateflow has MNI152 atlases, and this is for the purpose of
    # using the same 1000-parcel Schaefer atlas, transformed into T1w space.
    parser.add_argument(
        "--atlas",
        help="Specify the source atlas if overriding templateflow's MNI152-"
             "space atlas. Only Schaefer 1000-parcel atlases are supported.",
    )
    parser.add_argument(
        "--atlas-space",
        help="Use the templateflow atlas, but specify the exact space desired."
             " Using --atlas overrides this.",
    )
    parser.add_argument(
        "--region-names", nargs="+",
        help="The name of the regions in Schaefer 1000-parcel atlases",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    # Parse and validate arguments
    args = parser.parse_args()

    # Handle multiple ways to specify a subject
    if args.subject.startswith("sub-"):
        setattr(args, "subject", args.subject[4:])

    ok_to_run = True

    # Make sure we can find FreeSurfer data
    if args.subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            setattr(args, "subjects_dir", os.environ['SUBJECTS_DIR'])
        else:
            print(f"{err}No SUBJECTS_DIR in ENV, and not specified.")
            ok_to_run = False

    # Use Path objects rather than strings, and make sure they exist.
    if Path(args.subjects_dir).exists():
        print(f"Path '{args.subjects_dir}' exists.")
        setattr(args, "subjects_dir", Path(args.subjects_dir))
    else:
        print(f"{err}Path '{args.subjects_dir}' does not exist.")
        ok_to_run = False
    if Path(args.fmriprep_dir).exists():
        print(f"Path '{args.fmriprep_dir}' exists.")
        setattr(args, "fmriprep_dir", Path(args.fmriprep_dir))
    else:
        print(f"{err}Path '{args.fmriprep_dir}' does not exist.")
        ok_to_run = False
    if Path(args.output_dir).exists():
        print(f"Path '{args.output_dir}' exists.")
        setattr(args, "output_dir", Path(args.output_dir))
        setattr(args, "subject_dir",
                Path(args.output_dir) / f"sub-{args.subject}")
    else:
        print(f"{err}Path to output, '{args.output_dir}' does not exist.")
        ok_to_run = False

    # Resolve the atlas
    if "atlas" in args and args.atlas is not None:
        if Path(args.atlas).exists():
            setattr(args, "atlas_file", Path(args.atlas))
            setattr(args, "atlas_image", nib.load(args.atlas_file))
            # Our space-T1w atlas also has tpl-MNI152..., so
            # prefer getting space from 'space' key,
            # but if 'space' is not a key, 'tpl' is plan B.
            space_match = re.search(r"space-([A-Za-z0-9]+)_",
                                    args.atlas_file.name)
            if space_match:
                setattr(args, "atlas_space", space_match.group(1))
            else:
                tpl_match = re.search(r"tpl-([A-Za-z0-9]+)_",
                                      args.atlas_file.name)
                if tpl_match:
                    setattr(args, "atlas_space", tpl_match.group(1))
                else:
                    setattr(args, "atlas_space", "unknown")
        else:
            print(f"{err}Path to atlas, '{args.atlas}' does not exist.")
            ok_to_run = False
    else:
        if "atlas_space" not in args or args.atlas_space is None:
            # The safest default space:
            setattr(args, "atlas_space", "MNI152NLin2009cAsym")
        setattr(args, "atlas", get_schaefer_atlas(space=args.atlas_space))

    setattr(args, "toc", get_schaefer_toc(space="MNI152NLin2009cAsym"))

    if not ok_to_run:
        sys.exit(1)

    return args


def get_schaefer_toc(space="MNI152NLin2009cAsym"):
    """ Load the table of contents to the Schaefer atlas. """

    toc_path = tflow.get(
        space, atlas="Schaefer2018",
        desc="1000Parcels17Networks", suffix="dseg", extension="tsv"
    )
    if isinstance(toc_path, list) and len(toc_path) > 0:
        data = pd.read_csv(toc_path[0], sep='\t')
    elif isinstance(toc_path, Path):
        data = pd.read_csv(toc_path, sep='\t')
    else:
        raise FileNotFoundError(f"WTF is a {type(toc_path)}")

    return data


def get_schaefer_atlas(space="MNI152NLin2009cAsym"):
    """ Load the actual Schaefer atlas. """

    atlas_path = tflow.get(
        space, atlas="Schaefer2018",
        desc="1000Parcels17Networks", suffix="dseg", resolution=1,
        extension="nii.gz"
    )
    if isinstance(atlas_path, list) and len(atlas_path) > 0:
        data = nib.load(atlas_path[0])
    elif isinstance(atlas_path, Path):
        data = nib.load(atlas_path)
    else:
        raise FileNotFoundError(f"WTF is a {type(atlas_path)}")

    return data


def mask_from_indices(atlas, indices, region, verbose=False):
    """ Generate binary mask including all indices specified. """

    mask = np.zeros(atlas.shape, dtype=np.uint8)
    for idx in indices:
        # if verbose:
        #     print(f"Parcel {idx}: "
        #           f"{np.sum(atlas.get_fdata() == idx):,} voxels.")
        mask[atlas.get_fdata() == idx] = 1
    if verbose:
        print(f"  total for {region}: "
              f"{np.sum(mask):,} voxels across {len(indices)} parcels")
    return nib.Nifti1Image(mask, affine=atlas.affine)


def build_mask_images(args):
    """ Create three images: left, right, and bilateral versions of the ROI. """

    lh_filter = args.toc['name'].apply(lambda name: "_lh_" in name.lower())
    rh_filter = args.toc['name'].apply(lambda name: "_rh_" in name.lower())
    masks = {}
    for roi in args.region_names:
        roi_filter = args.toc['name'].apply(
            lambda name: f"_{roi.lower()}_" in name.lower()
        )
        print(f"{roi_filter.sum()} of {len(roi_filter)} parcels "
              f"from region {roi}")
        print(f"{lh_filter.sum()} of {len(lh_filter)} parcels from left, "
              f"{(lh_filter & roi_filter).sum()} in {roi}")
        print(f"{rh_filter.sum()} of {len(rh_filter)} parcels from right, "
              f"{(rh_filter & roi_filter).sum()} in {roi}")

        masks[roi] = {}
        masks[roi]['lh'] = mask_from_indices(
            args.atlas_image, args.toc[roi_filter & lh_filter]['index'],
            f"{roi}_lh", verbose=args.verbose
        )
        masks[roi]['rh'] = mask_from_indices(
            args.atlas_image, args.toc[roi_filter & rh_filter]['index'],
            f"{roi}_rh", verbose=args.verbose
        )
        masks[roi]['bi'] = mask_from_indices(
            args.atlas_image, args.toc[roi_filter]['index'],
            f"{roi}_bi", verbose=args.verbose
        )

    return masks


def main(args):
    """ Entry point """

    masks = build_mask_images(args)
    mask_dir = args.subject_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_file_template = "schaefer2018_space-{}_res-1_roi-{}_{}.nii.gz"
    for roi in sorted(masks.keys()):
        for lat in masks[roi].keys():
            filename = mask_file_template.format(args.atlas_space, roi, lat)
            nib.save(masks[roi][lat], mask_dir / filename)

    with open(args.subject_dir / "last_run.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main(get_arguments())
