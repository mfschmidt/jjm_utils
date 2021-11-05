#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
import argparse
import errno
import subprocess


def get_arguments():
    """ Parse the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "atlas", default="HarvardOxford",
        help="""The atlas to use
                (only HarvardOxford is supported)""",
    )
    parser.add_argument(
        "space", default="MNI152NLin2009cAsym",
        help="""The MNI space to use
                MNI152NLin2009cAsym for fMRIPrep,
                MNI152NLin6Sym for FSL and ABCD""",
    )
    parser.add_argument(
        "--contrast-map", default="",
        help="""You may specify a contrast map for combination with the atlas
                The contrast map should be binary, 1 to keep, 0 to exclude""",
    )
    parser.add_argument(
        "--labels", default="long",
        help="""Which label style to use for individual mask file names:
                'original' keeps the name from the atlas label xml
                'long' keeps the original name, but formats it without spaces
                'short' abbreviates the original name""",
    )
    parser.add_argument(
        "--atlas-threshold", default="50",
        help="""Which atlas ROI probability threshold to use for masks
                0, 25, or 50 for HarvardOxford""",
    )
    parser.add_argument(
        "--resolution", default="2",
        help="""Which atlas resolution to use for masks
                1 or 2 for HarvardOxford""",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="""Where should the masks be written?
                default to current directory""",
    )
    return parser.parse_args()


def get_labels(label_xml):
    """ Return label dictionary from xml file provided. """

    if os.path.isfile(label_xml):
        tree = ET.parse(label_xml)
        root = tree.getroot()
        labels = {}
        for label in root.findall("./data/label"):
            safe_name = label.text
            if "(" in safe_name:
                safe_name = safe_name[:safe_name.find("(")].strip()
            safe_name = safe_name.replace(" ", "_").replace("-", "_")
            safe_name = safe_name.replace("'", "")
            abbr = safe_name.replace(",", "").replace("'", "")
            abbr = abbr.replace("Left", "L").replace("Right", "R")
            abbr = abbr.replace("_division", "").replace("_part", "")
            if abbr.endswith("_"):
                abbr = abbr[:-1]
            # Labels are one-based, not zero-, because zero is empty space.
            # But the xml files still start at zero; go figure.
            idx = int(label.attrib.get('index')) + 1
            labels[idx] = {
                "x": label.attrib.get('x'),
                "y": label.attrib.get('y'),
                "z": label.attrib.get('z'),
                "original": label.text,
                "long": safe_name,
                "short": abbr,
            }

        return labels
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), label_xml
        )
    return None


def make_pure_mask(atlas, id, output_file):
    """ Just execute the fsl command to extract one ROI mask. """

    print(f"from {atlas.get('atlas')}")
    command = [
        os.path.join(os.environ["FSLDIR"], "bin", "fslmaths"),
        os.path.join(atlas["basepath"], atlas["atlas"]),
        "-thr", str(id), "-uthr", str(id),
        output_file,
    ]
    # print(" ".join(command))
    proc_mask = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def combine_masks(mask_a, mask_b, output_path):
    """ Combine two binary masks AND-wise for a combination mask. """

    print(f"combined with {mask_b}.")
    command = [
        os.path.join(os.environ["FSLDIR"], "bin", "fslmaths"),
        mask_a, "-mul", mask_b, output_path,
    ]
    # print(" ".join(command))
    proc_hotmask = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def make_filters(args, atlases):
    """ Orchestrate the creation of separate masks for each region in atlas. """

    for atlas in atlases:
        print(f"Extracting masks from {atlas['label_xml'][:-4]} atlas...")
        labels = get_labels(os.path.join(
            atlas["basepath"], atlas["label_xml"]
        ))
        for id, label in labels.items():

            print("  #{}. {}".format(id, label.get(args.labels)), end=" ")

            mask_path = os.path.join(args.output_dir, label.get(args.labels) + ".nii.gz")

            # The raw ROI mask must be created in all cases.
            make_pure_mask(atlas, id, mask_path)
            # Only in some cases should we overwrite it with a threshold map.
            if args.contrast_map != "":
                combine_masks(mask_path, args.contrast_map, mask_path)
                # Clean up old full atlas mask, no longer needed.
                # os.remove(full_mask_path)

def main(args):
    if args.atlas == "HarvardOxford":
        basepath = os.path.join(
            os.environ['FSLDIR'], "data", "atlases"
        )
        atlas_spec = f"maxprob-thr{args.atlas_threshold}-{args.resolution}mm"
        if args.space == "MNI152NLin6Sym":
            filters = [
                {
                    "basepath": basepath,
                    "atlas": f"{args.atlas}/{args.atlas}-cort-{atlas_spec}.nii.gz",
                    "label_xml": "HarvardOxford-Cortical.xml",
                },
                {
                    "basepath": basepath,
                    "atlas": f"{args.atlas}/{args.atlas}-sub-{atlas_spec}.nii.gz",
                    "label_xml": "HarvardOxford-Subcortical.xml",
                },
            ]
        elif args.space == "MNI152NLin2009cAsym":
            filters = [
                {
                    "basepath": basepath,
                    "atlas": f"{args.atlas}/{args.atlas}-HOCPAL-{atlas_spec}.nii.gz",
                    "label_xml": "HarvardOxford-Cortical-Lateralized.xml",
                },
                {
                    "basepath": basepath,
                    "atlas": f"{args.atlas}/{args.atlas}-HOSPA-{atlas_spec}.nii.gz",
                    "label_xml": "HarvardOxford-Subcortical.xml",
                },
            ]
        make_filters(args, filters)
    else:
        print("Only HarvardOxford is currently supported.")
        sys.exit(1)


if __name__ == "__main__":
    main(get_arguments())
