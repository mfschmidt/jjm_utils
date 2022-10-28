#!/usr/bin/env python3

import sys
import os
import xml.etree.ElementTree as ET
import argparse
import errno
import subprocess
import re
from pathlib import Path


"""Make region-specific binary masks from an atlas.

This script allows the user to specify an atlas and generate a separate
mask from each numbered region within the atlas. It can also combine those
single-ROI masks with an additional binary mask to AND them together,
resulting in a mask file for each region with only the voxels both
in-region and in the separate mask.

Usage:

    makemasks.py atlas

Running the command alone uses all defaults, which is identical to fully
specifying the following command:

    makemasks.py HarvardOxford \
                 --space MNI152NLin2009cAsym \
                 --labels long --atlas-threshold 0 --resolution 2 \
                 --output-dir .

For more, use the help.

    make_masks --help
    
"""

# TODO: If no arguments, usage
# TODO: Allow matching of mask to other resolution for later use.

class Atlas():
    """ There isn't really such a thing as an atlas for hippocampal subfields.
        Each subject has their own atlas without much of a legend.
        So, here is that legend.
    """

    def __init__(self, name, lut="default"):
        """ Create an atlas.

            By using a recognized name, like 'HBT', and nothing else,
            the Atlas object will read FreeSurfer's LUT and
            create a proper mapping of integer labels and names.
        """

        # Remember the atlas name
        self.name = name.upper()

        # Remember where to get label text
        if lut is None or lut is False:
            # Explicitly do NOT use a LUT
            self.lut = False
        elif lut is True or lut == "default":
            # Default to FreeSurfer's LUT
            self.lut = Path(os.environ["FREESURFER_HOME"]) / \
                       "FreeSurferColorLUT.txt"
        else:
            self.lut = Path(lut)

        # Use these hand-curated lists of ids available in different segs
        if self.name == "CA":
            self.labels = [
                203, 204, 205, 206, 208, 209, 211, 212, 215, 226,
            ]
        elif self.name == "HBT":
            self.labels = [
                226, 231, 232,
            ]
        elif self.name.startswith("FS6"):
            self.labels = [
                203, 204, 205, 206, 208, 209, 210, 211, 212, 214,
                215, 226,
            ]
        elif self.name.startswith("AMY"):
            self.labels = [
                7001, 7003, 7005, 7006, 7007, 7008, 7009, 7010, 7015,
            ]
        elif self.name.startswith("THAL"):
            # 27 left and 27 right nuclei in the same atlas
            # from $FREESURFER_HOME/FreeSurferColorLUT.txt
            self.labels = [
                8103, 8104, 8105, 8106, 8108, 8109, 8110, 8111, 8112,
                8113, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122,
                8123, 8125, 8126, 8127, 8128, 8129, 8130, 8133, 8134,
                8203, 8204, 8205, 8206, 8208, 8209, 8210, 8211, 8212,
                8213, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222,
                8223, 8225, 8226, 8227, 8228, 8229, 8230, 8233, 8234,
            ]
        else:  # name in ["", None, "FS7", ]:
            self.labels = [
                203, 211, 212, 215, 226, 233, 234, 235, 236, 237,
                238, 239, 240, 241, 242, 243, 244, 245, 246,
            ]

        # Use the specific ids to filter all LUT items into our atlas labels.
        self.map = self.make_map()

    def make_map(self):
        """ From a list of labels, build a label -> name LUT map.
        """

        atlas_map = {}

        # Handle situations without LUTs
        if not self.lut.exists():
            print(f"The lookup table, '{str(self.lut)}', cannot be found.")
        if not self.lut.exists() or self.lut is False:
            for lbl in self.lbls:
                atlas_map[lbl] = ""
            return atlas_map

        # Handle the expected LUT/label mapping
        regex = re.compile(r"^([\d]+)[\s]+([\S]+)[\s]+")
        with open(self.lut, "r") as f:
            for line in f:
                match = re.search(regex, line)
                if match and int(match.group(1)) in self.labels:
                    atlas_map[int(match.group(1))] = {
                        "name": match.group(2),
                        "abbr": match.group(2).replace("-", ""),
                    }

        return atlas_map


class AtlasConfig():
    """ Metadata to specify how to utilize a particular atlas
    """
    
    def __init__(self, base_path, atlas, labels, spec, res):
        """ Create an atlas configuration.
        """

        # Remember the atlas name
        self.base_path = Path(base_path).resolve()
        self.atlas = self.base_path / atlas
        self.name = self.atlas.name
        self.labels = self.base_path / labels
        self.spec = spec
        self.res = res
    

def get_arguments():
    """ Parse the command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "atlas",
        help="The atlas to use (only some are supported)",
    )
    parser.add_argument(
        "-s", "--subject", type=str, default="",
        help="The subject for whom to extract masks",
    )
    parser.add_argument(
        "-p", "--project", type=str, default="ims",
        help="the project containing masked data",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=".",
        help="""Where should the masks be written?
                default to current directory""",
    )
    parser.add_argument(
        "--space", default="MNI152NLin2009cAsym",
        help="""The MNI space to use
                MNI152NLin2009cAsym for fMRIPrep v2.20,
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
        "--atlas-threshold", default="0",
        help="""Which atlas ROI probability threshold to use for masks
                0, 25, or 50 for HarvardOxford""",
    )
    parser.add_argument(
        "--resolution", default="2",
        help="""Which atlas resolution to use for masks
                1 or 2 for HarvardOxford""",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )

    parsed_args = parser.parse_args()

    if parsed_args.subject.startswith("sub-"):
        parsed_args.subject = parsed_args.subject[4:]

    # If a subject-specific atlas is requested, ensure it exists.
    subjectives = [
        a.lower()
        for a in ["AMY", "FS7", "HBT", "CA", "FS60", "BS", "Thal", ]
    ]
    if parsed_args.atlas.lower() in subjectives:
        if parsed_args.subject == "":
            print(f"Subject is required for '{parsed_args.atlas}'.")
            parser.print_help()
            sys.exit(1)
        else:
            proj_path = Path("/mnt/derivatives/{parsed_args.project}")
            sub_path = proj_path / "freesurfer7" / f"sub-{parsed_args.subject}"
            if not sub_path.exists():
                print(f"No freesurfer data for subject '{parsed_args.subject}'"
                      f" at '{sub_path}'.")
                sys.exit(1)

    Path(parsed_args.output).mkdir(exist_ok=True)

    return parsed_args


def abbreviate_label(label):
    """ Return an abbreviation for a given label. """

    return {
        "Visual": "VIS",
        "Visual 1": "VIS1",
        "Visual 2": "VIS2",
        "Somatomotor": "SM",
        "Somatomotor 1": "SM1",
        "Somatomotor 2": "SM2",
        "Somatomotor 3": "SM3",
        "Dorsal Attention": "DAN",
        "Dorsal Attention 1": "DAN1",
        "Dorsal Attention 2": "DAN2",
        "Ventral Attention": "VAN",
        "Ventral Attention 1": "VAN1",
        "Frontoparietal": "FP",
        "Frontoparietal 1": "FP1",
        "Frontoparietal 2": "FP2",
        "Frontoparietal 3": "FP3",
        "Frontoparietal 4": "FP4",
        "Limbic": "LIM",
        "Limbic 1": "LIM1",
        "Limbic 2": "LIM2",
        "Default": "DMN",
        "Default 1": "DMN1",
        "Default 2": "DMN2",
        "Default 3": "DMN3",
        "Frontal Pole": "frPole",
        "Insular Cortex": "Ins",
        "Superior Frontal Gyrus": "supFG",
        "Middle Frontal Gyrus": "midFG",
        "Inferior Frontal Gyrus, pars triangularis": "IFGPtri",
        "Inferior Frontal Gyrus, pars opercularis": "IFGPop",
        "Precentral Gyrus": "preCG",
        "Temporal Pole": "TePole",
        "Superior Temporal Gyrus, anterior division": "antSTG",
        "Superior Temporal Gyrus, posterior division": "postSTG",
        "Middle Temporal Gyrus, anterior division": "antMTG",
        "Middle Temporal Gyrus, posterior division": "postMTG",
        "Middle Temporal Gyrus, temporooccipital part": "tempMTG",
        "Inferior Temporal Gyrus, anterior division": "antITG",
        "Inferior Temporal Gyrus, posterior division": "postITG",
        "Inferior Temporal Gyrus, temporooccipital part": "tempITG",
        "Postcentral Gyrus": "postCG",
        "Superior Parietal Lobule": "supPL",
        "Supramarginal Gyrus, anterior division": "antSMG",
        "Supramarginal Gyrus, posterior division": "postSMG",
        "Angular Gyrus": "Ang",
        "Lateral Occipital Cortex, superior division": "suplatOCort",
        "Lateral Occipital Cortex, inferior division": "inflatOCort",
        "Intracalcarine Cortex": "CalcC",
        "Frontal Medial Cortex": "medFC",
        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "suppMCort",
        "Subcallosal Cortex": "subCC",
        "Paracingulate Gyrus": "paraCG",
        "Cingulate Gyrus, anterior division": "antCG",
        "Cingulate Gyrus, posterior division": "postCG",
        "Precuneous Cortex": "preCun",
        "Cuneal Cortex": "Cun",
        "Frontal Orbital Cortex": "OFCort",
        "Parahippocampal Gyrus, anterior division": "antPHG",
        "Parahippocampal Gyrus, posterior division": "postPHG",
        "Lingual Gyrus": "LingG",
        "Temporal Fusiform Cortex, anterior division": "antTeFusC",
        "Temporal Fusiform Cortex, posterior division": "postTeFusC",
        "Temporal Occipital Fusiform Cortex": "occTeFusC",
        "Occipital Fusiform Gyrus": "occFusG",
        "Frontal Operculum Cortex": "FOper",
        "Central Opercular Cortex": "COper",
        "Parietal Operculum Cortex": "POper",
        "Planum Polare": "PlPol",
        "Heschl's Gyrus (includes H1 and H2)": "Heschl",
        "Planum Temporale": "PlTemp",
        "Supracalcarine Cortex": "SCalcC",
        "Occipital Pole": "occPole",
        "Left Cerebral White Matter": "LWhite",
        "Left Cerebral Cortex ": "LCort",  # the space is not a typo
        "Left Lateral Ventricle": "LLVent",
        "Left Thalamus": "LThal",
        "Left Caudate": "LCaud",
        "Left Putamen": "LPut",
        "Left Pallidum": "LPall",
        "Brain-Stem": "Stem",
        "Left Hippocampus": "LHip",
        "Left Amygdala": "LAmy",
        "Left Accumbens": "LAcc",
        "Right Cerebral White Matter": "RWhite",
        "Right Cerebral Cortex ": "RCort",  # the space is not a typo
        "Right Lateral Ventricle": "RLVent",
        "Right Thalamus": "RThal",
        "Right Caudate": "RCaud",
        "Right Putamen": "RPut",
        "Right Pallidum": "RPall",
        "Right Hippocampus": "RHip",
        "Right Amygdala": "RAmy",
        "Right Accumbens": "RAcc",
    }.get(label, label)


def get_fsl_labels(label_file):
    """ Return label dictionary from xml file provided. """

    import pathlib

    if pathlib.Path(label_file).is_file():
        tree = ET.parse(label_file)
        root = tree.getroot()
        labels = {}
        for label in root.findall("./data/label"):
            # Labels are one-based, not zero-, because zero is empty space.
            # But the xml files still start at zero; go figure.
            idx = int(label.attrib.get('index')) + 1
            labels[idx] = {
                "x": label.attrib.get('x'),
                "y": label.attrib.get('y'),
                "z": label.attrib.get('z'),
                "name": label.text,
                "abbr": abbreviate_label(label.text),
            }

        return labels
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), label_file
        )
    return None


def make_binary_mask(atlas, id, output_file, tool="freesurfer"):
    """ Just execute the fsl command to extract one ROI mask.
    """
    
    print(f"from {atlas.get('atlas').split('/')[-1].split('.')[0]}")

    if tool == "fsl":
        # We can do this the FSL way by extracting the id, then writing it.
        exe = Path(os.environ["FSLDIR"]) / "bin" / "fslmaths"
        fsl_extract_command = [
            str(exe.resolve()),
            str(Path(atlas["basepath"]) / atlas["atlas"]),
            "-thr", str(id), "-uthr", str(id), output_file,
        ]
        extract_proc = subprocess.run(
            fsl_extract_command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print_proc_if_needed(extract_proc)
    
        ones_command = [
            str(exe.resolve()),
            output_file, "-bin", output_file, "-odt", "char",
        ]
        ones_proc = subprocess.run(
            ones_command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print_proc_if_needed(ones_proc)

    else:  # "freesurfer"
        # We can do this the FreeSurfer way by binarizing the label.
        exe = Path(os.environ["FREESURFER_HOME"]) / "bin" / "mri_binarize"
        fs_binarize_command = [
            str(exe.resolve()),
            "--i", os.path.join(atlas["basepath"], atlas["atlas"]),
            "--o", str(output_file),
            "--match", str(id),
        ]
        binarize_result = subprocess.run(
            fs_binarize_command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print_proc_if_needed(binarize_result)


def combine_masks(mask_a, mask_b, output_path=None):
    """ Combine two binary masks AND-wise for a combination mask.
    """

    print(f"       combined with {mask_b.split('/')[-1]}.")

    # Default to overwriting the original mask.
    if output_path is None:
        output_path = mask_a
    
    exe = Path(os.environ["FSLDIR"]) / "bin" / "fslmaths"
    
    combine_command = [exe, mask_a, "-mul", mask_b, output_path, ]
    combine_proc = subprocess.run(
        combine_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    print_proc_if_needed(combine_proc)

    bin_command = [exe, output_path, "-bin", output_path, "-odt", "char", ]
    bin_proc = subprocess.run(
        bin_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    print_proc_if_needed(bin_proc)


def make_filters(configs, args):
    """ For each region in each atlas, build a binary mask.
    """

    for cfg in configs:
        print(f"Extracting masks from {cfg.label_xml[:-4]} atlas...")
        if isinstance(cfg.labels, Atlas):
            labels = cfg.labels.map
        else:
            labels = get_fsl_labels(cfg.labels)
        for id, label in labels.items():

            print(f"#{id}. {label.get('abbr')}")

            full_mask_path = Path(args.output) / \
                f"{args.atlas.lower()}_{label.get('abbr')}_mask{cfg.spec}.nii.gz"

            if args.verbose:
                print(f"  building {cfg.name} {label.get('abbr')} "
                      f"in {args.output} for subject {args.subject}.")

            # FSL does not like FreeSurfer's mgz files, so we must either
            # convert them to .nii.gz first or use FreeSurfer's tools.

            # The raw ROI mask must be created in all cases.
            out_path = full_mask_path.resolve().parent
            out_file = f"res-{cfg.res}_{full_mask_path.name}"
            make_binary_mask(cfg.atlas, out_path / out_file, id)
            
            # Only in some cases should we overwrite it with a threshold map.
            if args.contrast_map != "":
                combine_masks(out_path / out_file, args.contrast_map)


def get_labels(label_xml):
    """ Return label dictionary from xml file provided.

    :param label_xml: The path to the xml label file, usually from FSL
    :type label_xml: str
    :returns: a list of dictionaries, one for each label
    :rtype: list
    """

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


def print_proc_if_needed(proc):
    """ If a subprocess output anything, print it. """

    if len(proc.stdout) > 0:
        print("    stdout: '" + proc.stdout.decode("utf-8")  + "'")
    if len(proc.stderr) > 0:
        print("    stderr: '" + proc.stderr.decode("utf-8")  + "'")


def atlas_file(atlas, subtype, threshold, resolution):
    """ Return the filename corresponding to the arguments provided.
    """

    return "{a}/{a}-{st}-maxprob-thr{thr}-{res}mm.nii.gz".format(
        a=atlas, st=subtype, thr=threshold, res=resolution
    )


def main(args):
    """ entry point """

    fsl_basepath = os.path.join(os.environ['FSLDIR'], "data", "atlases", )
    if args.atlas.lower() == "HarvardOxford".lower():
        if args.space.lower() == "MNI152NLin6Sym".lower():
            cortical_config = AtlasConfig(
                base_path=fsl_basepath,
                atlas=atlas_file(args.atlas, "cort", args.atlas_threshold,
                                 args.resolution),
                labels="HarvardOxford-Cortical.xml",
                spec="",
                res=f"{args.resolution}mm",
            )
            subcortical_config = AtlasConfig(
                base_path=fsl_basepath,
                atlas=atlas_file(args.atlas, "sub", args.atlas_threshold,
                                 args.resolution),
                labels="HarvardOxford-Subcortical.xml",
                spec="",
                res=f"{args.resolution}mm",
            )
        elif args.space.lower() == "MNI152NLin2009cAsym".lower():
            cortical_config = AtlasConfig(
                base_path=fsl_basepath,
                atlas=atlas_file(args.atlas, "HOCPAL", args.atlas_threshold,
                                 args.resolution),
                labels="HarvardOxford-Cortical-Lateralized.xml",
                spec="",
                res=f"{args.resolution}mm",
            )
            subcortical_config = AtlasConfig(
                base_path=fsl_basepath,
                atlas=atlas_file(args.atlas, "HOSPA", args.atlas_threshold,
                                 args.resolution),
                labels="HarvardOxford-Subcortical.xml",
                spec="",
                res=f"{args.resolution}mm",
            )
        make_filters([cortical_config, subcortical_config, ], args)
    elif args.atlas.lower() == "Yeo17".lower():
        cortical_config = AtlasConfig(
            base_path=Path(".") / "atlases",
            atlas="yeo_17-network_resampled.nii.gz",
            labels="yeo_17-network.xml",
            spec=".MNI",
            res="yeo",
        )
        make_filters([cortical_config, subcortical_config, ], args)
    elif args.atlas.lower() == "Yeo07".lower():
        cortical_config = AtlasConfig(
            base_path=Path(".") / "atlases",
            atlas="yeo_07-network_resampled.nii.gz",
            labels="yeo_07-network.xml",
            spec=".MNI",
            res="yeo",
        )
        make_filters([cortical_config, subcortical_config, ], args)
    elif args.atlas.lower() in ["HBT".lower(), "CA".lower(), "FS60".lower(), ]:
        left_config = AtlasConfig(
            base_path=f"/mnt/derivatives/{args.project}/freesurfer7/"
                      f"sub-{args.subject.replace('T', 'U')}/mri",
            atlas=f"lh.hippoAmygLabels-T1-T2.v21.{args.atlas}.mgz",
            labels=Atlas(args.atlas),
            spec=".T1.lh",
            res="high",
        )
        right_config = AtlasConfig(
            base_path=f"/mnt/derivatives/{args.project}/freesurfer7/"
                      f"sub-{args.subject.replace('T', 'U')}/mri",
            atlas=f"rh.hippoAmygLabels-T1-T2.v21.{args.atlas}.mgz",
            labels=Atlas(args.atlas),
            spec=".T1.rh",
            res="high",
        )
        if args.subject == "":
            sys.exit(1)
        make_filters([left_config, right_config, ], args)
    elif args.atlas.lower() in ["FS7".lower(), "AMY".lower(), ]:
        left_config = AtlasConfig(
            base_path=f"/mnt/derivatives/{args.project}/freesurfer7/"
                      f"sub-{args.subject.replace('T', 'U')}/mri",
            atlas="lh.hippoAmygLabels-T1-T2.v21.mgz",
            labels=Atlas(args.atlas),
            spec=".T1.lh",
            res="high",
        )
        right_config = AtlasConfig(
            base_path=f"/mnt/derivatives/{args.project}/freesurfer7/"
                      f"sub-{args.subject.replace('T', 'U')}/mri",
            atlas="rh.hippoAmygLabels-T1-T2.v21.mgz",
            labels=Atlas(args.atlas),
            spec=".T1.rh",
            res="high",
        )
        make_filters([left_config, right_config, ], args)
    elif args.atlas.lower() in ["Thal".lower(), ]:
        atlas_config = AtlasConfig(
            base_path=f"/mnt/derivatives/{args.project}/freesurfer7/"
                        f"sub-{args.subject.replace('T', 'U')}/mri",
            atlas="ThalamicNuclei.v12.T1.mgz",
            labels=Atlas(args.atlas),
            spec=".T1",
            res="high",
        )
        make_filters([atlas_config, ], args)
    else:
        print("Only HarvardOxford, Yeo, "
              "and FreeSurfer hippocampal subfields "
              "are currently supported.")
        print("Only 'T1w' and two standard spaces are currently supported:")
        print("  - MNI152NLin2009cAsym: (97, 115, 97)")
        print("  - MNI152NLin6Sym: (91, 109, 91)")
        print(f"Your choice, '{args.space}', is not recognized.")
        sys.exit(1)


if __name__ == "__main__":
    main(get_arguments())
