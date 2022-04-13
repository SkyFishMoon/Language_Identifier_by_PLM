import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence, Set
import jsonlines
import pyconll
from sklearn.model_selection import KFold
import re
from string import digits
import pandas as pd


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def clean_sentence(line):
    # We remove some special characters and fix small errors in the data, to improve the quality of the data
    line = line.replace("\n", '') #{"text": "- Mor.\n", "label": "da"}
    line = line.replace("- ", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("_", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("\\", '')
    line = line.replace("\"", '')
    line = line.replace("  ", " ")
    remove_digits = str.maketrans('', '', digits)
    line = line.translate(remove_digits)
    words = line.split()
    new_words = []
    # Below fixes large I instead of l. Does not catch everything, but should also not really make any mistakes either
    for word in words:
        clean_word = word
        s = clean_word
        if clean_word[1:].__contains__("I"):
            indices = find(clean_word, "I")
            for indx in indices:
                if clean_word[indx-1].islower():
                    if len(clean_word) > indx + 1:
                        if clean_word[indx+1].islower():
                            s = s[:indx] + "l" + s[indx + 1:]
                    else:
                        s = s[:indx] + "l" + s[indx + 1:]
        new_words.append(s)
    new_line = " ".join(new_words)
    return new_line


def make_jsonl_from_UD20(dataset_path: Path, langs: Set[str], writer, exact: bool, max_window: int, min_window: int, num_datapoints=1000000):
    """
    Converts CoNLL-U files to JSONL files.

    :param dataset_path: Path where UD is located
    :param langs: Set of languages to read from UD
    :param writer: jsonl writer
    :param max_window: Maximum length of non-overlapping windows.
    """

    lang_limit = {lang: 0 for lang in langs}
    for f in dataset_path.glob("*/*.conllu"):
        lang_code = f.name[:2]
        if lang_code in langs:
            counter = lang_limit[lang_code]
            if counter > num_datapoints:
                continue
            logging.info(f"Processing file: {f}")
            conll_obj = pyconll.load_from_file(f)
            for sentence in conll_obj:
                full_string = clean_sentence(sentence.text)
                writer.write({'text': full_string, 'label': lang_code})
                # windows = get_windows_from_text(full_string, max_window, exact=exact)
                # counter += len(windows)
                # if counter > num_datapoints:
                #     break
                # for w in windows:
                #     writer.write({'text': w, 'label': lang_code})
            lang_limit[lang_code] = counter

languages_to_read = "af sq ar an hy ast az ba eu bar be bn bpy bs br bg my ca ceb ce zh zh-yue cv hr cs da nl en et fi fr gl ka de el gu ht he hi hu is io id ga it ja jv kn kk ky ko la lv lt lmo nds lb mk mg ms ml mr min ne new no nn oc fa pms pl pt pa ro ru sco sr sh scn sk sl azb es su sw sv tl tg ta tt te tr uk ur uz vi vo war cy fy pnb yo th"
# languages_to_read = "da, en, sv, no, de, cs, es, fr, pt, it, tr, nl, fi, pl, ro, hu, lt, ca, hr, et"
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                    datefmt='%m/%d/%Y %H:%M:%S')
parser = ParserWithUsage()
parser.description = "Converts UD or opensubtitles data to k-splits ready to use in experiments."
parser.add_argument("--folds", help="Number of folds for cross-validation", default=10,
                    type=int)
parser.add_argument("--seed", help="Seed for the cross-fold splitting", default=42,
                    type=int)
parser.add_argument("--max-window", help="Max window size in characters", default=50,
                    type=int)
parser.add_argument("--min-window", help="Min window size in characters", default=10, type=int)
parser.add_argument("--data_source", help="Is the provided path for 'UD20' or 'opensubtitles'?", required=True,
                    type=str, )
parser.add_argument("--dataset_path", help="Path to UD directory", required=True, type=Path)
parser.add_argument("--output_path", help="Path to directory where to save the processed output",
                    type=Path, required=True)
parser.add_argument("--force", "-f", help="Whether to overwrite the output directory",
                    action="store_true")
parser.add_argument("--languages", help="The languages to be included by language code, as commaseperated string",
                    default=languages_to_read, type=str)
parser.add_argument("--exact_length",
                    help="whether the string needs to be an exact length or contain full words, this will be = min_length",
                    default=False)
parser.add_argument("--eval_length", help="how long the evaluation set should be", required=True, type=int)
args = parser.parse_args()

k: int = args.folds
seed: int = args.seed
dataset_path: Path = args.dataset_path
out_path: Path = args.output_path
force: bool = args.force
max_window: int = args.max_window
min_window: int = args.min_window
languages_to_read: set = {str(item).strip() for item in args.languages.split(' ')}
data_source: str = args.data_source
exact_length: bool = args.exact_length
evaluation_length: int = args.eval_length

logging.info("STARTED")
if out_path.exists():
    msg = f"Output path already exists: {out_path}."
    if force:
        logging.warning(
            f"{msg} Will overwrite.")
        import shutil

        shutil.rmtree(out_path)
        out_path.mkdir(exist_ok=False)
    else:
        raise ValueError(f"{msg} Use --force to overwrite")
    if out_path.is_file():
        raise ValueError(f"Output path is a file. Please provide a directory: {out_path}")
else:
    out_path.mkdir(exist_ok=False)
if not dataset_path.exists():
    raise ValueError(f"Path does not exist: {dataset_path}")

output_file = out_path / "all.jsonl"
# languages_to_read = {'da', 'en'}
meta = {}
if data_source == "UD20":
    with jsonlines.open(output_file, mode='w') as writer:
        make_jsonl_from_UD20(dataset_path, languages_to_read, writer, exact_length, max_window, min_window)
    logging.info(f"Finished writing data, splitting into {k} sections.")

# elif data_source == "opensubtitles":
#     with jsonlines.open(output_file, mode='w') as writer:
#         make_jsonl_from_opensub(dataset_path, languages_to_read, writer, exact_length, max_window, min_window)
# dataset = LangIDDataSet(output_file)

initial_data = []
lang_set= set()
lang_to_idx = {}
with jsonlines.open(output_file) as reader:
    for line in reader:
        initial_data.append(line)
        lang_set.add(line['label'])

for lang in sorted(lang_set):
    if lang not in lang_to_idx.keys():
        lang_to_idx[lang] = len(lang_to_idx)
idx_to_lang = [0 for _ in range(len(lang_to_idx))]
for key in lang_to_idx:
    idx_to_lang[lang_to_idx[key]] = key
for line in initial_data:
    line['label'] = lang_to_idx[line['label']]
data = pd.DataFrame(initial_data)
data.to_csv(out_path / "all.csv")
# region Metainformation
for key, value in vars(args).items():
    meta[key] = str(value)
meta["num_examples"] = len(initial_data)
meta["num_labels"] = len(lang_to_idx)
meta["idx_to_lang"] = idx_to_lang
logging.info("Writing meta information")
out_meta = out_path / "meta.json"
with out_meta.open(mode="w") as o:
    import json

    json.dump(meta, o, indent=4)
# endregion
# split_data_set(dataset, out_path=out_path, k=k, seed=seed)
# make_matching_split(out_path, k, evaluation_length)
logging.info("DONE")