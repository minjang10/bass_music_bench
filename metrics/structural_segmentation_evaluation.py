from datetime import datetime
import json
# from mir_eval.segment import evaluate
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from json_repair import repair_json
import argparse

from iou_metric import calculate_iou_hungarian as evaluate

def normalize_span_output(data):
    """normalize section names by removing punctuation and converting to lowercase"""
    import string
    import re
    
    if not data:
        return data
    
    normalized = []
    for item in data:
        normalized_item = item.copy()
        
        # Get the section name
        if 'section' in normalized_item:
            section = normalized_item['section']
            
            # Remove numbers
            section = re.sub(r'\d+', '', section)
            
            # Remove punctuation except dashes
            punctuation_to_remove = string.punctuation.replace('-', '')
            section = section.translate(str.maketrans('', '', punctuation_to_remove))
            
            # Convert to lowercase
            section = section.lower()
            
            # Remove all whitespace
            section = section.replace(' ', '').replace('\t', '').replace('\n', '')
            
            normalized_item['section'] = section
        
        normalized.append(normalized_item)
    
    return normalized

def extract_intervals_labels(int_labs, section=None):
    """
    given a list of dictionaries from the model output,
    If timestamps given in "MM:SS" format, convert to seconds.
    extracts a list of intervals (ndarray: (N, 2))
    and a list of corresponding labels (list: (N,))
    """
    N = len(int_labs)
    intervals = np.ndarray((N, 2), dtype=np.float64)
    labels = []
    int_labs = normalize_span_output(int_labs)
    max_float = np.finfo(np.float64).max
    for i, section_obj in enumerate(int_labs):
        try:
            start = section_obj['start']
            end = section_obj['end']
            if ":" in str(start):
                mins, secs = start.split(":", maxsplit=1)
                start = 60 * float(mins) + float(secs)
            if ":" in str(end):
                mins, secs = end.split(":", maxsplit=1)
                end = 60 * float(mins) + float(secs)
            intervals[i] = [start, end]
        except OverflowError:
            try:
                start = float(start)
            except OverflowError:
                start = max_float
            try:
                end = float(end)
            except OverflowError:
                end = max_float
            intervals[i] = [start, end]
        except (KeyError, ValueError, TypeError) as e:
            return str(None)
        if section:
            labels.append(section.lower().strip())
        else:
            labels.append('full')
    return intervals, labels

def validate_pred(int_labs, full=False):
    """ensure that the prediction is a valid format"""
    valid_output = []
    for section in int_labs:
        try:
            if "start" not in section or "end" not in section:
                continue
            if full and "section" not in section:
                continue
            if not isinstance(section, dict):
                continue
            valid_output.append(section)
        except:
            pass
    return valid_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Path to the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]

    model_ious = {}
    model_ifrs = {}
    model_iou_std = {}
    iou_scores_per_task = collections.defaultdict(list)
    IFR_count = 0
    IFR_total = 0
    with open(result_file) as f:
        model_output = json.load(f)
        for question in model_output:
            task = question['task']
            is_section_structural_segmentation = (task == 'section_structural_segmentation')
            if question['prediction'] is None:
                IFR_total += 1
                if is_section_structural_segmentation:
                    iou_scores_per_task[question['section'].lower()].append(0.0)
                else:
                    iou_scores_per_task['full'].append(0.0)
                continue
            
            gt = question['gt']
            if is_section_structural_segmentation:
                ref = extract_intervals_labels(gt, question['section'].lower())
            else:
                ref = extract_intervals_labels(gt)
            ref_intervals, ref_labels = ref
            
            pred = question['prediction']
            pred = repair_json(pred, return_objects=True)
            if isinstance(pred, dict):
                pred = [pred]
            if not isinstance(pred, list):
                continue
            if is_section_structural_segmentation:
                curr_section = question['section'].lower()
                pred = validate_pred(pred, full=False)
                est = extract_intervals_labels(pred, curr_section)
                if est != str(None):
                    est_intervals, est_labels = est
                    IFR_count += 1
                    IFR_total += 1
                else:
                    IFR_total += 1
                    iou_scores_per_task[curr_section].append(0.0)
                    continue
            else:
                pred = validate_pred(pred, full=True)
                est = extract_intervals_labels(pred)
                if est != str(None):
                    est_intervals, est_labels = est
                    IFR_count += 1
                    IFR_total += 1
                else:
                    iou_scores_per_task['full'].append(0.0)
                    IFR_total += 1
                    continue
            iou = evaluate(ref_intervals, ref_labels, est_intervals, est_labels)[0]
            if question['task'] == 'section_structural_segmentation':
                curr_section = question['section'].lower()
                iou_scores_per_task[curr_section].append(iou)
            else:
                iou_scores_per_task['full'].append(iou)

        iou_scores = {
            section: np.mean(iou_scores_per_task[section]) if iou_scores_per_task[section] else 0
            for section in iou_scores_per_task
        }
        iou_std_devs = {
            section: np.std(iou_scores_per_task[section]) if iou_scores_per_task[section] else 0
            for section in iou_scores_per_task
        }
        model_ious[dir_name] = iou_scores
        model_iou_std[dir_name] = iou_std_devs

        print(
            f"{dir_name} Structural Segmentation iou scores by Section: {iou_scores}"
        )
        ifr = (IFR_count/IFR_total) * 100 if IFR_total > 0 else 0
        print(f"{dir_name} IFR: {ifr:.2f}%")
        model_ifrs[dir_name] = ifr

    print("\nWriting summary files...")
    with open(f"{dir_name}_summary.txt", "a") as f:
        f.write(f"Structural Analysis Evaluation Summary:\n")
        f.write(f"IFR: {model_ifrs[dir_name]:.2f}%\n\n")
    section_ious = []
    for task in model_ious[dir_name]:
        iou = model_ious[dir_name][task]
        std_dev = model_iou_std[dir_name][task]
        with open(f"{dir_name}_summary.txt", "a") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Average IoU: {iou:.4f}\n")
            f.write(f"IoU Standard Deviation: {std_dev:.4f}\n\n")
        if task != 'full':
            section_ious.append(iou)
    if section_ious:
        with open(f"{dir_name}_summary.txt", "a") as f:
            f.write(f"Average IoU for Sections: {np.mean(section_ious):.4f}\n\n")
            
if __name__ == '__main__':
    main()