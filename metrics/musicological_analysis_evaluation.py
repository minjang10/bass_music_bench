import json
import ast
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import argparse
import re

def extract_last_number(text):
    """
    Extracts the last integer or float from a string.
    Returns the number as int or float, or None if input is None
    or no number is found.
    """
    if text is None:
        return None

    matches = re.findall(r'-?\d+(?:\.\d+)?', str(text))
    if not matches:
        return None

    num = matches[-1]
    return float(num) if '.' in num else int(num)

def convert_to_number(text):
    """
    Converts an english number word to its numeric equivalent.
    E.g., "three" -> 3
    """
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    return number_words.get(str(text).lower(), text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Path to the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]
    
    random_baselines = {
        'ranking_genes': (1 / 24),
        'pair selection': (1 / 66),
        'multi_song_mcq_max': (1 / 4),
        'multiple choice question': (1 / 4),
        'multiple choice question (easy)': (1 / 4)
    }

    model_acc = {}
    model_std = {}
    model_ifrs = {}
    pairing_per_gene_correct = {}
    pairing_per_gene_total = {}
    total = 0
    IFR_count = 0
    IFR_total = 0
    num_correct_per_task = collections.Counter()
    total_per_task = collections.Counter()
    all_results_per_task = collections.defaultdict(list)
    with open(result_file) as f:
        model_output = json.load(f)
        for question in model_output:
            if question['prediction'] is None:
                IFR_total += 1
                total_per_task[question['task']] += 1
                all_results_per_task[question['task']].append(0.0)
                continue
            
            gt = question['gt']
            if isinstance(gt, str) and 'multiple choice question' not in question['task']:
                gt = ast.literal_eval(gt)
            pred = question['prediction']
            if question['task'] != 'multi_song_mcq_max':
                if isinstance(pred, list) and len(pred) == 4:
                    # majority voting
                    pred = str(max(set(pred), key=pred.count))
            else:
                if isinstance(pred, list) and len(pred) == 1:
                    pred = str(pred[0])
                    
            if isinstance(pred, str):
                try:
                    pred = ast.literal_eval(pred.removeprefix("```json").removesuffix("```").strip())
                except (SyntaxError, ValueError):
                    pass
            if pred is None:
                total_per_task[question['task']] += 1
                all_results_per_task[question['task']].append(0.0)
                IFR_total += 1
                continue
                    
            
            correct = False
            if "multiple choice question" in question['task'].lower():
                # Only require that the predicted answer contains the ground truth answer
                # prediction may contain additional text or reasoning etc.
                if isinstance(pred, int) or isinstance(pred, float):
                    pred = int(pred)
                    if 1 <= pred <= 4:
                        pred = question['answer_choices'][pred - 1]
                    else:
                        IFR_total += 1
                        all_results_per_task[question['task']].append(0.0)
                        total_per_task[question['task']] += 1
                        continue
                if isinstance(pred, str):
                    pred = pred.lower().strip()
                correct = (gt.lower().strip() in pred)
            elif question['task'] == 'pair selection':
                # if the prediction is a list of 2 items, check that both items are in the ground truth
                if isinstance(pred, list) and len(pred) == 2:
                    pred_lower = [p.lower().strip() for p in pred]
                    gt_lower = [g.lower().strip() for g in gt]
                    correct = sorted(pred_lower) == sorted(gt_lower)
                # if the prediction is a single string, check that both items in the ground truth are in the prediction
                # and that no other items in the answer choices are in the prediction
                elif isinstance(pred, str):
                    pred = pred.lower().strip()
                    distractors = [a.lower().strip() for a in question['answer_choices']]
                    distractors.remove(gt[0].lower().strip())
                    distractors.remove(gt[1].lower().strip())
                    correct = all(g.lower().strip() in pred for g in gt) and all(a.lower().strip() not in pred for a in distractors)
            elif question['task'] == 'multi_song_mcq_max':
                if not isinstance(pred, int):
                    pred = convert_to_number(pred)
                    pred = extract_last_number(pred)
                if not pred:
                    IFR_total += 1
                    total_per_task[question['task']] += 1
                    all_results_per_task[question['task']].append(0.0)
                    continue
                pred = int(pred)
                if 1 <= pred <= 4:
                    correct = (gt == pred)
            elif question['task'] == 'ranking_genes':
                # if prediction is a list of 4 items, check that it is the same as gt
                if isinstance(pred, list) and len(pred) == 4:
                    pred = [p.lower().strip() for p in pred if isinstance(p, str)]
                    gt_lower = [g.lower().strip() for g in gt]
                    correct = (pred == gt_lower)
                elif isinstance(pred, str):
                    # check that all 4 items in gt are in the prediction string
                    # in the correct order
                    pred = pred.lower().strip()
                    gt_lower = [g.lower().strip() for g in gt]
                    current_index = -1
                    correct = True
                    for g in gt_lower:
                        index = pred.find(g)
                        if index == -1 or index < current_index:
                            correct = False
                            break
                        current_index = index
                        
            task = question['task']
            # Merge easy and hard version of mcq into one task
            if 'multiple choice question' in task:
                task = 'multiple choice question'
            if correct:
                num_correct_per_task[task] += 1
            total_per_task[task] += 1
            all_results_per_task[task].append(1.0 if correct else 0.0)
            IFR_count += 1
            IFR_total += 1

        IFR = (IFR_count / IFR_total) * 100 if IFR_total > 0 else 0
        print(f"{dir_name} IFR: {IFR:.2f}%")
        model_ifrs[dir_name] = IFR

        acc_per_task = {
            task: (num_correct_per_task[task] / total_per_task[task]) if total_per_task[task] > 0 else 0
            for task in total_per_task
        }
        
        std_per_task = {
            task: np.std(all_results_per_task[task])
            for task in all_results_per_task
        }
        
        # Calculate and print statistics
        acc_values = list(acc_per_task.values())
        if acc_values:
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            print(f'{dir_name} - Overall: Mean={mean_acc:.3f}, StdDev={std_acc:.3f}')

        print(f'Num easy question: {total}')
        print(f'{dir_name} Accuracy per task:\n{acc_per_task}')
        model_acc[dir_name] = acc_per_task
        model_acc[dir_name + '-normalized'] = {
            task: ((acc_per_task[task] - random_baselines[task]) / (1 - random_baselines[task]))
            for task in acc_per_task
        }
        model_std[dir_name + '-normalized'] = {
            task: ((std_per_task[task] - random_baselines.get(task, 0)) / (1 - random_baselines.get(task, 0))) if (1 - random_baselines.get(task, 0)) != 0 else 0
            for task in std_per_task
        }
        model_std[dir_name] = std_per_task
    
    with open(f"{dir_name}_summary.txt", "a") as f:
        f.write(f"Musicological Analysis Evaluation Summary:\n")
        f.write(f"IFR: {model_ifrs[dir_name]:.2f}%\n")
    if model_acc[dir_name]:
        mean_acc = model_acc[dir_name]
        std_acc = model_std[dir_name]
        mean_acc_normalized = model_acc[dir_name + '-normalized']
        print(os.path.basename(dir_name), "MEAN ACC:", mean_acc)
        print(os.path.basename(dir_name), "NORMALIZED ACC", mean_acc_normalized)
        with open(f"{dir_name}_summary.txt", "a") as f:
            f.write(f"Exact Match Accuracy per Task:\n")
            for task in mean_acc:
                f.write(f"{task}: Accuracy: {mean_acc[task]:.4f}, Std Dev: {std_acc[task]:.4f}, Normalized: {mean_acc_normalized[task]:.4f}\n")

if __name__ == '__main__':
    main()