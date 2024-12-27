import json
import os
from src.utils.graph_utils import build_graph, compare_graphs_with_success_rate
from src.utils.plan_utils import get_parsed_planner_output_from_raw
from transformers import AutoTokenizer
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TinyAgent accuracy')
    parser.add_argument('--eval_data_path', type=str, default='/home/munyeolpark/spd/SPD_tinyagent/dataset/testing_data.json')
    return parser.parse_args()

def main():
    # args = parse_args()
    f = open('/home/munyeolpark/spd/SPD_tinyagent/planner_profile.txt')
    f.readline()
    profile_data_list = f.readlines()
    f.close()

    PATH = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[0:5]) + '/'
    f = open(os.path.join(PATH, 'dataset/testing_data.json'), 'r')
    test_database = json.load(f)
    f.close()
    test_database = list(test_database.values())

    
    # tokenizer = AutoTokenizer.from_pretrained("/home/munyeolpark/spd/models/test")
    tokenizer = AutoTokenizer.from_pretrained("/home/munyeolpark/spd/models/TinyAgent-1.1B")

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    n_planner_tokens_list = []
    for profile_data in profile_data_list:
        system_prompt = profile_data.split('●')[0].encode('utf-8').decode('unicode_escape')
        human_prompt = profile_data.split('●')[1].encode('utf-8').decode('unicode_escape')

        tokens = tokenizer.tokenize(system_prompt)
        n_system_prompt_token = len(tokens)
        tokens = tokenizer.tokenize(human_prompt)
        n_human_prompt_token = len(tokens)

        n_planner_tokens_list.append([n_system_prompt_token, n_human_prompt_token])

    df_tokens = pd.DataFrame(n_planner_tokens_list, columns=['System Prompt Tokens', 'Human Prompt Tokens'])
    df_tokens.to_excel(os.path.join('/home/munyeolpark/spd/SPD_tinyagent', 'planner_tokens.xlsx'), index=False)

    
    i = 0
    j = 0
    match_table = []
    while j < len(profile_data_list):
        prompt_request = profile_data_list[j].split('●')[1].encode('utf-8').decode('unicode_escape').split('Question: ')[1]
        prompt_database = test_database[i+j]['input']
        if prompt_request != prompt_database:
            i += 1
            print(f"Mismatched in Prompt: {j}")
            continue
        match_table.append([j, i+j])
        j += 1

        


    n_matched = 0
    n_data = len(profile_data_list)
    for i, j in match_table:
        planner_label = test_database[j]['output'][0]['raw_output']
        planner_pred = profile_data_list[i].split('●')[2].encode('utf-8').decode('unicode_escape')
        # Parse plans
        pred_plan = get_parsed_planner_output_from_raw(planner_pred)
        label_plan = get_parsed_planner_output_from_raw(planner_label)
        
        # Build and compare graphs
        try:
            pred_graph = build_graph(pred_plan)
            label_graph = build_graph(label_plan)
                # Check if graphs match
            is_identical = compare_graphs_with_success_rate(pred_graph, label_graph)
            if is_identical == 1.0:
                n_matched += 1
            else:
                print('[SYSTEM] Wrong Prediction')
                print('[SYSTEM] Prediction')
                print(planner_pred)
                print('[SYSTEM] Answer')
                print(planner_label)
        except:
            print('[SYSTEM] Cannot build Planner Graph')
            print('[SYSTEM] Prediction')
            print(planner_pred)
            print('[SYSTEM] Answer')
            print(planner_label)
        
        
    
    # for profile_data, test_data in zip(profile_data_list, test_database):
    #     planner_label = test_data['output'][0]['raw_output']
    #     planner_pred = profile_data.split('●')[2].encode('utf-8').decode('unicode_escape')
    #     # Parse plans
    #     pred_plan = get_parsed_planner_output_from_raw(planner_pred)
    #     label_plan = get_parsed_planner_output_from_raw(planner_label)
        
    #     # Build and compare graphs
    #     pred_graph = build_graph(pred_plan)
    #     label_graph = build_graph(label_plan)
        
    #     # Check if graphs match
    #     is_identical = compare_graphs_with_success_rate(pred_graph, label_graph)
    #     if is_identical == 1.0:
    #         n_matched += 1

    accuracy = n_matched / n_data
    print(f"\nEvaluation Results:")
    print(f"Total examples: {n_data}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()