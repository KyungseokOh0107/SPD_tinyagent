import json
import os
import requests
import pandas as pd

PATH = '/home/munyeolpark/spd/TinyAgent/'
# 0: Planner System Prompt o, Agent System Prompt o
# 1: Planner System Prompt x, Agent System Prompt x
# 2: Planner System Prompt x, Agent System Prompt o
# 3: Planner System Prompt o, Agent System Prompt x
# Baseline Setting: EXPERIMENT_ID 0, SYNC_GENERATION_TOKEN False
# Upperlimit Setting: EXPERIMENT_ID 1, SYNC_GENERATION_TOKEN True

EXPERIMENT_ID = 0
SYNC_GENERATION_TOKEN = False
USE_PLANNER_ANSWER = True



if os.path.exists(PATH + 'overall_result.txt'):
    os.remove(PATH + 'overall_result.txt')

if os.path.exists(PATH + 'tool_time.txt'):
    os.remove(PATH + 'tool_time.txt')


if SYNC_GENERATION_TOKEN:
    data = pd.read_csv(PATH + 'overall_result_baseline.txt')
    generation_token_list_db = data[['Planner Output Token', 'Agent Output Token']].values.astype('int')

f = open(os.path.join(PATH, 'dataset/training_data.json'), 'r')
dataset = json.load(f)
f.close()
dataset = list(dataset.values())

for i, data in enumerate(dataset[0:2000]):
    query = data['input']

    if USE_PLANNER_ANSWER:
        planner_answer = data['output'][0]['raw_output']
    else:
        planner_answer = 'None'

    if SYNC_GENERATION_TOKEN:
        generation_token_list = generation_token_list_db[i].tolist()
    else:
        generation_token_list = [0, 0]
    # import pdb;pdb.set_trace()
    # response = requests.post('http://127.0.0.1:51000/generate_experiment', json={'query': query, 'path': PATH, 'experiment_id': EXPERIMENT_ID, 'planner_answer': planner_answer, 'generation_token_list': generation_token_list})
    response = requests.post('http://127.0.0.1:51000/generate_experiment', json={'query': query, 'path': PATH, 'experiment_id': EXPERIMENT_ID, 'planner_answer': planner_answer, 'generation_token_list': generation_token_list})


if EXPERIMENT_ID == 0:
    os.rename(PATH + 'overall_result.txt', PATH + 'overall_result_baseline.txt')
    os.rename(PATH + 'tool_time.txt', PATH + 'tool_time_baseline.txt')
elif EXPERIMENT_ID == 1:
    os.rename(PATH + 'overall_result.txt', PATH + 'overall_result_upper.txt')
    os.rename(PATH + 'tool_time.txt', PATH + 'tool_time_upper.txt')
else:
    raise ValueError