import os
import numpy as np
import pandas as pd

def print_task_graph(arr, query_id):
    data = np.array(arr)
    id = data[:, 0]
    print(data[id==str(query_id)])

data_dir = '/home/munyeolpark/spd/TinyAgent/results/'

f = open(os.path.join(data_dir, 'tool_time_mac.txt'), 'r')
data = f.readlines()
f.close()

f = open(os.path.join(data_dir, 'overall_time_mac.txt'), 'r')
data2 = f.readlines()
f.close()

results = []
result_llm = []
result_nollm = []
check_list = ['append_note_content', 'create_note', 'compose_new_email', 'reply_to_email', 'summarize_pdf']

for i, elements in enumerate(data):
    
    elements = elements.strip().split(' ')
    elements = np.array(elements).reshape(-1, 4)

    status = data2[i].strip().split(' ')
    status = np.array([status[0]] * elements.shape[0]).reshape(-1, 1)

    latency = elements[:, 3].astype('float') - elements[:, 2].astype('float')
    latency = latency.reshape(-1, 1)

    id = (i+1)*np.ones((elements.shape[0], 1), dtype='int')
    data = np.concatenate((status, id, elements, latency), axis=1).tolist()


    is_llm = False
    tool_name = elements[:, 1]
    for check in check_list:
        if check in tool_name:
            is_llm = True
            break
    
    if is_llm:
        result_llm = result_llm + data
    else:
        result_nollm = result_nollm + data
    results = results + data
    

df_total = pd.DataFrame(results, columns=['Status', 'Query ID', 'Task ID', 'Task Name', 'Start Time', 'End Time', 'Latency'])
df_total.to_excel(os.path.join(data_dir, 'tool_time_mac_total.xlsx'), index=False)
df_llm = pd.DataFrame(result_llm, columns=['Status', 'Query ID', 'Task ID', 'Task Name', 'Start Time', 'End Time', 'Latency'])
df_llm.to_excel(os.path.join(data_dir, 'tool_time_mac_llm.xlsx'), index=False)
df_nollm = pd.DataFrame(result_nollm, columns=['Status', 'Query ID', 'Task ID', 'Task Name', 'Start Time', 'End Time', 'Latency'])
df_nollm.to_excel(os.path.join(data_dir, 'tool_time_mac_nollm.xlsx'), index=False)


n_query_list = []
n_query_list.append(np.max(df_total['Query ID'].values.astype('int')))
n_query_list.append(np.max(df_llm['Query ID'].values.astype('int')))
n_query_list.append(np.max(df_nollm['Query ID'].values.astype('int')))
print(n_query_list)

latency_list = []
latency_list.append(np.average(df_total[df_total['Task Name'].isin(check_list)]['Latency'].astype('float')))
latency_list.append(np.average(df_total[~df_total['Task Name'].isin(check_list)]['Latency'].astype('float')))

latency_list.append(np.average(df_llm[df_llm['Task Name'].isin(check_list)]['Latency'].astype('float')))
latency_list.append(np.average(df_llm[~df_llm['Task Name'].isin(check_list)]['Latency'].astype('float')))

latency_list.append(np.average(df_nollm[df_nollm['Task Name'].isin(check_list)]['Latency'].astype('float')))
latency_list.append(np.average(df_nollm[~df_nollm['Task Name'].isin(check_list)]['Latency'].astype('float')))

print(latency_list)