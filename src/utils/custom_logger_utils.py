import numpy as np
import os
import time
class CustomLogger:
    def __init__ (self,
                  file_dir: str,
                  experiment_id: int = 0,
                  planner_answer: str = None,
                  generation_token_list: list[int] = [None, None],
                  save_time_profile: bool = False,
                  save_planner_inout_profile: bool = False,
                  save_agent_inout_profile: bool = False,
                  save_tool_inout_profile: bool = False,
                  ):
        self.save_time_profile = save_time_profile
        self.save_planner_inout_profile = save_planner_inout_profile
        self.save_agent_inout_profile = save_agent_inout_profile
        self.save_tool_inout_profile = save_tool_inout_profile
        
        self.global_time = 0.0
        self.component_time_profile = {}
        self.tool_time_profile = []

        self.e2e_time = []
        self.init_time = []
        self.rag_time = []
        self.planner_time = []
        self.agent_time = []
        
        self.planner_tokens = [] # Number of planner input tokens, output tokens
        self.agent_tokens = [] # Number of agent input tokens, output tokens
        self.file_dir = file_dir
        self.delimiter = "●"
        # 0: Planner System Prompt o, Agent System Prompt o
        # 1: Planner System Prompt x, Agent System Prompt x
        # 2: Planner System Prompt x, Agent System Prompt o
        # 3: Planner System Prompt o, Agent System Prompt x
        self.experiment_id = experiment_id
        self.planner_answer = planner_answer
        self.generation_token_list = generation_token_list

        self.planner_profile = []
    
    def update_global_time(self, time):
        self.global_time = time

    def log_component_time(self, key):
        valid_keys = ['e2e_start', 'e2e_end', 'init_start', 'init_end', 'rag_start', 'rag_end', 'planner_start', 'planner_end', 'agent_start', 'agent_end']
        if key not in valid_keys:
            raise ValueError(f"Key must be in {valid_keys}")
        relative_time = time.time() - self.global_time
        self.component_time_profile[key] = relative_time
        print(f"[SYSTEM] {key.upper()}_TIME: {relative_time:.4f}")
        # self.time_profile.setdefault(key, []).append(time.time() - self.global_time)
    
    def log_tool_time(self, tool_id, tool_name, start_time, end_time):
        self.tool_time_profile.append([str(tool_id), str(tool_name), str(start_time), str(end_time)])
    
    
    def update_planner_token(self, n_input_token, n_output_token):
        self.planner_tokens = [str(n_input_token), str(n_output_token)]

    def update_agent_token(self, n_input_token, n_output_token):
        self.agent_tokens = [str(n_input_token), str(n_output_token)]

    def update_planner_profile(self, system_prompt, human_prompt, output_prompt):
        self.planner_profile = [system_prompt.encode('unicode_escape').decode('utf-8'), human_prompt.encode('unicode_escape').decode('utf-8'), output_prompt.encode('unicode_escape').decode('utf-8')]
    
    def save_profile(self):
        if self.save_time_profile:
            # Save systematic executed time
            error = []
            if 'rag_end' not in list(self.component_time_profile.keys()):
                self.component_time_profile['rag_start'] = 0
                self.component_time_profile['rag_end'] = 0
                error.append('RAG_Error')

            if 'planner_end' not in list(self.component_time_profile.keys()):
                self.component_time_profile['planner_start'] = 0
                self.component_time_profile['planner_end'] = 0
                error.append('Planner_Error')
            
            if 'agent_end' not in list(self.component_time_profile.keys()):
                self.component_time_profile['agent_start'] = 0
                self.component_time_profile['agent_end'] = 0
                error.append('Agent_Error')
            
            if self.tool_time_profile == []:
                self.tool_time_profile.append(['0', 'ERROR', '0', '0'])
                error.append('Tool_Error')

            if error == []:
                status = 'Success'
            else:
                status = "+".join(error)

            data = np.array(self.tool_time_profile)
            data = data[:, 2:4].astype('float') #data => [Task ID, Task Name, Start Time, End Time]

            self.component_time_profile['tool_start'] = np.min(data, axis=0)[0]
            self.component_time_profile['tool_end'] = np.max(data, axis=0)[1]

            valid_keys = ['e2e_start', 'e2e_end', 'init_start', 'init_end', 'rag_start', 'rag_end', 'planner_start', 'planner_end', 'tool_start', 'tool_end', 'agent_start', 'agent_end']
            result = [status]
            for key in valid_keys:
                result.append(str(self.component_time_profile[key]))
            result_text = f"{self.delimiter.join(result)}\n"

            name = self.file_dir + 'component_time_profile.txt'
            if os.path.isfile(name):
                f = open(name, 'a')
                f.write(result_text)
                f.close()
            else:
                header = ['Status', 'E2E Start', 'E2E End', 'Class Initialization Start', 'Class Initialization End', 'RAG Start', 'RAG End', 'Planner Start', 'Planner End', 'Tool Start', 'Tool End', 'Agent Start', 'Agent End']
                header_text = f"{self.delimiter.join(header)}\n"
                f = open(name, 'a')
                f.write(header_text)
                f.write(result_text)
                f.close()

            # Save time of executed tools
            result = [j for data in self.tool_time_profile for j in data]
            result_text = self.delimiter.join(result) + "\n"
            f = open(self.file_dir + 'tool_time_profile.txt', 'a')
            f.write(result_text)
            f.close()
        
        # Save planner input (system prompt, request) and planner output (response)
        if self.save_planner_inout_profile:
            if self.planner_time == []:
                self.planner_time = ['0', '0']
                self.planner_tokens = ['0', '0']

            result_text = f"{self.delimiter.join(self.planner_profile)}\n"
            name = self.file_dir + 'planner_inout_profile.txt'
            if os.path.isfile(name):
                f = open(name, 'a')
                f.write(result_text)
                f.close()
            else:
                header = ['System Prompt', 'Request', 'Planner Response']
                header_text = f"{self.delimiter.join(header)}\n"
                f = open(name, 'a')
                f.write(header_text)
                f.write(result_text)
                f.close()

        if self.save_agent_inout_profile:
            # 'Planner Input Token', 'Planner Output Token', 'Agent Input Token', 'Agent Output Token'
            if self.planner_time == []:
                self.planner_time = ['0', '0']
                self.planner_tokens = ['0', '0']
            raise ValueError('Not Implemented Yet')
        
        if self.save_tool_inout_profile:
            raise ValueError('Not Implemented Yet')