import numpy as np
import os
class CustomLogger:
    def __init__ (self, file_dir: str, experiment_id: int = 0, planner_answer: str = None, generation_token_list: list[int] = [None, None]):
        self.e2e_time = []
        self.init_time = []
        self.rag_time = []
        self.planner_time = []
        self.agent_time = []
        self.tool_time = []
        self.planner_tokens = [] # Number of planner input tokens, output tokens
        self.agent_tokens = [] # Number of agent input tokens, output tokens
        self.file_dir = file_dir
        self.delimiter = ","
        # 0: Planner System Prompt o, Agent System Prompt o
        # 1: Planner System Prompt x, Agent System Prompt x
        # 2: Planner System Prompt x, Agent System Prompt o
        # 3: Planner System Prompt o, Agent System Prompt x
        self.experiment_id = experiment_id
        self.planner_answer = planner_answer
        self.generation_token_list = generation_token_list
    
    def update_e2e_time(self, start_time, end_time):
        self.e2e_time = [str(start_time), str(end_time)]

    def update_init_time(self, start_time, end_time):
        self.init_time = [str(start_time), str(end_time)]

    def update_rag_time(self, start_time, end_time):
        self.rag_time = [str(start_time), str(end_time)]

    def update_planner_time(self, start_time, end_time):
        self.planner_time = [str(start_time), str(end_time)]

    def update_agent_time(self, start_time, end_time):
        self.agent_time = [str(start_time), str(end_time)]
    
    def update_tool_time(self, tool_id, tool_name, start_time, end_time):
        self.tool_time.append([str(tool_id), str(tool_name), str(start_time), str(end_time)])

    def update_planner_token(self, n_input_token, n_output_token):
        self.planner_tokens = [str(n_input_token), str(n_output_token)]

    def update_agent_token(self, n_input_token, n_output_token):
        self.agent_tokens = [str(n_input_token), str(n_output_token)]
    
    def save_logging_result(self):
        error = []
        if self.rag_time == []:
            self.rag_time = ['0', '0']
            error.append('RAG_Error')

        if self.planner_time == []:
            self.planner_time = ['0', '0']
            self.planner_tokens = ['0', '0']
            error.append('Planner_Error')
        
        if self.tool_time == []:
            self.tool_time.append(['1', 'ERROR', '0', '0'])
            error.append('Tool_Error')

        if self.agent_time == []:
            self.agent_time = ['0', '0']
            self.agent_tokens = ['0', '0']
            error.append('Agent_Error')

        if error == []:
            status = 'Success'
        else:
            status = "+".join(error)

        data = np.array(self.tool_time)
        data = data[:, 2:4].astype('float')

        tool_e2e_time = [str(np.min(data, axis=0)[0]), str(np.max(data, axis=0)[1])]
        result = [status] + self.e2e_time + self.init_time + self.rag_time + self.planner_time + tool_e2e_time + self.agent_time + self.planner_tokens + self.agent_tokens
        # text = f"E2E{delimiter.join(self.e2e_time)}|RAG{delimiter.join(self.rag_time)}|PLANNER{delimiter.join(self.planner_time)}|TOOL{delimiter.join(tool_e2e_time)}|AGENT{delimiter.join(self.agent_time)}\n"
        result_text = f"{self.delimiter.join(result)}\n"

        name = self.file_dir + 'overall_result.txt'
        if os.path.isfile(name):
            f = open(name, 'a')
            f.write(result_text)
            f.close()
        else:
            header = ['Status', 'E2E Start', 'E2E End', 'Class Initialization Start', 'Class Initialization End', 'RAG Start', 'RAG End', 'Planner Start', 'Planner End', 'Tool Start', 'Tool End', 'Agent Start', 'Agent End', 'Planner Input Token', 'Planner Output Token', 'Agent Input Token', 'Agent Output Token']
            header_text = f"{self.delimiter.join(header)}\n"
            f = open(name, 'a')
            f.write(header_text)
            f.write(result_text)
            f.close()
    
    def save_tool_time(self):
        result = [j for data in self.tool_time for j in data]
        result_text = self.delimiter.join(result) + "\n"
        f = open(self.file_dir + 'tool_time.txt', 'a')
        f.write(result_text)
        f.close()