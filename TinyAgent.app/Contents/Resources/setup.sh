#!/bin/zsh
LLMCompilerPath="$1"
source ~/.zshrc

# Clone LLMCompiler into Application Support directory
git clone git@github.com:SqueezeAILab/TinyAgent.git "$LLMCompilerPath" --branch main

conda create --name tinyagent-llmcompiler python=3.10 -y
conda activate tinyagent-llmcompiler

# Install requirements
cd "$LLMCompilerPath"
pip install -r requirements.txt

# Save where conda is installed
which python >> "$LLMCompilerPath/pythonpath.txt"
