---
license: mit
tags:
- function calling
- on-device language model
language:
- en
---

# TinyAgent: Function Calling at the Edge

<p align="center">
<a href="https://github.com/SqueezeAILab/TinyAgent/raw/main/TinyAgent.zip">Get the desktop app</a>‎ ‎ 
  |
<a href="https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/">Read the blog post</a>
</p>

![Thumbnail](https://cdn-uploads.huggingface.co/production/uploads/648903e1ce7b9a2abe3511aa/a1YuQosFiJQJ_7Ejribrd.png)

TinyAgent aims to enable complex reasoning and function calling capabilities in Small Language Models (SLMs) that can be deployed securely and privately at the edge. Traditional Large Language Models (LLMs) like GPT-4 and Gemini-1.5, while powerful, are often too large and resource-intensive for edge deployment, posing challenges in terms of privacy, connectivity, and latency. TinyAgent addresses these challenges by training specialized SLMs with high-quality, curated data, and focusing on function calling with [LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler). As a driving application, TinyAgent can interact with various MacOS applications, assisting users with day-to-day tasks such as composing emails, managing contacts, scheduling calendar events, and organizing Zoom meetings.


**Model Developers:** Squeeze AI Lab at University of California, Berkeley.

**Variations:** TinyAgent models come in 2 sizes: TinyAgent-1.1B and TinyAgent-7B

**License:** MIT


## Demo

<a href="https://youtu.be/0GvaGL9IDpQ" target="_blank" rel="noopener noreferrer">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/648903e1ce7b9a2abe3511aa/BpN-zPzfqa8wcRuJiYOYC.png" alt="TinyAgent Demo" width="700">
</a>

## How to Use
Please see our [Github](https://github.com/SqueezeAILab/TinyAgent) for details on how to use TinyAgent models. TinyAgent models can be used programmatically or through our user interface.

## Training Details

**Dataset:** 
We curated a [dataset](https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset) of **40,000** real-life use cases. We use GPT-3.5-Turbo to generate real-world instructions. These are then used to obtain synthetic execution plans using GPT-4-Turbo. Please check out our [blog post](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/) for more details on our dataset.

**Fine-tuning Procedure:**
TinyAgent models are fine-tuned from base models. Below is a table of each TinyAgent model with its base counterpart
| Model                                                                                                                                                       | Success Rate |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| GPT-3.5-turbo                                                                                                                                               | 65.04%       |
| GPT-4-turbo                                                                                                                                                 | 79.08%       |
| [TinyLLama-1.1B-32K-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct)                                                            | 12.71%       |
| [WizardLM-2-7b](https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF)                                                                                    | 41.25%       |
| TinyAgent-1.1B + ToolRAG / [[hf](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B)] [[gguf](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B-GGUF)] | **80.06%**   |
| TinyAgent-7B + ToolRAG / [[hf](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B)] [[gguf](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B-GGUF)]       | **84.95%**   |

Using the synthetic data generation process described above, we use parameter-efficient fine-tuning with LoRA to fine-tune the base models for 3 epochs. Please check out our [blog post](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/) for more details on our fine-tuning procedure.

### 🛠️ ToolRAG

When faced with challenging tasks, SLM agents require appropriate tools and in-context examples to guide them. If the model sees irrelevant examples, it can hallucinate. Likewise, if the model sees the descriptions of the tools that it doesn’t need, it usually gets confused, and these tools take up unnecessary prompt space. To tackle this, TinyAgent uses ToolRAG to retrieve the best tools and examples suited for a given query. This process has minimal latency and increases the accuracy of TinyAgent substantially. Please take a look at our [blog post](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/) and our [ToolRAG model](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG) for more details.

## Links
**Blog Post**: https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/

**Github:** https://github.com/SqueezeAILab/TinyAgent