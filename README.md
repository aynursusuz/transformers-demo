# Transformers Demo: Question-Answering Model

This repository demonstrates how to use a language model (Qwen2.5-0.5B-Instruct) to generate responses to user input with Hugging Face Transformers.

## Overview

This project utilizes the Hugging Face `transformers` library to load a pre-trained causal language model and generate text-based responses. The model used in this example is `Qwen/Qwen2.5-0.5B-Instruct`, and it is used to respond to input questions or prompts.

## Prerequisites

Before you can run the code, ensure you have the following dependencies installed:

- Python 3.x
- `transformers` library
- `torch` (PyTorch)

You can install the required libraries using pip:

```bash
pip install transformers torch
