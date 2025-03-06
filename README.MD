# 0N3 Distillation (Multi-Instance Reasoning)

0N3 Distillation is a multi-instance reasoning application leveraging FastAPI, PyTorch, and Node.js to enable distributed AI text generation with dynamic background model distillation.

## Features

- Multi-instance distributed AI text generation using Python and FastAPI.
- Real-time background training and model distillation.
- Interactive web interface for chatting and tuning model parameters.
- Node.js aggregator to manage multiple Python instances.

## Installation

1. Ensure [Chocolatey](https://chocolatey.org) is installed.
2. Open PowerShell and run the deployment script:
   ```powershell
   .\deploy.ps1 -NumInstances 4
