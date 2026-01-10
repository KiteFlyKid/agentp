<div align="center">

# Agent+P: Guiding UI Agents via Symbolic Planning

<div align="center">

![Agent+P Overview](Overview.png)

**A novel agentic framework that leverages symbolic planning to guide LLM-based UI automation agents**


## Overview

Agent+P is a plug-and-play module that enhances existing UI agents by providing UI transition information through symbolic planning. The key innovation is using a man-in-the-middle (MITM) proxy to intercept agent prompts and augment them with planning information, eliminating the need for heavy modifications to existing agent codebases.

### How It Works

Agent+P operates as a proxy layer between UI agents and LLM API servers:

1. **UI Transition Graph (UTG) Construction**: Uses ICCBot to statically analyze Android apps and build comprehensive UI transition graphs
2. **Proxy Interception**: Implements MITM proxy to intercept agent prompts containing the `<AGENTP>` tag
3. **Symbolic Planning**: Uses Fast Downward planner to compute optimal action sequences based on UTG
4. **Prompt Augmentation**: Injects planning information into agent prompts to guide decision-making

This architecture allows Agent+P to work with any UI agent framework without requiring source code modifications.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentp.git
cd agentp

# Create and activate conda environment
conda create -n agentp python=3.12 -y
conda activate agentp

# Install dependencies using uv
pip install uv
uv pip install -r requirements.txt
```

## Setup

### Step 1: Build UI Transition Graphs with ICCBot

We provide pre-collected APK files from the AndroidWorld benchmark:

**Download APKs**: [Google Drive Link](https://drive.google.com/drive/folders/1UNvNQisnGgzj7Ped6eMXIdx9ggSXERc4?usp=sharing)

```bash
# Clone ICCBot
git clone https://github.com/SQUARE-RG/ICCBot
cd ICCBot

# Follow ICCBot instructions to analyze APKs and generate UTGs
# Output files should be placed in the agentp/ICCBot_output folder
```

**Note**: Pre-generated UTG outputs are included in the `ICCBot_output` folder for convenience.

### Step 2: Configure MITM Proxy

Add proxy management functions to your shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
proxy() {
    local proxy_address="http://127.0.0.1:8080"
    
    export http_proxy="$proxy_address"
    export https_proxy="$proxy_address"
    export HTTP_PROXY="$proxy_address"
    export HTTPS_PROXY="$proxy_address"
    echo "✅ Proxy enabled (http://127.0.0.1:8080)"
}

unproxy() {
    unset http_proxy
    unset https_proxy
    unset HTTP_PROXY
    unset HTTPS_PROXY
    echo "❌ Proxy disabled"
}
```

Install and configure mitmproxy certificates:

```bash
# Install mitmproxy certificates (required for HTTPS interception)
# Follow the official guide: https://docs.mitmproxy.org/stable/overview/getting-started/

# After installation, set the certificate path
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### Step 3: Install Fast Downward Planner

```bash
# Clone Fast Downward
git clone https://github.com/aibasel/downward.git
```

### Step 4: Setup UI Agent

This repository uses [DroidRun](https://docs.droidrun.ai/) as an example agent. Follow the [DroidRun quickstart guide](https://docs.droidrun.ai/v3/quickstart) to install it.

**Modify Agent Prompt**: Add the `<AGENTP>` tag to the agent's prompt template where it describes the UI or automation goal. For DroidRun, modify:

```
droidrun/agent/context/personas/ui_expert.py (line 28)
```

Add the tag like this:

```python
prompt = f"""
You are a UI automation expert. Your task is to accomplish the following goal:
{goal}

<AGENTP>
Current UI state:
{ui_tree}
"""
```

The `<AGENTP>` tag signals the proxy to inject planning information at this location.

## Usage

### Start the Agent+P Proxy

```bash
# Enable proxy in your terminal
proxy

# Launch the MITM proxy with UTG directory
python mitmproxy.py --utg_dir ICCBot_output
```

The proxy will now intercept API calls and augment prompts with planning guidance.

### Run Your UI Agent

With the proxy running, execute your agent normally:

```bash
# Example: Running DroidRun with Agent+P
python demo_agent.py
```

The agent will automatically benefit from Agent+P's symbolic planning.

## Project Structure

```
agentp/
├── mitmproxy.py           # Main proxy implementation
├── demo_agent.py          # Example DroidRun integration
├── ICCBot_output/         # Pre-generated UTG files
├── requirements.txt       # Python dependencies
├── agentpOverview.png     # Architecture diagram
└── README.md              # This file
```

## Key Features

- **Plug-and-Play**: Only need to add a prompt tag of existing agent codebases
- **Framework Agnostic**: Works with any LLM-based UI agent
- **Symbolic Planning**: Leverages formal planning for reliable action sequences
- **Transparent Integration**: Operates as a transparent proxy layer

## Troubleshooting

**Proxy Connection Issues**: Ensure the proxy is running before starting your agent and that environment variables are set correctly with the `proxy` command.

**Certificate Errors**: If you encounter SSL certificate errors, verify that mitmproxy certificates are properly installed following the [official documentation](https://docs.mitmproxy.org/stable/concepts-certificates/).

**Planning Failures**: Check that Fast Downward is correctly built and that UTG files in `ICCBot_output` are valid and complete.