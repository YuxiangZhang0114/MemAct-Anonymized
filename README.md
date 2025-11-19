# Memory-as-Action: DCPO Training Framework

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Paper**: [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635)

A reinforcement learning training framework based on **verl 0.5.0**, implementing the **Dynamic Context Policy Optimization (DCPO)** algorithm for training agents that can autonomously manage context.

## 📖 Introduction

This project provides a complete training framework supporting:
- Autonomous context management by agents (via memory editing tools)
- DCPO algorithm for reinforcement learning training
- Multi-turn conversations and tool calling
- Distributed training (with Ray support)

## ✨ Core Features

- **DCPO Training**: Reinforcement learning algorithm supporting trajectory segmentation and advantage estimation
- **Memory Management Tools**: Agents can actively edit context through tool calls
- **Multi-turn Conversations**: Support for long-horizon multi-turn interactions
- **Tool Integration**: Built-in search tools and context pruning tools
- **Distributed Training**: Ray-based distributed training support

## 🏗️ Project Structure

```
rl_train/
├── DCPO/                          # DCPO related configs and scripts
│   ├── config/                    # Training configuration files
│   │   ├── mem_agent_loop_config.yaml
│   │   └── mem_search_tool_config_single.yaml
│   ├── data/                      # Training and validation data
│   ├── scripts/                   # Training scripts
│   │   ├── run_dcpo_7B.sh
│   │   ├── run_dcpo_14B_single.sh
│   │   └── reward_service.sh
│   └── tool_service/              # Tool service implementations
│       ├── search_search_services.py
│       └── search_tool_single.py
├── verl/                          # Core training framework
│   ├── experimental/
│   │   └── agent_loop/            # Agent loop implementations
│   │       ├── mem_agent_loop.py  # MemAct agent loop
│   │       ├── agent_loop.py      # Base agent loop
│   │       └── tool_parser.py     # Tool parser
│   ├── trainer/                   # Trainers
│   │   ├── main_ppo.py            # PPO main training entry
│   │   └── ppo/
│   │       ├── core_algos.py      # Core algorithms (including DCPO)
│   │       └── ray_trainer.py     # Ray distributed training
│   ├── tools/                     # Tool implementations
│   │   ├── base_tool.py
│   │   └── search_tool.py
│   └── utils/                     # Utility functions
└── cold_start/                    # Cold start related
    ├── data/
    └── scripts/
```

## 🚀 Quick Start

### Requirements

- verl 0.5.0
- Multi-GPU environment (recommended: 8x H100 or equivalent)

### Installation

1. **Install verl 0.5.0**
```bash
pip install verl==0.5.0
```

2. **Install additional dependencies**
```bash
pip install shortuuid uuid
pip install numpy==1.26.4
```

### Data

Data files should be placed in `*/data/` directories.

### Training Configuration

1. **Modify training scripts** (e.g., `DCPO/scripts/run_dcpo_7B.sh`):
   - Set model path: `model_path=/path/to/your/model`
   - Set data paths: `train_files` and `test_files`
   - Set log directory: `logdir`
   - Configure GPU count and other training parameters

2. **Configure tools** (if needed):
   Edit `DCPO/config/mem_search_tool_config_single.yaml` to configure tool service addresses and parameters

### Running Training

Before RL training, you need to start the reward service and retrieval service in advance.

## 📝 Configuration

### Training Script Parameters

Main configurations are set in training scripts, overriding defaults via Hydra:

```bash
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mem_agent_loop_config' \
    algorithm.adv_estimator=dcpo \          # Use DCPO algorithm
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \          # Max prompt length
    data.max_response_length=20480 \       # Max response length
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40 \
    actor_rollout_ref.rollout.n=12 \       # Number of segments per prompt used for training
    actor_rollout_ref.rollout.actual_n=8 \ # Number of trajectories generated per prompt
    ...
```

### Configuration Files

- **`DCPO/config/mem_agent_loop_config.yaml`**: Main training configuration
- **`DCPO/config/mem_search_tool_config_single.yaml`**: Tool configuration
  - Search tool: Configure retrieval service URL and parameters
  - Context pruning tool: For managing conversation history

### Key Parameters

- `algorithm.adv_estimator=dcpo`: Must be set to `dcpo` to use DCPO algorithm
- `actor_rollout_ref.rollout.actual_n`: DCPO sampling strategy, sample `n` segments from `actual_n` trajectories for training
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns`: Maximum conversation turns
- `actor_rollout_ref.rollout.multi_turn.tool_config_path`: Tool configuration file path

## 📚 Related Resources

- **verl**: https://github.com/volcengine/verl (version 0.5.0)
- Training is based on verl's PPO framework, using DCPO as the advantage estimator

## 📄 Citation

If you use this project, please cite our paper:

```bibtex
@article{zhang2025memory,
  title={Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks},
  author={Zhang, Yuxiang and Shu, Jiangming and Ma, Ye and Lin, Xueyuan and Wu, Shangxi and Sang, Jitao},
  journal={arXiv preprint arXiv:2510.12635},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please contact us

