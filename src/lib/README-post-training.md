# Post Training Flow for LLM Graph Trainer

This module provides a visual graph-based interface for designing and implementing post-training procedures for Large Language Models (LLMs). It supports a sequence of training methods that can be applied to a pre-trained model to enhance its capabilities.

## Supported Training Methods

### 1. Supervised Fine-Tuning (SFT)

SFT is the process of fine-tuning a pre-trained language model on a specific dataset with human-labeled examples. This helps the model learn to follow instructions and generate responses that align with human preferences.

**Key Parameters:**
- Learning Rate: Typically 5e-5 to 1e-6
- Batch Size: 8-32 depending on GPU memory
- Number of Epochs: 2-5
- Weight Decay: 0.01
- Warmup Steps: 500

### 2. Proximal Policy Optimization (PPO)

PPO is a reinforcement learning algorithm that uses a reward model to guide the language model's outputs. It helps the model learn to generate responses that maximize a given reward function.

**Key Parameters:**
- Learning Rate: 1e-5 to 1e-6
- Clip Epsilon: 0.1-0.2
- Value Coefficient: 0.1
- Entropy Coefficient: 0.01
- Discount Factor: 0.99
- GAE Lambda: 0.95

### 3. Direct Preference Optimization (DPO)

DPO is a method that directly optimizes a language model to align with human preferences without using a separate reward model. It uses pairs of preferred and non-preferred responses to guide the training.

**Key Parameters:**
- Learning Rate: 5e-6 to 1e-7
- Beta: 0.1-0.5
- Reference Model: Usually the SFT model

### 4. Generalized Reward-Penalized Optimization (GRPO)

GRPO is an advanced reinforcement learning method that combines elements of PPO and DPO. It uses a reward threshold to determine whether to reward or penalize model outputs.

**Key Parameters:**
- Learning Rate: 2e-6 to 1e-7
- Clip Epsilon: 0.1
- Reward Threshold: 0.7

## How to Use

1. **Create a Flow**: Use the visual editor to create a flow by adding training nodes and connecting them in sequence.

2. **Configure Parameters**: Adjust the parameters for each training node based on your specific requirements.

3. **Generate Code**: Click the "Generate Code" button to create Python code that implements your training pipeline.

4. **Run the Pipeline**: Execute the generated code to train your model using the specified methods.

## Example Pipeline

The default post-training flow demonstrates a complete pipeline:

1. Start with a pre-trained model (e.g., Llama-2-7b)
2. Apply SFT training with a custom dataset
3. Apply PPO training with a reward model
4. Apply DPO training with preference pairs
5. Apply GRPO training for final refinement

## Requirements

To run the generated code, you'll need:

- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- A GPU with sufficient memory (at least 24GB recommended)

## Tips for Effective Training

1. **Dataset Quality**: The quality of your training data is crucial, especially for SFT and DPO.

2. **Hyperparameter Tuning**: Start with the default parameters and adjust based on your specific model and task.

3. **Monitoring**: Monitor training metrics carefully to detect issues like overfitting or divergence.

4. **Evaluation**: Regularly evaluate your model on a separate test set to ensure it's improving.

5. **Progressive Training**: Consider training with smaller models first before scaling up to larger ones.

## References

- [Supervised Fine-tuning for Large Language Models](https://arxiv.org/abs/2109.01652)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [GRPO: Generalized Reward-Penalized Optimization](https://arxiv.org/abs/2305.18290) 