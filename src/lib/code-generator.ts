import { Node, Edge } from 'reactflow';
import { 
  LLMNodeData, 
  EmbeddingNodeData, 
  PositionalEncodingNodeData,
  QKVAttentionNodeData,
  FFNNodeData,
  OutputNodeData,
  LayerNormNodeData,
  SFTTrainingNodeData,
  PPOTrainingNodeData,
  DPOTrainingNodeData,
  GRPOTrainingNodeData
} from './types';
import { OptimizationSettings, defaultOptimizationSettings } from '@/components/optimization-panel/OptimizationPanel';

/**
 * Generates Python code for PyTorch based on the nodes and edges in the flow
 */
export function generatePythonCode(
  nodes: Node<LLMNodeData>[],
  edges: Edge[],
  optimizationSettings?: OptimizationSettings
): string {
  // Separate model nodes from training nodes
  const modelNodes = nodes.filter(node => 
    !['sftTraining', 'ppoTraining', 'dpoTraining', 'grpoTraining'].includes(node.data.type)
  );
  
  const trainingNodes = nodes.filter(node => 
    ['sftTraining', 'ppoTraining', 'dpoTraining', 'grpoTraining'].includes(node.data.type)
  );
  
  // Sort nodes by dependencies
  const sortedModelNodes = sortNodesByDependencies(modelNodes, edges);
  
  // Sort training nodes in the correct sequence (SFT -> PPO -> DPO -> GRPO)
  const sortedTrainingNodes = sortTrainingNodesBySequence(trainingNodes);
  
  // Generate code
  const imports = generateImports(optimizationSettings);
  const hyperparameters = generateHyperparameters(optimizationSettings);
  const deviceDetection = generateDeviceDetectionCode(optimizationSettings || defaultOptimizationSettings);
  const componentDefinitions = sortedModelNodes.map(node => 
    generateComponentDefinition(node, optimizationSettings)
  ).join('\n\n');
  const modelClass = `class LLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        ${sortedModelNodes.map(node => {
          const componentName = `self.${node.id}`;
          return `${componentName} = ${node.id.charAt(0).toUpperCase() + node.id.slice(1)}()`;
        }).join('\n        ')}
        
    def forward(self, x):
${generateForwardPass(sortedModelNodes, edges, optimizationSettings)}
        return ${sortedModelNodes.length > 0 ? `x_${sortedModelNodes.length}` : 'x'}`;

  const trainingCode = optimizationSettings ? generateTrainingCode(optimizationSettings) : '';
  const experimentCode = optimizationSettings?.experiment?.enabled ? 
    generateExperimentCode(optimizationSettings) : '';
    
  // Generate post-training pipeline if training nodes exist
  const postTrainingPipeline = sortedTrainingNodes.length > 0 ? 
    generatePostTrainingPipeline(sortedTrainingNodes) : '';

  return `${imports}

${hyperparameters}

${deviceDetection}

${componentDefinitions}

${modelClass}

${trainingCode}

${experimentCode}

${postTrainingPipeline}

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Graph Builder Model")
    print("=" * 50)
    
    model = LLMModel()
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if ${optimizationSettings?.experiment?.enabled || 'False'}:
        run_experiment(model)
        print("\\nExperiment completed successfully!")
    
    ${sortedTrainingNodes.length > 0 ? 'if input("\\nRun post-training pipeline? (y/n): ").lower() == "y":\\n    run_post_training_pipeline(model)' : ''}
`;
}

/**
 * Generates the necessary imports based on optimization settings
 */
function generateImports(optimizationSettings?: OptimizationSettings): string {
  let imports = `import torch
import torch.nn as nn
import math
import os
import platform
import time
import subprocess
from typing import Optional, Tuple, List, Dict
`;

  if (optimizationSettings) {
    if (optimizationSettings.fsdp.enabled) {
      imports += `import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
`;
    }
    
    if (optimizationSettings.deepSpeed.enabled) {
      imports += `import deepspeed
`;
    }
    
    if (optimizationSettings.flashAttention) {
      imports += `from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
`;
    }
    
    if (optimizationSettings.xformers) {
      imports += `import xformers
import xformers.ops
`;
    }
    
    if (optimizationSettings.mixedPrecision !== 'none') {
      imports += `from torch.cuda.amp import autocast, GradScaler
`;
    }
    
    // Add MoE imports if enabled
    if (optimizationSettings.moe?.enabled) {
      imports += `# Mixture of Experts imports
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
`;
    }
    
    // Add imports for device detection and memory tracking
    if (optimizationSettings.deviceDetection.enabled) {
      imports += `try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Some memory tracking features will be disabled.")
    print("Install with: pip install psutil")
`;
    }
    
    // Add imports for experiment tracking if enabled
    if (optimizationSettings.experiment?.enabled) {
      imports += `import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
`;
    }
  }
  
  return imports;
}

/**
 * Sorts nodes based on their dependencies (connections)
 */
function sortNodesByDependencies(
  nodes: Node<LLMNodeData>[],
  edges: Edge[]
): Node<LLMNodeData>[] {
  // Create a map of node IDs to their dependencies
  const dependencies: Record<string, string[]> = {};
  const nodeMap: Record<string, Node<LLMNodeData>> = {};
  
  // Initialize dependencies and node map
  nodes.forEach(node => {
    dependencies[node.id] = [];
    nodeMap[node.id] = node;
  });
  
  // Add dependencies based on edges
  edges.forEach(edge => {
    if (dependencies[edge.target]) {
      dependencies[edge.target].push(edge.source);
    }
  });
  
  // Topological sort
  const visited = new Set<string>();
  const temp = new Set<string>();
  const result: Node<LLMNodeData>[] = [];
  
  function visit(nodeId: string) {
    if (temp.has(nodeId)) {
      // Circular dependency detected
      return;
    }
    
    if (!visited.has(nodeId)) {
      temp.add(nodeId);
      
      // Visit all dependencies
      dependencies[nodeId].forEach(depId => {
        visit(depId);
      });
      
      temp.delete(nodeId);
      visited.add(nodeId);
      result.push(nodeMap[nodeId]);
    }
  }
  
  // Visit all nodes
  nodes.forEach(node => {
    if (!visited.has(node.id)) {
      visit(node.id);
    }
  });
  
  return result;
}

/**
 * Generates component definition code for a node
 */
function generateComponentDefinition(
  node: Node<LLMNodeData>,
  optimizationSettings?: OptimizationSettings
): string {
  const { type, params } = node.data;
  const componentName = `self.${node.id}`;
  
  let code = '';
  
  switch (type) {
    case 'embedding':
      code = generateEmbeddingCode(componentName, params as EmbeddingNodeData['params']);
      break;
    case 'positionalEncoding':
      code = generatePositionalEncodingCode(componentName, params as PositionalEncodingNodeData['params']);
      break;
    case 'qkvAttention':
      code = generateQKVAttentionCode(componentName, params as QKVAttentionNodeData['params'], optimizationSettings);
      break;
    case 'ffn':
      code = generateFFNCode(componentName, params as FFNNodeData['params'], optimizationSettings);
      break;
    case 'layerNorm':
      code = generateLayerNormCode(componentName, params as LayerNormNodeData['params']);
      break;
    case 'output':
      code = generateOutputCode(componentName, params as OutputNodeData['params']);
      break;
    case 'sftTraining':
      return generateSFTTrainingCode(node as Node<SFTTrainingNodeData>);
    case 'ppoTraining':
      return generatePPOTrainingCode(node as Node<PPOTrainingNodeData>);
    case 'dpoTraining':
      return generateDPOTrainingCode(node as Node<DPOTrainingNodeData>);
    case 'grpoTraining':
      return generateGRPOTrainingCode(node as Node<GRPOTrainingNodeData>);
    default:
      return `# Unknown node type: ${type}`;
  }
  
  // Capitalize the first letter of the node ID for the class name
  const className = node.id.split('-').map(part => 
    part.charAt(0).toUpperCase() + part.slice(1)
  ).join('-');
  
  return `class ${className}(nn.Module):
    def __init__(self):
        super().__init__()
${code}
        
    def forward(self, x):
        return ${componentName}(x)`;
}

/**
 * Generates the forward pass code based on the sorted nodes and edges
 */
function generateForwardPass(
  sortedNodes: Node<LLMNodeData>[],
  edges: Edge[],
  optimizationSettings?: OptimizationSettings
): string {
  // Create a map of node IDs to their output variable names
  const outputVars: Record<string, string> = {};
  
  // Create a map of node IDs to their input connections
  const inputConnections: Record<string, string[]> = {};
  
  // Initialize input connections
  sortedNodes.forEach(node => {
    inputConnections[node.id] = [];
  });
  
  // Map edges to input connections
  edges.forEach(edge => {
    if (inputConnections[edge.target]) {
      inputConnections[edge.target].push(edge.source);
    }
  });
  
  // Generate forward pass code
  let forwardCode = '';
  
  sortedNodes.forEach((node, index) => {
    const nodeId = node.id;
    const componentName = `self.${nodeId}`;
    
    // Determine input variable
    let inputVar = 'x';
    
    // If this node has input connections, use those instead
    if (inputConnections[nodeId] && inputConnections[nodeId].length > 0) {
      // If there's only one input, use it directly
      if (inputConnections[nodeId].length === 1) {
        const sourceId = inputConnections[nodeId][0];
        inputVar = outputVars[sourceId] || 'x';
      } else {
        // If there are multiple inputs, we need to handle them based on node type
        const inputs = inputConnections[nodeId].map(sourceId => outputVars[sourceId] || 'x');
        
        // Different handling based on node type
        switch (node.data.type) {
          case 'qkvAttention':
            // For attention, we might need to combine inputs
            inputVar = `torch.cat([${inputs.join(', ')}, dim=-1])`;
            break;
          default:
            // Default behavior: use the first input
            inputVar = inputs[0];
        }
      }
    }
    
    // Generate output variable name - use sequential numbers instead of node IDs
    const outputVar = `x_${index + 1}`;
    outputVars[nodeId] = outputVar;
    
    // Generate the forward pass line
    let forwardLine = '';
    
    switch (node.data.type) {
      case 'embedding':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      case 'positionalEncoding':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      case 'qkvAttention':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      case 'ffn':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      case 'layerNorm':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      case 'output':
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
        break;
      default:
        forwardLine = `${outputVar} = ${componentName}(${inputVar})`;
    }
    
    forwardCode += `        ${forwardLine}\n`;
    
    // Add debug print if needed
    if (optimizationSettings?.debug) {
      forwardCode += `        print(f"Output of ${nodeId}: {${outputVar}.shape}")\n`;
    }
  });
  
  return forwardCode;
}

/**
 * Generates code for an Embedding component
 */
function generateEmbeddingCode(
  componentName: string,
  params: EmbeddingNodeData['params']
): string {
  const { vocabSize, embeddingDim, paddingIdx, maxNorm, normType, scaleGradByFreq, sparse } = params;
  
  let code = `${componentName} = nn.Embedding(
            num_embeddings=${vocabSize},
            embedding_dim=${embeddingDim}`;
  
  if (paddingIdx !== undefined) code += `,\n            padding_idx=${paddingIdx}`;
  if (maxNorm !== undefined) code += `,\n            max_norm=${maxNorm}`;
  if (normType !== undefined) code += `,\n            norm_type=${normType}`;
  if (scaleGradByFreq !== undefined) code += `,\n            scale_grad_by_freq=${scaleGradByFreq}`;
  if (sparse !== undefined) code += `,\n            sparse=${sparse}`;
  
  code += ')';
  return code;
}

/**
 * Generates code for a Positional Encoding component
 */
function generatePositionalEncodingCode(
  componentName: string,
  params: PositionalEncodingNodeData['params']
): string {
  const { encodingType, embeddingDim, maxSeqLength, dropout } = params;
  
  switch (encodingType) {
    case 'sinusoidal':
      return `${componentName} = SinusoidalPositionalEncoding(
            d_model=${embeddingDim},
            max_len=${maxSeqLength},
            dropout=${dropout || 0.1})`;
    case 'learned':
      return `${componentName} = nn.Embedding(
            num_embeddings=${maxSeqLength},
            embedding_dim=${embeddingDim})`;
    case 'rotary':
      return `${componentName} = RotaryPositionalEncoding(
            dim=${embeddingDim},
            max_seq_len=${maxSeqLength})`;
    case 'alibi':
      return `${componentName} = ALiBiPositionalEncoding(
            num_heads=${Math.ceil(embeddingDim / 64)},
            max_seq_len=${maxSeqLength})`;
    default:
      return `# Unknown positional encoding type: ${encodingType}`;
  }
}

/**
 * Generates code for a QKV Attention component
 */
function generateQKVAttentionCode(
  componentName: string,
  params: QKVAttentionNodeData['params'],
  optimizationSettings?: OptimizationSettings
): string {
  const { embeddingDim, numHeads, dropout, batchFirst, attentionType, causal } = params;
  
  let code = `${componentName} = nn.MultiheadAttention(
            embed_dim=${embeddingDim},
            num_heads=${numHeads}`;
  
  if (dropout !== undefined) code += `,\n            dropout=${dropout}`;
  if (batchFirst !== undefined) code += `,\n            batch_first=${batchFirst}`;
  
  code += ')';
  
  // Add a comment for attention type and causal mask
  if (attentionType || causal) {
    code += ` # ${attentionType || 'scaled_dot_product'} attention`;
    if (causal) code += ', causal mask will be applied in forward pass';
  }
  
  // Add optimization comments
  if (optimizationSettings?.flashAttention) {
    code += `\n        # Flash Attention will be used in forward pass for better performance`;
  } else if (optimizationSettings?.xformers) {
    code += `\n        # xFormers memory-efficient attention will be used in forward pass`;
  }
  
  return code;
}

/**
 * Generates code for a Feed Forward Network component
 */
function generateFFNCode(
  componentName: string,
  params: FFNNodeData['params'],
  optimizationSettings?: OptimizationSettings
): string {
  const { inputDim, hiddenDim, outputDim, dropout, activation, layerNorm } = params;
  
  // Check if MoE is enabled
  if (optimizationSettings?.moe?.enabled) {
    return generateMoEFFNCode(componentName, params, optimizationSettings);
  }
  
  let activationClass = 'nn.ReLU()';
  if (activation === 'gelu') activationClass = 'nn.GELU()';
  if (activation === 'silu') activationClass = 'nn.SiLU()';
  if (activation === 'tanh') activationClass = 'nn.Tanh()';
  
  let code = `${componentName} = nn.Sequential(
            nn.Linear(${inputDim}, ${hiddenDim}),
            ${activationClass},`;
  
  if (dropout !== undefined && dropout > 0) {
    code += `\n            nn.Dropout(${dropout}),`;
  }
  
  code += `\n            nn.Linear(${hiddenDim}, ${outputDim})`;
  
  if (layerNorm) {
    code += `,\n            nn.LayerNorm(${outputDim})`;
  }
  
  code += ')';
  
  // Add gradient checkpointing comment if enabled
  if (optimizationSettings?.gradientCheckpointing) {
    code += `\n        # This module will use gradient checkpointing to save memory`;
  }
  
  return code;
}

/**
 * Generates code for a Mixture of Experts FFN component
 */
function generateMoEFFNCode(
  componentName: string,
  params: FFNNodeData['params'],
  optimizationSettings: OptimizationSettings
): string {
  const { inputDim, hiddenDim, outputDim, dropout, activation, layerNorm } = params;
  const { numExperts, topK, capacityFactorTrain, expertDropout } = optimizationSettings.moe;
  
  let activationName = 'relu';
  if (activation === 'gelu') activationName = 'gelu';
  if (activation === 'silu') activationName = 'silu';
  if (activation === 'tanh') activationName = 'tanh';
  
  return `# Mixture of Experts FFN
        ${componentName} = MixtureOfExperts(
            input_dim=${inputDim},
            hidden_dim=${hiddenDim},
            output_dim=${outputDim},
            num_experts=${numExperts},
            top_k=${topK},
            activation='${activationName}',
            capacity_factor_train=${capacityFactorTrain},
            capacity_factor_eval=${optimizationSettings.moe.capacityFactorEval},
            dropout=${dropout || 0},
            expert_dropout=${expertDropout},
            use_expert_parallelism=${optimizationSettings.moe.expertParallelism},
            use_layer_norm=${layerNorm || false}
        )`;
}

/**
 * Generates code for an Output component
 */
function generateOutputCode(
  componentName: string,
  params: OutputNodeData['params']
): string {
  const { inputDim, outputDim, activation } = params;
  
  let code = `${componentName} = nn.Linear(${inputDim}, ${outputDim})`;
  
  if (activation && activation !== 'none') {
    let activationClass = '';
    if (activation === 'softmax') activationClass = 'nn.Softmax(dim=-1)';
    if (activation === 'sigmoid') activationClass = 'nn.Sigmoid()';
    if (activation === 'tanh') activationClass = 'nn.Tanh()';
    
    if (activationClass) {
      code = `${componentName} = nn.Sequential(
            nn.Linear(${inputDim}, ${outputDim}),
            ${activationClass})`;
    }
  }
  
  return code;
}

/**
 * Generates training code with optimizations
 */
function generateTrainingCode(optimizationSettings: OptimizationSettings): string {
  let code = `\n\n# Training setup with optimizations
def create_training_setup(model, learning_rate=1e-4):
    """
    Sets up the model for training with the specified optimizations.
    """`;
  
  // FSDP setup
  if (optimizationSettings.fsdp.enabled) {
    code += `
    # Check if CUDA is available for FSDP
    if not torch.cuda.is_available():
        print("Warning: FSDP requires CUDA. Falling back to standard training.")
        device = get_device(prefer_mps=${optimizationSettings.deviceDetection.preferMps})
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        return model, optimizer
    
    # Initialize distributed process group
    dist.init_process_group("nccl")
    
    # Wrap model with FSDP
    sharding_strategy = ShardingStrategy.${optimizationSettings.fsdp.shardingStrategy}
    
    if ${optimizationSettings.fsdp.autoWrap}:
        # Auto wrap policy based on module size
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=${optimizationSettings.fsdp.minNumParams}
        )
        
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
        )
    else:
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
        )
        
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    `;
  }
  
  // DeepSpeed setup
  else if (optimizationSettings.deepSpeed.enabled) {
    code += `
    # Check if CUDA is available for DeepSpeed
    if not torch.cuda.is_available():
        print("Warning: DeepSpeed requires CUDA. Falling back to standard training.")
        device = get_device(prefer_mps=${optimizationSettings.deviceDetection.preferMps})
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        return model, optimizer
    
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": 32,
        "fp16": {
            "enabled": ${optimizationSettings.mixedPrecision === 'fp16'},
        },
        "bf16": {
            "enabled": ${optimizationSettings.mixedPrecision === 'bf16'},
        },
        "zero_optimization": {
            "stage": ${optimizationSettings.deepSpeed.stageThree ? 3 : 2},
            "offload_optimizer": {
                "device": "cpu" if ${optimizationSettings.deepSpeed.offloadOptimizer} else "none",
            },
            "offload_param": {
                "device": "cpu" if ${optimizationSettings.deepSpeed.offloadParams} else "none",
            },
        },
    }
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )
    `;
  }
  
  // Standard training setup with other optimizations
  else {
    code += `
    # Get the best available device
    device = get_device(prefer_mps=${optimizationSettings.deviceDetection.preferMps})
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    `;
    
    // Mixed precision
    if (optimizationSettings.mixedPrecision !== 'none') {
      code += `
    # Setup mixed precision training
    scaler = GradScaler()
    
    def train_step(model, inputs, labels):
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        with autocast(dtype=${optimizationSettings.mixedPrecision === 'fp16' ? 'torch.float16' : 'torch.bfloat16'}):
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Scale gradients and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss
    `;
    } else {
      code += `
    def train_step(model, inputs, labels):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss
    `;
    }
  }
  
  // torch.compile
  if (optimizationSettings.torchCompile) {
    code += `
    # Check if PyTorch version supports torch.compile
    if hasattr(torch, 'compile'):
        print("Using torch.compile for faster execution")
        model = torch.compile(
            model, 
            mode="${optimizationSettings.torchCompileMode}"
        )
    else:
        print("Warning: torch.compile not available in this PyTorch version. Skipping compilation.")
    `;
  }
  
  code += `
    return model, optimizer
`;

  // Add MoE implementation if enabled
  if (optimizationSettings.moe.enabled) {
    code += `

# Mixture of Experts implementation
class MixtureOfExperts(nn.Module):
    """
    Implements a Mixture of Experts layer where tokens are routed to different expert FFNs.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = ${optimizationSettings.moe.numExperts},
        top_k: int = ${optimizationSettings.moe.topK},
        activation: str = 'gelu',
        capacity_factor_train: float = ${optimizationSettings.moe.capacityFactorTrain},
        capacity_factor_eval: float = ${optimizationSettings.moe.capacityFactorEval},
        dropout: float = 0.0,
        expert_dropout: float = ${optimizationSettings.moe.expertDropout},
        use_expert_parallelism: bool = ${optimizationSettings.moe.expertParallelism},
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Can't select more experts than we have
        self.use_expert_parallelism = use_expert_parallelism
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        
        # Create the router (token-to-expert assignment)
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Create experts
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential()
            expert.add_module("linear1", nn.Linear(input_dim, hidden_dim))
            
            # Activation function
            if activation == 'relu':
                expert.add_module("activation", nn.ReLU())
            elif activation == 'gelu':
                expert.add_module("activation", nn.GELU())
            elif activation == 'silu':
                expert.add_module("activation", nn.SiLU())
            elif activation == 'tanh':
                expert.add_module("activation", nn.Tanh())
            
            if dropout > 0:
                expert.add_module("dropout", nn.Dropout(dropout))
            
            expert.add_module("linear2", nn.Linear(hidden_dim, output_dim))
            
            self.experts.append(expert)
        
        # Layer norm if requested
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        
        # Expert dropout
        self.expert_dropout = expert_dropout
        
        # Load balancing loss coefficient
        self.router_z_loss_coef = 0.001  # Penalize router logits that are too large
        self.router_aux_loss_coef = 0.01  # Encourage balanced expert assignment
    
    def _compute_router_probabilities(self, inputs):
        """
        Compute the probabilities for routing tokens to experts.
        """
        # Get router logits
        router_logits = self.router(inputs)  # [batch_size, seq_len, num_experts]
        
        # Apply router z-loss: penalize large logits to improve stability
        router_z_loss = torch.mean(torch.square(router_logits)) * self.router_z_loss_coef
        
        # Get router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        return router_logits, router_probs, router_z_loss
    
    def _compute_routing_instructions(self, router_probs, inputs_shape):
        """
        Compute routing instructions based on router probabilities.
        """
        # Get the top-k experts for each token
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize the expert weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute capacity
        # Each expert should process approximately (tokens_per_batch / num_experts * capacity_factor) tokens
        tokens_per_batch = inputs_shape[0] * inputs_shape[1]  # batch_size * seq_len
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = int(tokens_per_batch // self.num_experts * capacity_factor)
        capacity = max(capacity, 4)  # Ensure minimum capacity
        
        # Create a mask for each expert
        expert_mask = torch.zeros(
            (tokens_per_batch, self.num_experts),
            device=router_probs.device,
            dtype=torch.bool
        )
        
        # Flatten indices and create position indices
        batch_size, seq_len = inputs_shape[0], inputs_shape[1]
        flat_indices = torch.arange(batch_size * seq_len, device=router_probs.device)
        
        # Reshape expert indices and weights for easier processing
        expert_indices = expert_indices.reshape(-1, self.top_k)  # [batch_size*seq_len, top_k]
        expert_weights = expert_weights.reshape(-1, self.top_k)  # [batch_size*seq_len, top_k]
        
        # Assign tokens to experts
        for k in range(self.top_k):
            # For each token, get the k-th expert
            expert_idx = expert_indices[:, k]
            
            # Count how many tokens are routed to each expert
            expert_counts = torch.bincount(expert_idx, minlength=self.num_experts)
            
            # Identify which tokens go over capacity for their assigned expert
            over_capacity = expert_counts > capacity
            if over_capacity.any():
                # Get the experts that are over capacity
                over_capacity_experts = torch.nonzero(over_capacity).squeeze(-1)
                
                for expert_id in over_capacity_experts:
                    # Find tokens assigned to this expert
                    expert_token_indices = torch.nonzero(expert_idx == expert_id).squeeze(-1)
                    
                    # Get the weights for these tokens
                    token_weights = expert_weights[expert_token_indices, k]
                    
                    # Sort tokens by weight (lowest first)
                    sorted_indices = torch.argsort(token_weights)
                    
                    # Keep only the top 'capacity' tokens
                    keep_indices = expert_token_indices[sorted_indices[-capacity:]]
                    
                    # Set mask for kept tokens
                    expert_mask[keep_indices, expert_id] = True
            else:
                # If no expert is over capacity, keep all assignments
                expert_mask[flat_indices, expert_idx] = True
        
        # Compute load balancing auxiliary loss
        # We want each expert to receive an equal proportion of tokens
        router_probs_mean = router_probs.mean(dim=(0, 1))  # Mean probability for each expert
        aux_loss = torch.mean(router_probs_mean * router_probs_mean) * self.num_experts * self.router_aux_loss_coef
        
        return expert_mask, expert_weights, expert_indices, aux_loss
    
    def forward(self, inputs):
        """
        Forward pass through the MoE layer.
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            outputs: Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Save original shape
        original_shape = inputs.shape
        batch_size, seq_len, input_dim = original_shape
        
        # Reshape inputs for routing
        inputs_reshaped = inputs.reshape(-1, input_dim)  # [batch_size*seq_len, input_dim]
        
        # Get router probabilities
        router_logits, router_probs, router_z_loss = self._compute_router_probabilities(inputs)
        
        # Compute routing instructions
        expert_mask, expert_weights, expert_indices, aux_loss = self._compute_routing_instructions(
            router_probs, original_shape
        )
        
        # Initialize output tensor
        final_output = torch.zeros(
            (batch_size * seq_len, self.output_dim),
            device=inputs.device,
            dtype=inputs.dtype
        )
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens go to this expert
            token_indices = torch.nonzero(expert_mask[:, expert_idx]).squeeze(-1)
            
            if token_indices.numel() == 0:
                # No tokens routed to this expert
                continue
            
            # Get the corresponding inputs
            expert_inputs = inputs_reshaped[token_indices]
            
            # Apply expert dropout during training
            if self.training and self.expert_dropout > 0:
                if torch.rand(1).item() < self.expert_dropout:
                    # Skip this expert
                    continue
            
            # Process tokens with this expert
            if self.use_expert_parallelism and torch.cuda.device_count() > 1:
                # Distribute experts across devices
                device_idx = expert_idx % torch.cuda.device_count()
                device = torch.device(f'cuda:{device_idx}')
                
                # Move expert and inputs to the device
                expert_device = expert.to(device)
                expert_inputs_device = expert_inputs.to(device)
                
                # Process on device
                expert_outputs = expert_device(expert_inputs_device)
                
                # Move back to original device
                expert_outputs = expert_outputs.to(inputs.device)
            else:
                # Process normally
                expert_outputs = expert(expert_inputs)
            
            # Find the weights for these tokens
            # We need to find which of the top-k experts for each token matches this expert
            flat_token_indices = token_indices.unsqueeze(1).expand(-1, self.top_k)
            flat_expert_indices = expert_indices.gather(0, flat_token_indices)
            
            # Find where the current expert appears in the top-k for each token
            mask = (flat_expert_indices == expert_idx)
            
            # Get the corresponding weights
            token_expert_weights = expert_weights.gather(0, flat_token_indices)
            token_expert_weights = token_expert_weights * mask.float()
            
            # Sum the weights for each token (in case the same expert appears multiple times)
            token_weights = token_expert_weights.sum(dim=1).unsqueeze(1)
            
            # Scale outputs by weights and add to final output
            final_output[token_indices] += expert_outputs * token_weights
        
        # Reshape back to original dimensions
        outputs = final_output.reshape(batch_size, seq_len, self.output_dim)
        
        # Apply layer norm if specified
        if self.layer_norm is not None:
            outputs = self.layer_norm(outputs)
        
        # Save the auxiliary losses for later use
        self.router_z_loss = router_z_loss
        self.aux_loss = aux_loss
        
        return outputs
`;
  }
  
  return code;
}

/**
 * Generates device detection code
 */
function generateDeviceDetectionCode(optimizationSettings: OptimizationSettings): string {
  return `
def get_device(prefer_mps=${optimizationSettings.deviceDetection.preferMps}):
    """
    Detects and returns the best available device for PyTorch.
    
    Args:
        prefer_mps: Whether to prefer MPS (Metal Performance Shaders) on Mac over CPU
        
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        print("CUDA device detected. Using GPU.")
        return torch.device("cuda")
    
    # Check for MPS (Apple Silicon GPU)
    if prefer_mps and platform.system() == "Darwin":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Apple Silicon GPU (MPS) detected. Using MPS.")
            return torch.device("mps")
        else:
            print("MPS requested but not available. Using CPU instead.")
            if platform.processor() == 'arm':
                print("Note: You're using Apple Silicon but MPS is not available.")
                print("Make sure you have PyTorch 1.12+ installed with MPS support.")
    
    print("No GPU detected. Using CPU.")
    return torch.device("cpu")

def print_device_info():
    """
    Prints information about the available devices.
    """
    print("\nDevice Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.version()}")
    print(f"Processor: {platform.processor()}")
    
    if torch.cuda.is_available():
        print("\nCUDA Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    CUDA Capability: {props.major}.{props.minor}")
            print(f"    Multi-Processor Count: {props.multi_processor_count}")
    
    # Check for MPS (Apple Silicon GPU)
    if platform.system() == "Darwin":
        print("\nMPS (Metal Performance Shaders) Information:")
        if hasattr(torch.backends, "mps"):
            print(f"MPS available: {torch.backends.mps.is_available()}")
            print(f"MPS built: {torch.backends.mps.is_built()}")
            if torch.backends.mps.is_available():
                print("  Apple Silicon GPU detected")
                # Try to get more info about the device
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                    if result.returncode == 0:
                        total_ram = int(result.stdout.strip()) / 1e9
                        print(f"  System RAM: {total_ram:.2f} GB")
                except:
                    pass
        else:
            print("MPS not supported in this PyTorch version")
    
    print("\nCPU Information:")
    print(f"CPU count: {os.cpu_count()}")
    
    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        print(f"CPU Model: {line.split(':')[1].strip()}")
                        break
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"CPU Model: {result.stdout.strip()}")
    except:
        pass

def track_memory_usage(device=None):
    """
    Tracks and returns the current memory usage.
    
    Args:
        device: The device to track memory for
        
    Returns:
        dict: Memory usage statistics
    """
    memory_stats = {}
    
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        # CUDA memory stats
        memory_stats['allocated'] = torch.cuda.memory_allocated(device) / 1e9  # GB
        memory_stats['reserved'] = torch.cuda.memory_reserved(device) / 1e9  # GB
        memory_stats['max_allocated'] = torch.cuda.max_memory_allocated(device) / 1e9  # GB
        memory_stats['max_reserved'] = torch.cuda.max_memory_reserved(device) / 1e9  # GB
    
    # Get system memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory_stats['system_total'] = vm.total / 1e9  # GB
        memory_stats['system_available'] = vm.available / 1e9  # GB
        memory_stats['system_used'] = vm.used / 1e9  # GB
        memory_stats['system_percent'] = vm.percent  # %
    except ImportError:
        print("psutil not available. Install with 'pip install psutil' for system memory tracking.")
    
    return memory_stats

def print_memory_usage(device=None):
    """
    Prints the current memory usage.
    
    Args:
        device: The device to track memory for
    """
    memory_stats = track_memory_usage(device)
    
    print("\nMemory Usage:")
    
    if 'allocated' in memory_stats:
        print(f"CUDA Memory:")
        print(f"  Allocated: {memory_stats['allocated']:.2f} GB")
        print(f"  Reserved: {memory_stats['reserved']:.2f} GB")
        print(f"  Max Allocated: {memory_stats['max_allocated']:.2f} GB")
        print(f"  Max Reserved: {memory_stats['max_reserved']:.2f} GB")
    
    if 'system_total' in memory_stats:
        print(f"System Memory:")
        print(f"  Total: {memory_stats['system_total']:.2f} GB")
        print(f"  Available: {memory_stats['system_available']:.2f} GB")
        print(f"  Used: {memory_stats['system_used']:.2f} GB")
        print(f"  Used Percent: {memory_stats['system_percent']:.1f}%")
`;
}

/**
 * Generates experiment code with synthetic data and metrics tracking
 */
function generateExperimentCode(optimizationSettings: OptimizationSettings): string {
  return `
def generate_synthetic_data(vocab_size=10000, dataset_size=${optimizationSettings.experiment.datasetSize}, seq_length=BLOCK_SIZE):
    """Generate synthetic data for training and evaluation."""
    # Generate random token sequences
    X = torch.randint(0, vocab_size, (dataset_size, seq_length)).long()
    # For a simple language modeling task, the target is the next token
    y = torch.randint(0, vocab_size, (dataset_size,)).long()
    
    # Split into train and validation sets (90/10)
    split_idx = int(0.9 * dataset_size)
    train_data = {'input_ids': X[:split_idx], 'labels': y[:split_idx]}
    val_data = {'input_ids': X[split_idx:], 'labels': y[split_idx:]}
    
    return train_data, val_data

def run_experiment(model, epochs=None, batch_size=None, seq_length=None):
    """Run a small-scale experiment with the model."""
    if epochs is None:
        epochs = ${optimizationSettings.experiment.epochs}
    if batch_size is None:
        batch_size = BATCH_SIZE
    if seq_length is None:
        seq_length = BLOCK_SIZE
    
    print(f"\\nRunning experiment with {epochs} epochs, batch size {batch_size}, sequence length {seq_length}")
    
    # Get the appropriate device
    device = get_device(prefer_mps=${optimizationSettings.deviceDetection.preferMps})
    print_device_info()
    
    # Move model to device
    model = model.to(device)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    train_data, val_data = generate_synthetic_data(
        dataset_size=${optimizationSettings.experiment.datasetSize}, 
        seq_length=seq_length
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data['input_ids'], train_data['labels']),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data['input_ids'], val_data['labels']),
        batch_size=batch_size
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': [],
        'memory_usage': []
    }
    
    # Track initial memory usage
    if ${optimizationSettings.deviceDetection.enabled}:
        try:
            initial_memory = track_memory_usage(device)
            metrics['memory_usage'].append(initial_memory)
            print_memory_usage(initial_memory)
        except Exception as e:
            print(f"Warning: Could not track memory usage: {e}")
    
    # Create directory for experiment results
    os.makedirs('experiment_results', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape outputs if needed (depends on your model architecture)
            if outputs.dim() > 2:
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_epoch_loss += loss.item()
            
            # Track memory usage every 10 batches
            if batch_idx % 10 == 0 and ${optimizationSettings.deviceDetection.enabled}:
                try:
                    memory_stats = track_memory_usage(device)
                    metrics['memory_usage'].append(memory_stats)
                except Exception:
                    pass
        
        # Calculate average training loss
        train_epoch_loss /= len(train_loader)
        metrics['train_loss'].append(train_epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Reshape outputs if needed
                if outputs.dim() > 2:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        metrics['val_loss'].append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        metrics['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Print memory usage
        if ${optimizationSettings.deviceDetection.enabled}:
            try:
                memory_stats = track_memory_usage(device)
                print_memory_usage(memory_stats)
            except Exception:
                pass
        
        # Save checkpoint if enabled
        if ${optimizationSettings.experiment.saveCheckpoints}:
            checkpoint_path = f"experiment_results/model_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save metrics
    if ${optimizationSettings.experiment.trackMetrics}:
        try:
            import json
            import matplotlib.pyplot as plt
            
            # Save metrics to JSON
            with open('experiment_results/metrics.json', 'w') as f:
                json.dump({
                    'train_loss': metrics['train_loss'],
                    'val_loss': metrics['val_loss'],
                    'epoch_times': metrics['epoch_times'],
                    'memory_usage': metrics['memory_usage'] if 'memory_usage' in metrics else None
                }, f)
            
            # Plot training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['train_loss'], label='Training Loss')
            plt.plot(metrics['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig('experiment_results/loss_plot.png')
            
            # Plot epoch times
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['epoch_times'])
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.title('Epoch Training Time')
            plt.savefig('experiment_results/time_plot.png')
            
            # Plot memory usage if available
            if 'memory_usage' in metrics and metrics['memory_usage']:
                try:
                    # Extract GPU memory usage if available
                    if 'gpu' in metrics['memory_usage'][0]:
                        gpu_memory = [m['gpu']['used'] / (1024 ** 2) for m in metrics['memory_usage'] if 'gpu' in m]
                        plt.figure(figsize=(10, 5))
                        plt.plot(gpu_memory)
                        plt.xlabel('Measurement (every 10 batches)')
                        plt.ylabel('Memory Usage (MB)')
                        plt.title('GPU Memory Usage')
                        plt.savefig('experiment_results/gpu_memory_plot.png')
                    
                    # Extract system memory usage
                    if 'system' in metrics['memory_usage'][0]:
                        sys_memory = [m['system']['percent'] for m in metrics['memory_usage'] if 'system' in m]
                        plt.figure(figsize=(10, 5))
                        plt.plot(sys_memory)
                        plt.xlabel('Measurement (every 10 batches)')
                        plt.ylabel('Memory Usage (%)')
                        plt.title('System Memory Usage')
                        plt.savefig('experiment_results/system_memory_plot.png')
                except Exception as e:
                    print(f"Warning: Could not plot memory usage: {e}")
            
            print("Metrics saved to experiment_results/")
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    return metrics
`;
}

function generateHyperparameters(optimizationSettings?: OptimizationSettings): string {
  if (!optimizationSettings?.hyperparameters) {
    return '# Default hyperparameters\nBATCH_SIZE = 64\nBLOCK_SIZE = 256  # maximum context length for predictions\nMAX_ITERS = 5000\nEVAL_INTERVAL = 500\nLEARNING_RATE = 3e-4\nEVAL_ITERS = 200\nN_EMBD = 384\nN_HEAD = 6\nN_LAYER = 6\nDROPOUT = 0.2';
  }
  
  const hp = optimizationSettings.hyperparameters;
  
  return `# Training hyperparameters
BATCH_SIZE = ${hp.batchSize}
BLOCK_SIZE = ${hp.blockSize}  # maximum context length for predictions
MAX_ITERS = ${hp.maxIters}
EVAL_INTERVAL = ${hp.evalInterval}
LEARNING_RATE = ${hp.learningRate.toExponential()}
EVAL_ITERS = ${hp.evalIters}
N_EMBD = ${hp.nEmbd}
N_HEAD = ${hp.nHead}
N_LAYER = ${hp.nLayer}
DROPOUT = ${hp.dropout}`;
}

/**
 * Generates code for a Layer Normalization component
 */
function generateLayerNormCode(
  componentName: string,
  params: LayerNormNodeData['params']
): string {
  const { normalizedShape, eps, elementwiseAffine, bias } = params;
  
  return `        ${componentName} = nn.LayerNorm(
            normalized_shape=${normalizedShape},
            eps=${eps || 1e-5},
            elementwise_affine=${elementwiseAffine === false ? 'False' : 'True'},
            bias=${bias === false ? 'False' : 'True'}
        )`;
}

/**
 * Generates code for SFT Training
 */
function generateSFTTrainingCode(node: Node<SFTTrainingNodeData>): string {
  const { params } = node.data;
  
  return `
class SupervisedFineTuning:
    def __init__(self, model, tokenizer, dataset_path="${params.datasetPath || 'path/to/dataset'}"):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        
        # Training parameters
        self.learning_rate = ${params.learningRate}
        self.batch_size = ${params.batchSize}
        self.num_epochs = ${params.numEpochs}
        self.weight_decay = ${params.weightDecay || 0.01}
        self.max_grad_norm = ${params.maxGradNorm || 1.0}
        self.warmup_steps = ${params.warmupSteps || 500}
        
    def load_dataset(self):
        """
        Load and prepare the dataset for supervised fine-tuning
        """
        from datasets import load_dataset
        
        # Load dataset (adjust as needed for your specific dataset)
        dataset = load_dataset(self.dataset_path)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Add prompt template if needed
            inputs = self.tokenizer(examples["input"], padding="max_length", truncation=True)
            outputs = self.tokenizer(examples["output"], padding="max_length", truncation=True)
            
            # Create labels for the decoder
            inputs["labels"] = outputs["input_ids"]
            return inputs
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
        
    def train(self):
        """
        Run the supervised fine-tuning process
        """
        from transformers import Trainer, TrainingArguments
        
        # Load and prepare dataset
        tokenized_dataset = self.load_dataset()
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            ${params.optimizer ? `optim="${params.optimizer}",` : ''}
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        self.model.save_pretrained("./sft_model")
        self.tokenizer.save_pretrained("./sft_model")
        
        return self.model
`;
}

/**
 * Generates code for PPO Training
 */
function generatePPOTrainingCode(node: Node<PPOTrainingNodeData>): string {
  const { params } = node.data;
  
  return `
class PPOTrainer:
    def __init__(self, model, tokenizer, reward_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model or "${params.rewardModel || 'path/to/reward_model'}"
        
        # PPO parameters
        self.learning_rate = ${params.learningRate}
        self.batch_size = ${params.batchSize}
        self.num_epochs = ${params.numEpochs}
        self.clip_epsilon = ${params.clipEpsilon}
        self.value_coefficient = ${params.valueCoefficient}
        self.entropy_coefficient = ${params.entropyCoefficient}
        self.max_grad_norm = ${params.maxGradNorm || 1.0}
        self.discount_factor = ${params.discountFactor || 0.99}
        self.gae_lambda = ${params.gaeLambda || 0.95}
        
    def load_reward_model(self):
        """
        Load the reward model for PPO training
        """
        from transformers import AutoModelForSequenceClassification
        
        # Load reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model)
        reward_model.eval()  # Set to evaluation mode
        return reward_model
        
    def train(self, prompts):
        """
        Run the PPO training process
        """
        import torch
        from trl import PPOTrainer, PPOConfig
        
        # Load reward model
        reward_model = self.load_reward_model()
        
        # Define PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            ppo_epochs=self.num_epochs,
            clip_range=self.clip_epsilon,
            value_loss_coef=self.value_coefficient,
            entropy_coef=self.entropy_coefficient,
            max_grad_norm=self.max_grad_norm,
            gamma=self.discount_factor,
            lambda_=self.gae_lambda,
            ${params.optimizer ? `optimizer_class=torch.optim.${params.optimizer.charAt(0).toUpperCase() + params.optimizer.slice(1)},` : ''}
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # Define reward function
        def reward_fn(generated_texts):
            inputs = self.tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                rewards = reward_model(**inputs).logits[:, 0]
            return rewards
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Generate responses
            query_tensors = [self.tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
            response_tensors = []
            
            for query in query_tensors:
                response = ppo_trainer.generate(query)
                response_tensors.append(response)
            
            # Decode responses
            responses = [self.tokenizer.decode(r[0]) for r in response_tensors]
            
            # Calculate rewards
            rewards = reward_fn(responses)
            
            # Update model with PPO
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            print(f"Epoch {epoch}: {stats}")
        
        # Save the model
        self.model.save_pretrained("./ppo_model")
        self.tokenizer.save_pretrained("./ppo_model")
        
        return self.model
`;
}

/**
 * Generates code for DPO Training
 */
function generateDPOTrainingCode(node: Node<DPOTrainingNodeData>): string {
  const { params } = node.data;
  
  return `
class DPOTrainer:
    def __init__(self, model, tokenizer, reference_model="${params.referenceModelName}", dataset_path="${params.datasetPath || 'path/to/dataset'}"):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model_name = reference_model
        self.dataset_path = dataset_path
        
        # DPO parameters
        self.learning_rate = ${params.learningRate}
        self.batch_size = ${params.batchSize}
        self.num_epochs = ${params.numEpochs}
        self.beta = ${params.beta}
        self.max_grad_norm = ${params.maxGradNorm || 1.0}
        self.weight_decay = ${params.weightDecay || 0.01}
        
    def load_dataset(self):
        """
        Load and prepare the dataset for DPO training
        """
        from datasets import load_dataset
        
        # Load dataset (adjust as needed for your specific dataset)
        dataset = load_dataset(self.dataset_path)
        
        # Ensure dataset has the required columns: prompt, chosen, rejected
        if not all(col in dataset["train"].column_names for col in ["prompt", "chosen", "rejected"]):
            raise ValueError("Dataset must contain 'prompt', 'chosen', and 'rejected' columns")
            
        return dataset
        
    def train(self):
        """
        Run the DPO training process
        """
        from transformers import AutoModelForCausalLM
        from trl import DPOTrainer as TRL_DPOTrainer
        
        # Load reference model
        reference_model = AutoModelForCausalLM.from_pretrained(self.reference_model_name)
        
        # Load and prepare dataset
        dataset = self.load_dataset()
        
        # Initialize DPO trainer
        dpo_trainer = TRL_DPOTrainer(
            model=self.model,
            ref_model=reference_model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            beta=self.beta,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            ${params.optimizer ? `optim="${params.optimizer}",` : ''}
        )
        
        # Start training
        dpo_trainer.train()
        
        # Save the model
        self.model.save_pretrained("./dpo_model")
        self.tokenizer.save_pretrained("./dpo_model")
        
        return self.model
`;
}

/**
 * Generates code for GRPO Training
 */
function generateGRPOTrainingCode(node: Node<GRPOTrainingNodeData>): string {
  const { params } = node.data;
  
  return `
class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model or "${params.rewardModel || 'path/to/reward_model'}"
        
        # GRPO parameters
        self.learning_rate = ${params.learningRate}
        self.batch_size = ${params.batchSize}
        self.num_epochs = ${params.numEpochs}
        self.clip_epsilon = ${params.clipEpsilon}
        self.reward_threshold = ${params.rewardThreshold}
        self.max_grad_norm = ${params.maxGradNorm || 1.0}
        self.weight_decay = ${params.weightDecay || 0.01}
        
    def load_reward_model(self):
        """
        Load the reward model for GRPO training
        """
        from transformers import AutoModelForSequenceClassification
        
        # Load reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model)
        reward_model.eval()  # Set to evaluation mode
        return reward_model
        
    def train(self, prompts):
        """
        Run the GRPO training process
        """
        import torch
        import torch.nn.functional as F
        from torch.optim import ${params.optimizer ? params.optimizer.charAt(0).toUpperCase() + params.optimizer.slice(1) : 'AdamW'}
        
        # Load reward model
        reward_model = self.load_reward_model()
        
        # Setup optimizer
        optimizer = ${params.optimizer ? params.optimizer.charAt(0).toUpperCase() + params.optimizer.slice(1) : 'AdamW'}(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i+self.batch_size]
                
                # Tokenize prompts
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
                
                # Generate responses with the current model
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=100,
                        do_sample=True,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                
                # Get generated sequences and their log probabilities
                sequences = outputs.sequences
                log_probs = self._compute_log_probs(outputs)
                
                # Compute rewards using the reward model
                generated_texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                rewards = self._compute_rewards(generated_texts, reward_model)
                
                # Compute GRPO loss
                loss = self._compute_grpo_loss(log_probs, rewards)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch}: Loss = {total_loss}")
        
        # Save the model
        self.model.save_pretrained("./grpo_model")
        self.tokenizer.save_pretrained("./grpo_model")
        
        return self.model
        
    def _compute_log_probs(self, outputs):
        """
        Compute log probabilities of generated sequences
        """
        # This is a simplified implementation
        # In practice, you would need to compute token-by-token log probs
        return torch.mean(outputs.scores, dim=1)
        
    def _compute_rewards(self, texts, reward_model):
        """
        Compute rewards for generated texts using the reward model
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            rewards = reward_model(**inputs).logits[:, 0]
        return rewards
        
    def _compute_grpo_loss(self, log_probs, rewards):
        """
        Compute the GRPO loss
        """
        # Determine which samples exceed the reward threshold
        above_threshold = rewards >= self.reward_threshold
        
        # For samples above threshold, maximize probability (minimize negative log prob)
        # For samples below threshold, apply penalty proportional to how far below threshold
        penalty = torch.clamp(self.reward_threshold - rewards, min=0)
        
        # Compute loss with clipping (similar to PPO)
        loss = -log_probs * above_threshold + self.clip_epsilon * penalty * (~above_threshold)
        
        return loss.mean()
`;
}

/**
 * Sorts training nodes in the correct sequence (SFT -> PPO -> DPO -> GRPO)
 */
function sortTrainingNodesBySequence(trainingNodes: Node<LLMNodeData>[]): Node<LLMNodeData>[] {
  const nodeTypeOrder = {
    'sftTraining': 0,
    'ppoTraining': 1,
    'dpoTraining': 2,
    'grpoTraining': 3
  };
  
  return [...trainingNodes].sort((a, b) => {
    return nodeTypeOrder[a.data.type as keyof typeof nodeTypeOrder] - 
           nodeTypeOrder[b.data.type as keyof typeof nodeTypeOrder];
  });
}

/**
 * Generates code for the post-training pipeline
 */
function generatePostTrainingPipeline(trainingNodes: Node<LLMNodeData>[]): string {
  if (trainingNodes.length === 0) return '';
  
  // Generate imports for training modules
  const imports = trainingNodes.map(node => {
    switch (node.data.type) {
      case 'sftTraining':
        return 'from sft_module import SupervisedFineTuning';
      case 'ppoTraining':
        return 'from ppo_module import PPOTrainer';
      case 'dpoTraining':
        return 'from dpo_module import DPOTrainer';
      case 'grpoTraining':
        return 'from grpo_module import GRPOTrainer';
      default:
        return '';
    }
  }).filter(Boolean).join('\n');
  
  // Generate training stages
  const trainingStages = trainingNodes.map(node => {
    switch (node.data.type) {
      case 'sftTraining':
        return generateSFTTrainingStage(node as Node<SFTTrainingNodeData>);
      case 'ppoTraining':
        return generatePPOTrainingStage(node as Node<PPOTrainingNodeData>);
      case 'dpoTraining':
        return generateDPOTrainingStage(node as Node<DPOTrainingNodeData>);
      case 'grpoTraining':
        return generateGRPOTrainingStage(node as Node<GRPOTrainingNodeData>);
      default:
        return '';
    }
  }).join('\n\n    ');
  
  return `
# Post-training pipeline
${imports}

def run_post_training_pipeline(base_model):
    """
    Complete post-training pipeline that applies multiple training methods in sequence:
    ${trainingNodes.map(node => `- ${node.data.label}`).join('\n    ')}
    
    Args:
        base_model: The base model to start with
    
    Returns:
        The final trained model
    """
    print("\\n" + "="*50)
    print("Starting post-training pipeline...")
    print("="*50)
    
    # Make a copy of the base model for training
    import copy
    model = copy.deepcopy(base_model)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # Replace with appropriate tokenizer
    
    ${trainingStages}
    
    # Save the final model
    final_model_path = "./final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, tokenizer
`;
}

/**
 * Generates code for SFT training stage
 */
function generateSFTTrainingStage(node: Node<SFTTrainingNodeData>): string {
  const params = node.data.params;
  
  return `# 1. Supervised Fine-Tuning (SFT)
    print("\\nStep 1: Starting Supervised Fine-Tuning (SFT)")
    
    sft_trainer = SupervisedFineTuning(
        model=model,
        tokenizer=tokenizer,
        dataset_path="${params.datasetPath}",
        learning_rate=${params.learningRate},
        batch_size=${params.batchSize},
        num_epochs=${params.numEpochs},
        weight_decay=${params.weightDecay},
        max_grad_norm=${params.maxGradNorm},
        warmup_steps=${params.warmupSteps},
        optimizer="${params.optimizer}",
        loss_function="${params.lossFunction}"
    )
    model = sft_trainer.train()
    print("SFT training completed.")`;
}

/**
 * Generates code for PPO training stage
 */
function generatePPOTrainingStage(node: Node<PPOTrainingNodeData>): string {
  const params = node.data.params;
  
  return `# 2. Proximal Policy Optimization (PPO)
    print("\\nStep 2: Starting Proximal Policy Optimization (PPO)")
    
    # Sample prompts for PPO training
    prompts = [
        "Write a story about a robot learning to feel emotions.",
        "Explain quantum computing to a 5-year-old.",
        "Compose a poem about the changing seasons.",
        # Add more prompts as needed
    ]
    
    ppo_trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model="${params.rewardModel}",
        learning_rate=${params.learningRate},
        batch_size=${params.batchSize},
        num_epochs=${params.numEpochs},
        clip_epsilon=${params.clipEpsilon},
        value_coefficient=${params.valueCoefficient},
        entropy_coefficient=${params.entropyCoefficient},
        max_grad_norm=${params.maxGradNorm},
        discount_factor=${params.discountFactor},
        gae_lambda=${params.gaeLambda},
        optimizer="${params.optimizer}"
    )
    model = ppo_trainer.train(prompts)
    print("PPO training completed.")`;
}

/**
 * Generates code for DPO training stage
 */
function generateDPOTrainingStage(node: Node<DPOTrainingNodeData>): string {
  const params = node.data.params;
  
  return `# 3. Direct Preference Optimization (DPO)
    print("\\nStep 3: Starting Direct Preference Optimization (DPO)")
    
    dpo_trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reference_model="${params.referenceModelName}",
        dataset_path="${params.datasetPath}",
        learning_rate=${params.learningRate},
        batch_size=${params.batchSize},
        num_epochs=${params.numEpochs},
        beta=${params.beta},
        max_grad_norm=${params.maxGradNorm},
        weight_decay=${params.weightDecay},
        optimizer="${params.optimizer}"
    )
    model = dpo_trainer.train()
    print("DPO training completed.")`;
}

/**
 * Generates code for GRPO training stage
 */
function generateGRPOTrainingStage(node: Node<GRPOTrainingNodeData>): string {
  const params = node.data.params;
  
  return `# 4. Generalized Reward-Penalized Optimization (GRPO)
    print("\\nStep 4: Starting Generalized Reward-Penalized Optimization (GRPO)")
    
    # Additional prompts for GRPO training
    grpo_prompts = [
        "Provide a balanced analysis of climate change policies.",
        "Discuss the ethical implications of artificial intelligence.",
        "Explain the pros and cons of remote work.",
        # Add more prompts as needed
    ]
    
    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model="${params.rewardModel}",
        learning_rate=${params.learningRate},
        batch_size=${params.batchSize},
        num_epochs=${params.numEpochs},
        clip_epsilon=${params.clipEpsilon},
        reward_threshold=${params.rewardThreshold},
        max_grad_norm=${params.maxGradNorm},
        weight_decay=${params.weightDecay},
        optimizer="${params.optimizer}"
    )
    model = grpo_trainer.train(grpo_prompts)
    print("GRPO training completed.")`;
} 