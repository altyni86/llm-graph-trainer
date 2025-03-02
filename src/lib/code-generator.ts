import { Node, Edge } from 'reactflow';
import { 
  LLMNodeData, 
  EmbeddingNodeData, 
  PositionalEncodingNodeData,
  QKVAttentionNodeData,
  FFNNodeData,
  OutputNodeData,
  LayerNormNodeData
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
  // Sort nodes by dependencies
  const sortedNodes = sortNodesByDependencies(nodes, edges);
  
  // Generate code
  const imports = generateImports(optimizationSettings);
  const hyperparameters = generateHyperparameters(optimizationSettings);
  const deviceDetection = generateDeviceDetectionCode(optimizationSettings || defaultOptimizationSettings);
  const componentDefinitions = sortedNodes.map(node => 
    generateComponentDefinition(node, optimizationSettings)
  ).join('\n\n');
  const modelClass = `class LLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        ${sortedNodes.map(node => {
          const componentName = `self.${node.id}`;
          return `${componentName} = ${node.id.charAt(0).toUpperCase() + node.id.slice(1)}()`;
        }).join('\n        ')}
        
    def forward(self, x):
${generateForwardPass(sortedNodes, edges, optimizationSettings)}
        return x`;

  const trainingCode = optimizationSettings ? generateTrainingCode(optimizationSettings) : '';
  const experimentCode = optimizationSettings?.experiment?.enabled ? 
    generateExperimentCode(optimizationSettings) : '';

  return `${imports}

${hyperparameters}

${deviceDetection}

${componentDefinitions}

${modelClass}

${trainingCode}

${experimentCode}

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
  }
  
  return `class ${node.id.charAt(0).toUpperCase() + node.id.slice(1)}(nn.Module):
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
    
    // Generate output variable name
    const outputVar = index === sortedNodes.length - 1 ? 'x' : `x_${nodeId}`;
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