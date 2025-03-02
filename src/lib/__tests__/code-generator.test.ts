import { describe, it, expect } from 'vitest';
import { generatePythonCode } from '../code-generator';
import { Node, Edge } from 'reactflow';
import { LLMNodeData } from '../types';
import { OptimizationSettings } from '@/components/optimization-panel/OptimizationPanel';

describe('Code Generator', () => {
  it('generates basic code for a simple model', () => {
    const nodes: Node<LLMNodeData>[] = [
      {
        id: 'embedding-1',
        type: 'custom',
        position: { x: 0, y: 0 },
        data: {
          type: 'embedding',
          label: 'Embedding',
          params: {
            vocabSize: 50000,
            embeddingDim: 512
          }
        }
      },
      {
        id: 'ffn-1',
        type: 'custom',
        position: { x: 0, y: 100 },
        data: {
          type: 'ffn',
          label: 'Feed Forward',
          params: {
            hiddenDim: 2048,
            activation: 'gelu',
            useMoE: false,
            numExperts: 8,
            topK: 2
          }
        }
      }
    ];
    
    const edges: Edge[] = [
      {
        id: 'edge-1',
        source: 'embedding-1',
        target: 'ffn-1',
        sourceHandle: 'output',
        targetHandle: 'input'
      }
    ];
    
    const optimizationSettings: OptimizationSettings = {
      fsdp: { enabled: false, shardingStrategy: 'FULL_SHARD', autoWrap: false, minNumParams: 1000000 },
      deepSpeed: { enabled: false, stageThree: false, offloadOptimizer: false, offloadParams: false },
      moe: { 
        enabled: false, 
        numExperts: 8, 
        topK: 2, 
        capacityFactorTrain: 1.25, 
        capacityFactorEval: 2.0, 
        expertParallelism: false, 
        expertDropout: 0.1 
      },
      hyperparameters: {
        batchSize: 32,
        blockSize: 1024,
        maxIters: 5000,
        evalInterval: 500,
        learningRate: 0.0003,
        evalIters: 200,
        nEmbd: 512,
        nHead: 8,
        nLayer: 6,
        dropout: 0.1
      },
      flashAttention: false,
      xformers: false,
      gradientCheckpointing: false,
      mixedPrecision: 'none',
      torchCompile: false,
      torchCompileMode: 'default',
      deviceDetection: { enabled: true, preferMps: false },
      experiment: {
        enabled: false,
        batchSize: 32,
        epochs: 3,
        trackMetrics: true,
        saveCheckpoints: true,
        generateSyntheticData: true,
        datasetSize: 10000,
        sequenceLength: 512
      }
    };
    
    const code = generatePythonCode(nodes, edges, optimizationSettings);
    
    // Check for class definitions with the correct ID format
    expect(code).toContain('class Embedding-1(nn.Module):');
    expect(code).toContain('num_embeddings=50000');
    expect(code).toContain('embedding_dim=512');
    
    expect(code).toContain('class Ffn-1(nn.Module):');
    expect(code).toContain('nn.GELU()');
  });

  it('generates code with LayerNorm component', () => {
    const nodes: Node<LLMNodeData>[] = [
      {
        id: 'embedding-1',
        type: 'custom',
        position: { x: 0, y: 0 },
        data: {
          type: 'embedding',
          label: 'Embedding',
          params: {
            vocabSize: 50000,
            embeddingDim: 512
          }
        }
      },
      {
        id: 'layernorm-1',
        type: 'custom',
        position: { x: 0, y: 100 },
        data: {
          type: 'layerNorm',
          label: 'Layer Norm',
          params: {
            normalizedShape: 512,
            eps: 1e-5,
            elementwiseAffine: true,
            bias: true
          }
        }
      },
      {
        id: 'ffn-1',
        type: 'custom',
        position: { x: 0, y: 200 },
        data: {
          type: 'ffn',
          label: 'Feed Forward',
          params: {
            hiddenDim: 2048,
            activation: 'gelu',
            useMoE: false,
            numExperts: 8,
            topK: 2
          }
        }
      }
    ];
    
    const edges: Edge[] = [
      {
        id: 'edge-1',
        source: 'embedding-1',
        target: 'layernorm-1',
        sourceHandle: 'output',
        targetHandle: 'input'
      },
      {
        id: 'edge-2',
        source: 'layernorm-1',
        target: 'ffn-1',
        sourceHandle: 'output',
        targetHandle: 'input'
      }
    ];
    
    const optimizationSettings: OptimizationSettings = {
      fsdp: { enabled: false, shardingStrategy: 'FULL_SHARD', autoWrap: false, minNumParams: 1000000 },
      deepSpeed: { enabled: false, stageThree: false, offloadOptimizer: false, offloadParams: false },
      moe: { 
        enabled: false, 
        numExperts: 8, 
        topK: 2, 
        capacityFactorTrain: 1.25, 
        capacityFactorEval: 2.0, 
        expertParallelism: false, 
        expertDropout: 0.1 
      },
      hyperparameters: {
        batchSize: 32,
        blockSize: 1024,
        maxIters: 5000,
        evalInterval: 500,
        learningRate: 0.0003,
        evalIters: 200,
        nEmbd: 512,
        nHead: 8,
        nLayer: 6,
        dropout: 0.1
      },
      flashAttention: false,
      xformers: false,
      gradientCheckpointing: false,
      mixedPrecision: 'none',
      torchCompile: false,
      torchCompileMode: 'default',
      deviceDetection: { enabled: true, preferMps: false },
      experiment: {
        enabled: false,
        batchSize: 32,
        epochs: 3,
        trackMetrics: true,
        saveCheckpoints: true,
        generateSyntheticData: true,
        datasetSize: 10000,
        sequenceLength: 512
      }
    };
    
    const code = generatePythonCode(nodes, edges, optimizationSettings);
    
    // Check for class definitions with the correct ID format
    expect(code).toContain('class Layernorm-1(nn.Module):');
    expect(code).toContain('normalized_shape=512');
    expect(code).toContain('eps=0.00001');
    expect(code).toContain('elementwise_affine=True');
    expect(code).toContain('bias=True');
    
    // Check that the model initialization includes the LayerNorm
    expect(code).toContain('self.layernorm-1 = Layernorm-1()');
    
    // Check that the forward method connects the components correctly
    expect(code).toMatch(/x_1 = self\.embedding-1\(x\)/);
    expect(code).toMatch(/x_2 = self\.layernorm-1\(x_1\)/);
    expect(code).toMatch(/x_3 = self\.ffn-1\(x_2\)/);
  });
}); 