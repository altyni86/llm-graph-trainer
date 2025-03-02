// Node Types
export type NodeType = 
  | 'embedding' 
  | 'positionalEncoding' 
  | 'qkvAttention' 
  | 'ffn' 
  | 'output'
  | 'layerNorm';

// Base node data interface
export interface LLMNodeData {
  label: string;
  type: NodeType;
  params: Record<string, unknown>;
  connectionErrors?: string[];
}

// Embedding node data
export interface EmbeddingNodeData extends LLMNodeData {
  type: 'embedding';
  params: {
    vocabSize: number;
    embeddingDim: number;
    paddingIdx?: number;
    maxNorm?: number;
    normType?: number;
    scaleGradByFreq?: boolean;
    sparse?: boolean;
  };
}

// Positional Encoding node data
export interface PositionalEncodingNodeData extends LLMNodeData {
  type: 'positionalEncoding';
  params: {
    encodingType: 'sinusoidal' | 'learned' | 'rotary' | 'alibi';
    embeddingDim: number;
    maxSeqLength: number;
    dropout?: number;
  };
}

// QKV Attention node data
export interface QKVAttentionNodeData extends LLMNodeData {
  type: 'qkvAttention';
  params: {
    embeddingDim: number;
    numHeads: number;
    dropout?: number;
    batchFirst?: boolean;
    attentionType?: 'dot_product' | 'scaled_dot_product' | 'additive';
    causal?: boolean;
  };
}

// Feed Forward Network node data
export interface FFNNodeData extends LLMNodeData {
  type: 'ffn';
  params: {
    inputDim: number;
    hiddenDim: number;
    outputDim: number;
    dropout?: number;
    activation?: 'relu' | 'gelu' | 'silu' | 'tanh';
    layerNorm?: boolean;
  };
}

// Output node data
export interface OutputNodeData extends LLMNodeData {
  type: 'output';
  params: {
    inputDim: number;
    outputDim: number;
    activation?: 'softmax' | 'sigmoid' | 'tanh' | 'none';
  };
}

// Layer Normalization node data
export interface LayerNormNodeData extends LLMNodeData {
  type: 'layerNorm';
  params: {
    normalizedShape: number;
    eps?: number;
    elementwiseAffine?: boolean;
    bias?: boolean;
  };
} 