"use client";

import { Node } from 'reactflow';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LLMNodeData, NodeType } from '@/lib/types';
import { useState, useEffect, useRef } from 'react';

interface NodePanelProps {
  setNodes: React.Dispatch<React.SetStateAction<Node<LLMNodeData>[]>>;
  activeTab: string;
  setActiveTab: React.Dispatch<React.SetStateAction<string>>;
}

interface NodeTemplate {
  type: NodeType;
  label: string;
  description: string;
  icon: string;
  defaultParams: Record<string, unknown>;
}

const nodeTemplates: NodeTemplate[] = [
  {
    type: 'embedding',
    label: 'Embedding',
    description: 'Converts token IDs to embeddings',
    icon: '📊',
    defaultParams: {
      vocabSize: 50000,
      embeddingDim: 512,
      paddingIdx: 0,
    },
  },
  {
    type: 'positionalEncoding',
    label: 'Positional Encoding',
    description: 'Adds position information to embeddings',
    icon: '📍',
    defaultParams: {
      encodingType: 'sinusoidal',
      embeddingDim: 512,
      maxSeqLength: 1024,
      dropout: 0.1,
    },
  },
  {
    type: 'qkvAttention',
    label: 'Multi-Head Attention',
    description: 'Self-attention mechanism',
    icon: '🔍',
    defaultParams: {
      embeddingDim: 512,
      numHeads: 8,
      dropout: 0.1,
      batchFirst: true,
      attentionType: 'scaled_dot_product',
      causal: false,
    },
  },
  {
    type: 'ffn',
    label: 'Feed Forward Network',
    description: 'Processes features with non-linearity',
    icon: '🔄',
    defaultParams: {
      inputDim: 512,
      hiddenDim: 2048,
      outputDim: 512,
      dropout: 0.1,
      activation: 'gelu',
      layerNorm: true,
    },
  },
  {
    type: 'layerNorm',
    label: 'Layer Normalization',
    description: 'Normalizes features across channels',
    icon: '📏',
    defaultParams: {
      normalizedShape: 512,
      eps: 1e-5,
      elementwiseAffine: true,
      bias: true,
    },
  },
  {
    type: 'output',
    label: 'Output Layer',
    description: 'Final projection layer',
    icon: '🎯',
    defaultParams: {
      inputDim: 512,
      outputDim: 50000,
      activation: 'softmax',
    },
  },
];

// Define training node templates
const trainingNodeTemplates: NodeTemplate[] = [
  {
    type: 'sftTraining',
    label: 'SFT Training',
    description: 'Supervised Fine-Tuning',
    icon: '🧠',
    defaultParams: {
      learningRate: 5e-5,
      batchSize: 16,
      numEpochs: 3,
      weightDecay: 0.01,
      maxGradNorm: 1.0,
      warmupSteps: 500,
      optimizer: 'adamw',
      lossFunction: 'crossentropy',
      datasetPath: './data/sft_dataset.json',
    },
  },
  {
    type: 'ppoTraining',
    label: 'PPO Training',
    description: 'Proximal Policy Optimization',
    icon: '🤖',
    defaultParams: {
      learningRate: 1e-5,
      batchSize: 8,
      numEpochs: 4,
      clipEpsilon: 0.2,
      valueCoefficient: 0.5,
      entropyCoefficient: 0.01,
      maxGradNorm: 1.0,
      discountFactor: 0.99,
      gaeLambda: 0.95,
      optimizer: 'adamw',
      rewardModel: 'reward_model.pt',
    },
  },
  {
    type: 'dpoTraining',
    label: 'DPO Training',
    description: 'Direct Preference Optimization',
    icon: '📈',
    defaultParams: {
      learningRate: 2e-5,
      batchSize: 16,
      numEpochs: 3,
      beta: 0.1,
      referenceModelName: 'reference_model.pt',
      maxGradNorm: 1.0,
      weightDecay: 0.01,
      optimizer: 'adamw',
      datasetPath: './data/preference_dataset.json',
    },
  },
  {
    type: 'grpoTraining',
    label: 'GRPO Training',
    description: 'Generalized Reward-Penalized Optimization',
    icon: '🎯',
    defaultParams: {
      learningRate: 3e-5,
      batchSize: 8,
      numEpochs: 3,
      clipEpsilon: 0.2,
      rewardThreshold: 0.5,
      maxGradNorm: 1.0,
      weightDecay: 0.01,
      optimizer: 'adamw',
      rewardModel: 'reward_model.pt',
    },
  },
];

export function NodePanel({ setNodes, activeTab, setActiveTab }: NodePanelProps) {
  // Add a memoized state to maintain component persistence across renders
  const [componentsState] = useState<{
    postTraining: { visible: boolean };
    components: { visible: boolean };
  }>({
    postTraining: { visible: activeTab === 'postTraining' },
    components: { visible: activeTab === 'components' },
  });

  // Store the node templates in a ref to prevent re-creation on each render
  const nodeTemplatesRef = useRef(nodeTemplates);
  const trainingNodeTemplatesRef = useRef(trainingNodeTemplates);

  useEffect(() => {
    // Update visibility state when tab changes
    componentsState.postTraining.visible = activeTab === 'postTraining';
    componentsState.components.visible = activeTab === 'components';
    
    // Save the current state to localStorage
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') return;
      
      localStorage.setItem('llm-graph-component-state', JSON.stringify(componentsState));
    } catch (error) {
      console.error('Error saving component state to localStorage:', error);
    }
  }, [activeTab, componentsState]);

  // Load component state from localStorage on initial render
  useEffect(() => {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') return;
      
      const savedState = localStorage.getItem('llm-graph-component-state');
      if (savedState) {
        const parsedState = JSON.parse(savedState);
        if (parsedState.templates) {
          // Handle migration from old state format
          componentsState.postTraining.visible = parsedState.templates.visible;
        } else if (parsedState.postTraining) {
          componentsState.postTraining.visible = parsedState.postTraining.visible;
        }
        componentsState.components.visible = parsedState.components.visible;
      }
    } catch (error) {
      console.error('Error loading component state from localStorage:', error);
    }
  }, [componentsState]);

  const onDragStart = (event: React.DragEvent, nodeType: NodeType, nodeData: LLMNodeData) => {
    try {
      const serializedData = JSON.stringify(nodeData);
      
      // Set data in both formats for maximum compatibility
      event.dataTransfer.setData('application/reactflow', serializedData);
      event.dataTransfer.setData('text/plain', serializedData);
      event.dataTransfer.effectAllowed = 'move';
    } catch (error) {
      console.error('Error in onDragStart:', error);
    }
  };

  const addNodeToCanvas = (nodeTemplate: NodeTemplate) => {
    const nodeData: LLMNodeData = {
      label: nodeTemplate.label,
      type: nodeTemplate.type,
      params: { ...nodeTemplate.defaultParams },
    };

    const newNode: Node<LLMNodeData> = {
      id: `${nodeTemplate.type}-${Date.now()}`,
      type: nodeTemplate.type,
      position: { x: 100, y: 100 },
      data: nodeData,
    };

    setNodes((nds) => {
      const updatedNodes = nds.concat(newNode);
      // Save to localStorage immediately
      try {
        // Check if we're in a browser environment
        if (typeof window !== 'undefined') {
          localStorage.setItem('llm-graph-nodes', JSON.stringify(updatedNodes));
        }
      } catch (error) {
        console.error('Error saving nodes to localStorage:', error);
      }
      return updatedNodes;
    });
  };

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-lg font-semibold mb-4">Components</h3>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="components">Components</TabsTrigger>
          <TabsTrigger value="postTraining">Post Training</TabsTrigger>
        </TabsList>
        
        <TabsContent 
          value="components" 
          className="flex-1 overflow-y-auto"
          forceMount
          style={{ display: activeTab === 'components' ? 'block' : 'none' }}
        >
          <div className="space-y-3">
            {nodeTemplatesRef.current.map((template) => (
              <Card 
                key={template.type}
                className="p-3 cursor-grab bg-slate-700 hover:bg-slate-600 transition-colors"
                draggable
                onDragStart={(event) => onDragStart(event, template.type, {
                  label: template.label,
                  type: template.type,
                  params: { ...template.defaultParams },
                })}
                onClick={() => addNodeToCanvas(template)}
              >
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{template.icon}</div>
                  <div>
                    <h4 className="font-medium">{template.label}</h4>
                    <p className="text-xs text-slate-300">{template.description}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent 
          value="postTraining" 
          className="flex-1 overflow-y-auto"
          forceMount
          style={{ display: activeTab === 'postTraining' ? 'block' : 'none' }}
        >
          <div className="space-y-3">
            {trainingNodeTemplatesRef.current.map((template) => (
              <Card 
                key={template.type}
                className="p-3 cursor-grab bg-slate-700 hover:bg-slate-600 transition-colors"
                draggable
                onDragStart={(event) => onDragStart(event, template.type, {
                  label: template.label,
                  type: template.type,
                  params: { ...template.defaultParams },
                })}
                onClick={() => addNodeToCanvas(template)}
              >
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{template.icon}</div>
                  <div>
                    <h4 className="font-medium">{template.label}</h4>
                    <p className="text-xs text-slate-300">{template.description}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
      
      <div className="mt-4 pt-4 border-t border-slate-700">
        <Button 
          variant="outline" 
          className="w-full mb-2" 
          size="sm"
          onClick={() => {
            // Check if we're in a browser environment
            if (typeof window === 'undefined') return;
            
            // Confirm before clearing the canvas
            if (window.confirm('Are you sure you want to clear the canvas? This action cannot be undone.')) {
              setNodes([]);
              // Also clear the nodes from localStorage
              localStorage.setItem('llm-graph-nodes', JSON.stringify([]));
              localStorage.setItem('llm-graph-edges', JSON.stringify([]));
            }
          }}
        >
          Clear Canvas
        </Button>
        
        <Button 
          variant="outline" 
          className="w-full" 
          size="sm"
          onClick={() => {
            // Check if we're in a browser environment
            if (typeof window === 'undefined') return;
            
            // Confirm before resetting storage
            if (window.confirm('Are you sure you want to reset all stored data? This will clear your canvas and all saved preferences.')) {
              // Clear all app-related localStorage items
              localStorage.removeItem('llm-graph-nodes');
              localStorage.removeItem('llm-graph-edges');
              localStorage.removeItem('llm-graph-active-tab');
              localStorage.removeItem('llm-graph-component-state');
              
              // Reload the page to apply changes
              window.location.reload();
            }
          }}
        >
          Reset Storage
        </Button>
      </div>
    </div>
  );
} 