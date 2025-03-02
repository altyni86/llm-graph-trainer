"use client";

import { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  // Edge removed to fix linter error
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  ConnectionLineType,
  Panel,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Button } from '@/components/ui/button';
import { NodePanel } from './NodePanel';
import { generatePythonCode } from '@/lib/code-generator';
import { LLMNodeData } from '@/lib/types';
import { EmbeddingNode } from './nodes/EmbeddingNode';
import { PositionalEncodingNode } from './nodes/PositionalEncodingNode';
import { QKVAttentionNode } from './nodes/QKVAttentionNode';
import { FFNNode } from './nodes/FFNNode';
import { OutputNode } from './nodes/OutputNode';
import { LayerNormNode } from './nodes/LayerNormNode';
import { OptimizationSettings } from '@/components/optimization-panel/OptimizationPanel';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

// Register custom node types
const nodeTypes = {
  embedding: EmbeddingNode,
  positionalEncoding: PositionalEncodingNode,
  qkvAttention: QKVAttentionNode,
  ffn: FFNNode,
  output: OutputNode,
  layerNorm: LayerNormNode,
};

interface FlowEditorProps {
  onGenerateCode: (code: string) => void;
  optimizationSettings?: OptimizationSettings;
}

// Node Properties Component
function NodeProperties({ node, onChange }: { 
  node: Node<LLMNodeData>; 
  onChange: (id: string, data: LLMNodeData) => void;
}) {
  console.log('NodeProperties rendering for node:', node.id, node.data);
  
  // Create a memoized version of the node data to prevent unnecessary re-renders
  const memoizedNodeData = useCallback(() => {
    return {
      ...node.data,
      params: { ...node.data.params }
    };
  }, [node.data]);
  
  const handleParamChange = (paramName: string, value: unknown) => {
    console.log(`Parameter ${paramName} changing to:`, value);
    
    // Ensure the value has the correct type
    let typedValue = value;
    if (typeof value === 'string' && !isNaN(Number(value))) {
      // Convert numeric strings to numbers
      typedValue = Number(value);
    } else if (value === 'true' || value === 'false') {
      // Convert string booleans to actual booleans
      typedValue = value === 'true';
    }
    
    const updatedData = {
      ...node.data,
      params: {
        ...node.data.params,
        [paramName]: typedValue
      }
    };
    
    console.log('Updated node data:', updatedData);
    onChange(node.id, updatedData);
  };
  
  const renderParamField = (paramName: string, value: unknown) => {
    // Get parameter type
    const type = typeof value;
    console.log(`renderParamField for ${paramName}:`, value, 'type:', type);
    
    // Handle special cases for known boolean parameters
    if (['useFlashAttention', 'useSlidingWindow', 'useMoE', 'elementwiseAffine', 'bias'].includes(paramName)) {
      console.log(`Rendering special boolean field for ${paramName}`);
      return (
        <div className="mb-4 flex items-center justify-between" key={paramName}>
          <Label className="text-sm">
            {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
          </Label>
          <Switch
            checked={!!value}
            onCheckedChange={(checked) => handleParamChange(paramName, checked)}
          />
        </div>
      );
    }
    
    // Handle special cases for known numeric parameters
    if (['windowSize', 'numExperts', 'topK', 'eps'].includes(paramName)) {
      console.log(`Rendering special numeric field for ${paramName}`);
      const numValue = typeof value === 'number' ? value : Number(value || 0);
      return (
        <div className="mb-4" key={paramName}>
          <Label className="mb-2 block text-sm">
            {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
          </Label>
          <Input
            type="number"
            value={numValue}
            onChange={(e) => handleParamChange(paramName, Number(e.target.value))}
            className="bg-slate-700 border-slate-600"
            step={paramName === 'eps' ? "0.000001" : "1"}
            min={paramName === 'eps' ? "0.000001" : "0"}
          />
        </div>
      );
    }
    
    switch (type) {
      case 'number':
        console.log(`Rendering number field for ${paramName}`);
        return (
          <div className="mb-4" key={paramName}>
            <Label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Input
              type="number"
              value={value as number}
              onChange={(e) => handleParamChange(paramName, Number(e.target.value))}
              className="bg-slate-700 border-slate-600"
            />
          </div>
        );
        
      case 'boolean':
        console.log(`Rendering boolean field for ${paramName}`);
        return (
          <div className="mb-4 flex items-center justify-between" key={paramName}>
            <Label className="text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Switch
              checked={value as boolean}
              onCheckedChange={(checked) => handleParamChange(paramName, checked)}
            />
          </div>
        );
        
      case 'string':
        console.log(`Rendering string field for ${paramName}`);
        // Check if this is an enum-like field (activation, encodingType, etc.)
        if (
          ['activation', 'encodingType', 'attentionType'].includes(paramName) ||
          paramName.toLowerCase().includes('type')
        ) {
          let options: string[] = [];
          
          // Define options based on parameter name
          if (paramName === 'activation') {
            options = ['relu', 'gelu', 'silu', 'tanh', 'swish'];
          } else if (paramName === 'encodingType') {
            options = ['sinusoidal', 'learned', 'rotary', 'alibi'];
          } else if (paramName === 'attentionType') {
            options = ['dot_product', 'scaled_dot_product', 'additive'];
          }
          
          console.log(`Rendering select field for ${paramName} with options:`, options);
          return (
            <div className="mb-4" key={paramName}>
              <Label className="mb-2 block text-sm">
                {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </Label>
              <Select
                value={value as string}
                onValueChange={(val) => handleParamChange(paramName, val)}
              >
                <SelectTrigger className="bg-slate-700 border-slate-600">
                  <SelectValue placeholder={`Select ${paramName}`} />
                </SelectTrigger>
                <SelectContent>
                  {options.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          );
        }
        
        // Default string input
        return (
          <div className="mb-4" key={paramName}>
            <Label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Input
              type="text"
              value={value as string}
              onChange={(e) => handleParamChange(paramName, e.target.value)}
              className="bg-slate-700 border-slate-600"
            />
          </div>
        );
        
      default:
        console.log(`Rendering default field for ${paramName} with unknown type:`, type);
        return (
          <div className="mb-4" key={paramName}>
            <Label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Input
              type="text"
              value={JSON.stringify(value)}
              onChange={(e) => {
                try {
                  handleParamChange(paramName, JSON.parse(e.target.value));
                } catch (error) {
                  console.error('Invalid JSON:', error);
                }
              }}
              className="bg-slate-700 border-slate-600"
            />
          </div>
        );
    }
  };
  
  // Add node-type specific optimizations
  const renderOptimizations = () => {
    console.log('renderOptimizations called for node type:', node.data.type);
    console.log('Node params:', node.data.params);
    
    // Get the node data with proper type handling
    const nodeData = memoizedNodeData();
    
    switch (node.data.type) {
      case 'qkvAttention':
        console.log('Rendering qkvAttention optimizations');
        
        // Ensure proper boolean values
        const useFlashAttention = !!nodeData.params.useFlashAttention;
        const useSlidingWindow = !!nodeData.params.useSlidingWindow;
        const windowSize = typeof nodeData.params.windowSize === 'number' 
          ? nodeData.params.windowSize 
          : Number(nodeData.params.windowSize || 512);
        
        console.log('useFlashAttention:', useFlashAttention);
        console.log('useSlidingWindow:', useSlidingWindow);
        console.log('windowSize:', windowSize);
        
        return (
          <div className="mt-6 border-t border-slate-700 pt-4">
            <h4 className="text-md font-semibold mb-4">Optimizations</h4>
            <div className="mb-4 flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="text-sm">Flash Attention</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Enables Flash Attention for faster computation with reduced memory usage.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Switch
                checked={useFlashAttention}
                onCheckedChange={(checked: boolean) => handleParamChange('useFlashAttention', checked)}
              />
            </div>
            <div className="mb-4 flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="text-sm">Sliding Window</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Applies sliding window attention to reduce computation for long sequences.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Switch
                checked={useSlidingWindow}
                onCheckedChange={(checked: boolean) => handleParamChange('useSlidingWindow', checked)}
              />
            </div>
            {useSlidingWindow && (
              <div className="mb-4">
                <Label className="mb-2 block text-sm">Window Size</Label>
                <Input
                  type="number"
                  value={windowSize}
                  onChange={(e) => handleParamChange('windowSize', Number(e.target.value))}
                  className="bg-slate-700 border-slate-600"
                />
              </div>
            )}
          </div>
        );
        
      case 'ffn':
        console.log('Rendering ffn optimizations');
        
        // Ensure proper boolean and numeric values
        const useMoE = !!nodeData.params.useMoE;
        const numExperts = typeof nodeData.params.numExperts === 'number' 
          ? nodeData.params.numExperts 
          : Number(nodeData.params.numExperts || 8);
        const topK = typeof nodeData.params.topK === 'number' 
          ? nodeData.params.topK 
          : Number(nodeData.params.topK || 2);
        
        console.log('useMoE:', useMoE);
        console.log('numExperts:', numExperts);
        console.log('topK:', topK);
        
        return (
          <div className="mt-6 border-t border-slate-700 pt-4">
            <h4 className="text-md font-semibold mb-4">Optimizations</h4>
            <div className="mb-4 flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="text-sm">Use MoE</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Enables Mixture of Experts for this FFN layer.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Switch
                checked={useMoE}
                onCheckedChange={(checked: boolean) => handleParamChange('useMoE', checked)}
              />
            </div>
            {useMoE && (
              <>
                <div className="mb-4">
                  <Label className="mb-2 block text-sm">Number of Experts</Label>
                  <Slider
                    value={[numExperts]}
                    min={4}
                    max={32}
                    step={4}
                    onValueChange={(values) => {
                      handleParamChange('numExperts', values[0]);
                    }}
                    className="my-2"
                  />
                  <div className="text-xs text-right">{numExperts}</div>
                </div>
                <div className="mb-4">
                  <Label className="mb-2 block text-sm">Top-K Experts</Label>
                  <Select
                    value={String(topK)}
                    onValueChange={(val) => handleParamChange('topK', Number(val))}
                  >
                    <SelectTrigger className="bg-slate-700 border-slate-600">
                      <SelectValue placeholder="Select top-k" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 (Switch Transformer)</SelectItem>
                      <SelectItem value="2">2 (Standard MoE)</SelectItem>
                      <SelectItem value="4">4 (High Capacity)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </>
            )}
          </div>
        );
        
      case 'layerNorm':
        console.log('Rendering layerNorm optimizations');
        
        // Ensure proper boolean and numeric values
        const elementwiseAffine = !!nodeData.params.elementwiseAffine;
        const bias = !!nodeData.params.bias;
        const eps = typeof nodeData.params.eps === 'number' 
          ? nodeData.params.eps 
          : Number(nodeData.params.eps || 1e-5);
        
        console.log('elementwiseAffine:', elementwiseAffine);
        console.log('bias:', bias);
        console.log('eps:', eps);
        
        return (
          <div className="mt-6 border-t border-slate-700 pt-4">
            <h4 className="text-md font-semibold mb-4">Advanced Options</h4>
            <div className="mb-4 flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="text-sm">Elementwise Affine</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">When true, applies learnable per-element scale and bias.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Switch
                checked={elementwiseAffine}
                onCheckedChange={(checked: boolean) => handleParamChange('elementwiseAffine', checked)}
              />
            </div>
            <div className="mb-4 flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="text-sm">Use Bias</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">When true, adds a learnable bias term to the normalization.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Switch
                checked={bias}
                onCheckedChange={(checked: boolean) => handleParamChange('bias', checked)}
              />
            </div>
            <div className="mb-4">
              <Label className="mb-2 block text-sm">Epsilon</Label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Input
                      type="number"
                      value={eps}
                      onChange={(e) => handleParamChange('eps', Number(e.target.value))}
                      className="bg-slate-700 border-slate-600"
                      step="0.000001"
                      min="0.000001"
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Small constant added to the denominator for numerical stability.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-4">{node.data.label} Properties</h3>
      
      {/* Render all parameters */}
      {Object.entries(node.data.params).map(([paramName, value]) => 
        renderParamField(paramName, value)
      )}
      
      {/* Render node-specific optimizations */}
      {renderOptimizations()}
    </div>
  );
}

export function FlowEditor({ onGenerateCode, optimizationSettings }: FlowEditorProps) {
  console.log('FlowEditor rendering');
  
  // Initialize nodes and edges from localStorage if available
  const getInitialNodes = (): Node<LLMNodeData>[] => {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') return [];
      
      const savedNodes = localStorage.getItem('llm-graph-nodes');
      return savedNodes ? JSON.parse(savedNodes) : [];
    } catch (error) {
      console.error('Error loading nodes from localStorage:', error);
      return [];
    }
  };
  
  const getInitialEdges = () => {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') return [];
      
      const savedEdges = localStorage.getItem('llm-graph-edges');
      return savedEdges ? JSON.parse(savedEdges) : [];
    } catch (error) {
      console.error('Error loading edges from localStorage:', error);
      return [];
    }
  };
  
  const [nodes, setNodes, onNodesChange] = useNodesState(getInitialNodes());
  const [edges, setEdges, onEdgesChange] = useEdgesState(getInitialEdges());
  const [selectedNode, setSelectedNode] = useState<Node<LLMNodeData> | null>(null);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const renderCount = useRef(0);
  // Add state for active tab in NodePanel
  const [activeNodePanelTab, setActiveNodePanelTab] = useState<string>(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') return 'components';
    
    const savedTab = localStorage.getItem('llm-graph-active-tab');
    return savedTab || 'components';
  });
  
  // Create a ref to store the component state
  const componentsRef = useRef<{
    [key: string]: {
      visible: boolean;
      data: Record<string, unknown>;
    }
  }>({});
  
  // Increment render count on each render
  renderCount.current += 1;
  console.log(`Render count: ${renderCount.current}`);

  // Save nodes and edges to localStorage whenever they change
  useEffect(() => {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') return;
      
      localStorage.setItem('llm-graph-nodes', JSON.stringify(nodes));
      localStorage.setItem('llm-graph-edges', JSON.stringify(edges));
      console.log('Saved graph state to localStorage');
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }, [nodes, edges]);

  // Add effect to track tab changes
  useEffect(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') return;
    
    console.log('Tab changed to:', activeNodePanelTab);
    if (selectedNode) {
      console.log('Selected node during tab change:', selectedNode.id, selectedNode.data);
    }
    
    // Store component visibility state when switching tabs
    if (componentsRef.current) {
      // When tab changes, we need to preserve whatever was visible in the previous tab
      if (activeNodePanelTab === 'components') {
        componentsRef.current.templates = { visible: false, data: componentsRef.current.templates?.data || {} };
        componentsRef.current.components = { visible: true, data: componentsRef.current.components?.data || {} };
      } else if (activeNodePanelTab === 'templates') {
        componentsRef.current.components = { visible: false, data: componentsRef.current.components?.data || {} };
        componentsRef.current.templates = { visible: true, data: componentsRef.current.templates?.data || {} };
      }
    }
    
    // Save active tab to localStorage
    localStorage.setItem('llm-graph-active-tab', activeNodePanelTab);
  }, [activeNodePanelTab, selectedNode]);
  
  // Load active tab from localStorage on initial render
  useEffect(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') return;
    
    const savedTab = localStorage.getItem('llm-graph-active-tab');
    if (savedTab) {
      setActiveNodePanelTab(savedTab);
    }
  }, []);

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({ 
        ...params, 
        type: 'smoothstep',
        animated: true,
      }, eds));
    },
    [setEdges]
  );

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node as Node<LLMNodeData>);
  }, []);

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // Function to validate connections and provide visual feedback
  const validateConnections = useCallback(() => {
    console.log('validateConnections called');
    
    // Create a map of node IDs to their connection status
    const connectionStatus = new Map<string, { 
      hasInputs: boolean, 
      hasOutputs: boolean,
      isValid: boolean,
      connectionErrors: string[]
    }>();
    
    // Initialize all nodes as not connected
    nodes.forEach(node => {
      connectionStatus.set(node.id, { 
        hasInputs: false, 
        hasOutputs: false,
        isValid: false,
        connectionErrors: []
      });
    });
    
    // Define valid connection rules
    const validConnections: Record<string, string[]> = {
      'embedding': ['positionalEncoding', 'layerNorm', 'qkvAttention', 'ffn', 'output'],
      'positionalEncoding': ['layerNorm', 'qkvAttention', 'ffn', 'output'],
      'layerNorm': ['qkvAttention', 'ffn', 'output', 'layerNorm'],
      'qkvAttention': ['layerNorm', 'ffn', 'output', 'qkvAttention'],
      'ffn': ['layerNorm', 'qkvAttention', 'output', 'ffn'],
      'output': [] // Output nodes shouldn't connect to anything
    };
    
    // Check connections
    edges.forEach(edge => {
      // Get source and target nodes
      const sourceNode = nodes.find(node => node.id === edge.source);
      const targetNode = nodes.find(node => node.id === edge.target);
      
      if (!sourceNode || !targetNode) return;
      
      // Check if this connection is valid based on node types
      const sourceType = sourceNode.data.type;
      const targetType = targetNode.data.type;
      
      const isValidConnection = validConnections[sourceType]?.includes(targetType);
      
      // Mark source node as having outputs
      const sourceStatus = connectionStatus.get(edge.source);
      if (sourceStatus) {
        const updatedStatus = { 
          ...sourceStatus, 
          hasOutputs: true 
        };
        
        // Add error if connection is invalid
        if (!isValidConnection) {
          updatedStatus.connectionErrors.push(
            `Invalid connection: ${sourceNode.data.label} cannot connect to ${targetNode.data.label}`
          );
        }
        
        connectionStatus.set(edge.source, updatedStatus);
      }
      
      // Mark target node as having inputs
      const targetStatus = connectionStatus.get(edge.target);
      if (targetStatus) {
        const updatedStatus = { 
          ...targetStatus, 
          hasInputs: true 
        };
        
        // Add error if connection is invalid (only add to target if not already added to source)
        if (!isValidConnection && !sourceStatus?.connectionErrors.some(err => 
          err.includes(`Invalid connection: ${sourceNode.data.label} cannot connect to ${targetNode.data.label}`)
        )) {
          updatedStatus.connectionErrors.push(
            `Invalid connection: ${sourceNode.data.label} cannot connect to ${targetNode.data.label}`
          );
        }
        
        connectionStatus.set(edge.target, updatedStatus);
      }
    });
    
    // Validate each node based on its type
    nodes.forEach(node => {
      const status = connectionStatus.get(node.id);
      if (!status) return;
      
      // Different validation rules based on node type
      switch (node.data.type) {
        case 'embedding':
          // Embedding can be a starting node (no inputs required)
          connectionStatus.set(node.id, { 
            ...status, 
            isValid: status.hasOutputs && status.connectionErrors.length === 0
          });
          break;
          
        case 'output':
          // Output should have inputs but doesn't need outputs
          connectionStatus.set(node.id, { 
            ...status, 
            isValid: status.hasInputs && status.connectionErrors.length === 0
          });
          break;
          
        default:
          // All other nodes should have both inputs and outputs
          connectionStatus.set(node.id, { 
            ...status, 
            isValid: status.hasInputs && status.hasOutputs && status.connectionErrors.length === 0
          });
      }
    });
    
    console.log('Connection validation results:', Object.fromEntries(connectionStatus));
    
    // Update node styles based on validation
    console.log('About to update node styles');
    const updatedNodes = nodes.map(node => {
      const status = connectionStatus.get(node.id);
      
      // Add a visual indicator of connection status
      return {
        ...node,
        style: {
          ...node.style,
          // Add a subtle border color based on validation status
          borderColor: status?.isValid ? '#10b981' : 
                      (status?.connectionErrors && status.connectionErrors.length > 0) ? '#ef4444' : 
                      (status?.hasInputs || status?.hasOutputs) ? '#f59e0b' : '#ef4444',
          borderWidth: 2,
          // Add a subtle background color for invalid nodes
          backgroundColor: status?.isValid ? undefined : 
                          (status?.connectionErrors && status.connectionErrors.length > 0) ? 'rgba(239, 68, 68, 0.1)' : 
                          'rgba(239, 68, 68, 0.05)'
        },
        // Store connection errors in the node data for tooltips
        data: {
          ...node.data,
          connectionErrors: status?.connectionErrors || []
        }
      };
    });
    
    console.log('Node styles updated, checking if nodes changed');
    // Only update if the nodes have actually changed
    const nodesChanged = JSON.stringify(updatedNodes) !== JSON.stringify(nodes);
    if (nodesChanged) {
      console.log('Nodes changed, updating state');
      setNodes(updatedNodes);
    } else {
      console.log('Nodes unchanged, skipping update');
    }
    
    // Return overall validity
    return Array.from(connectionStatus.values()).every(status => status.isValid);
  }, [nodes, edges, setNodes]);

  // Call validation when nodes or edges change
  useEffect(() => {
    console.log('useEffect for validation triggered');
    if (nodes.length > 0) {
      validateConnections();
    }
  }, [nodes.length, edges.length, validateConnections]);

  const handleGenerateCode = useCallback(() => {
    // Validate connections before generating code
    const isValid = validateConnections();
    
    if (!isValid) {
      // Show a warning if the graph is not properly connected
      alert('Warning: Some nodes are not properly connected. This may result in invalid code.');
    }
    
    const code = generatePythonCode(nodes, edges, optimizationSettings);
    onGenerateCode(code);
  }, [nodes, edges, onGenerateCode, optimizationSettings, validateConnections]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) {
        console.error('React Flow wrapper or instance not available');
        return;
      }

      // Get the bounds of the ReactFlow component
      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      
      // Try to get data from both formats
      let dataTransferText = event.dataTransfer.getData('application/reactflow');
      
      if (!dataTransferText) {
        // Fallback to plain text
        dataTransferText = event.dataTransfer.getData('text/plain');
      }
      
      if (!dataTransferText) {
        console.error('No data found in dataTransfer');
        return;
      }

      try {
        const nodeData = JSON.parse(dataTransferText) as LLMNodeData;
        
        // Calculate position of the node
        const position = reactFlowInstance.project({
          x: event.clientX - reactFlowBounds.left,
          y: event.clientY - reactFlowBounds.top,
        });
        
        // Create new node
        const newNode: Node<LLMNodeData> = {
          id: `${nodeData.type}-${Date.now()}`,
          type: nodeData.type,
          position,
          data: nodeData,
        };
        
        setNodes((nds) => nds.concat(newNode));
      } catch (error) {
        console.error('Error processing drop event:', error);
      }
    },
    [reactFlowInstance, setNodes]
  );

  // Add a useEffect to keep selectedNode in sync with nodes array
  useEffect(() => {
    // If there's a selected node, find its latest version in the nodes array
    if (selectedNode) {
      console.log('useEffect for selectedNode sync triggered');
      console.log('Current selectedNode:', selectedNode.id, selectedNode.data);
      
      const updatedNode = nodes.find(node => node.id === selectedNode.id);
      
      if (updatedNode) {
        console.log('Found updatedNode in nodes array:', updatedNode.id, updatedNode.data);
        
        // If the node exists and has changed, update the selectedNode reference
        if (JSON.stringify(updatedNode) !== JSON.stringify(selectedNode)) {
          console.log('Node has changed, updating selectedNode');
          setSelectedNode(updatedNode as Node<LLMNodeData>);
        } else {
          console.log('Node has not changed, skipping update');
        }
      } else {
        console.log('Node not found in nodes array');
      }
    }
  }, [nodes, selectedNode]);

  // Modify the handleNodeUpdate function to ensure it updates both the nodes array and selectedNode
  const handleNodeUpdate = useCallback((id: string, data: LLMNodeData) => {
    console.log('handleNodeUpdate called with:', id, data);
    
    setNodes((nds) => {
      console.log('Updating nodes array with new data');
      const updatedNodes = nds.map((node) => {
        if (node.id === id) {
          // Create the updated node
          const updatedNode = { ...node, data };
          console.log('Created updated node:', updatedNode.id, updatedNode.data);
          
          // If this is the currently selected node, update the selectedNode state too
          if (selectedNode && selectedNode.id === id) {
            console.log('This is the selected node, updating selectedNode state');
            // We need to use setTimeout to avoid React batch updates overriding this
            setTimeout(() => {
              console.log('Inside setTimeout, updating selectedNode');
              setSelectedNode(updatedNode as Node<LLMNodeData>);
            }, 0);
          }
          
          return updatedNode;
        }
        return node;
      });
      
      return updatedNodes;
    });
  }, [setNodes, selectedNode]);

  // Modify the ReactFlow component to use a modified onNodesChange
  return (
    <div className="w-full h-full flex">
      <div className="w-64 bg-slate-800 border-r border-slate-700 p-4 overflow-y-auto">
        <NodePanel 
          setNodes={setNodes} 
          activeTab={activeNodePanelTab}
          setActiveTab={setActiveNodePanelTab}
        />
      </div>
      
      <div className="flex-1 h-full" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={(changes) => {
            console.log('onNodesChange called with:', changes);
            
            // If we have a selected node, check if it's being modified
            if (selectedNode) {
              const isSelectedNodeChanged = changes.some(change => 
                'id' in change && change.id === selectedNode.id
              );
              if (isSelectedNodeChanged) {
                console.log('Selected node is being modified by onNodesChange');
              }
            }
            
            // Apply the changes
            onNodesChange(changes);
            
            // Update selectedNode if it was changed
            if (selectedNode) {
              setTimeout(() => {
                const updatedNode = nodes.find(node => node.id === selectedNode.id);
                if (updatedNode && JSON.stringify(updatedNode) !== JSON.stringify(selectedNode)) {
                  console.log('Updating selectedNode after onNodesChange:', updatedNode);
                  setSelectedNode(updatedNode as Node<LLMNodeData>);
                }
              }, 0);
            }
          }}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={handleNodeClick}
          onPaneClick={handlePaneClick}
          nodeTypes={nodeTypes}
          connectionLineType={ConnectionLineType.SmoothStep}
          onInit={setReactFlowInstance}
          onDragOver={onDragOver}
          onDrop={onDrop}
          fitView
        >
          <Background color="#475569" gap={16} />
          <Controls />
          <Panel position="bottom-right" className="bg-slate-800 p-2 rounded-md shadow-md">
            <Button 
              onClick={handleGenerateCode}
              className="bg-blue-600 hover:bg-blue-700"
            >
              Generate PyTorch Code
            </Button>
          </Panel>
        </ReactFlow>
      </div>
      
      {selectedNode && (
        <div className="w-80 bg-slate-800 border-l border-slate-700 overflow-y-auto">
          <NodeProperties 
            node={selectedNode} 
            onChange={handleNodeUpdate} 
          />
        </div>
      )}
    </div>
  );
} 