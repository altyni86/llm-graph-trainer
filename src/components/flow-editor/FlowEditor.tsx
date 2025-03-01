"use client";

import { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
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
  
  const handleParamChange = (paramName: string, value: unknown) => {
    console.log(`Parameter ${paramName} changing to:`, value);
    
    const updatedData = {
      ...node.data,
      params: {
        ...node.data.params,
        [paramName]: value
      }
    };
    
    console.log('Updated node data:', updatedData);
    onChange(node.id, updatedData);
  };
  
  const renderParamField = (paramName: string, value: any) => {
    // Get parameter type
    const type = typeof value;
    
    switch (type) {
      case 'number':
        return (
          <div className="mb-4" key={paramName}>
            <Label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Input
              type="number"
              value={value}
              onChange={(e) => handleParamChange(paramName, Number(e.target.value))}
              className="bg-slate-700 border-slate-600"
            />
          </div>
        );
        
      case 'boolean':
        return (
          <div className="mb-4 flex items-center justify-between" key={paramName}>
            <Label className="text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Label>
            <Switch
              checked={value}
              onCheckedChange={(checked) => handleParamChange(paramName, checked)}
            />
          </div>
        );
        
      case 'string':
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
          
          return (
            <div className="mb-4" key={paramName}>
              <Label className="mb-2 block text-sm">
                {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </Label>
              <Select
                value={value}
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
              value={value}
              onChange={(e) => handleParamChange(paramName, e.target.value)}
              className="bg-slate-700 border-slate-600"
            />
          </div>
        );
        
      default:
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
    switch (node.data.type) {
      case 'qkvAttention':
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
                checked={!!node.data.params.useFlashAttention}
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
                checked={!!node.data.params.useSlidingWindow}
                onCheckedChange={(checked: boolean) => handleParamChange('useSlidingWindow', checked)}
              />
            </div>
            {node.data.params.useSlidingWindow && (
              <div className="mb-4">
                <Label className="mb-2 block text-sm">Window Size</Label>
                <Input
                  type="number"
                  value={node.data.params.windowSize || 512}
                  onChange={(e) => handleParamChange('windowSize', Number(e.target.value))}
                  className="bg-slate-700 border-slate-600"
                />
              </div>
            )}
          </div>
        );
        
      case 'ffn':
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
                checked={!!node.data.params.useMoE}
                onCheckedChange={(checked: boolean) => handleParamChange('useMoE', checked)}
              />
            </div>
            {node.data.params.useMoE && (
              <>
                <div className="mb-4">
                  <Label className="mb-2 block text-sm">Number of Experts</Label>
                  {console.log('Current numExperts value:', node.data.params.numExperts)}
                  <Slider
                    value={[node.data.params.numExperts as number || 8]}
                    min={4}
                    max={32}
                    step={4}
                    onValueChange={(values) => {
                      console.log('Slider value changed to:', values[0]);
                      handleParamChange('numExperts', values[0]);
                    }}
                    className="my-2"
                  />
                  <div className="text-xs text-right">{node.data.params.numExperts || 8}</div>
                </div>
                <div className="mb-4">
                  <Label className="mb-2 block text-sm">Top-K Experts</Label>
                  <Select
                    value={String(node.data.params.topK || 2)}
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
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node<LLMNodeData> | null>(null);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const renderCount = useRef(0);
  
  // Increment render count on each render
  renderCount.current += 1;
  console.log(`Render count: ${renderCount.current}`);

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
      isValid: boolean 
    }>();
    
    // Initialize all nodes as not connected
    nodes.forEach(node => {
      connectionStatus.set(node.id, { 
        hasInputs: false, 
        hasOutputs: false,
        isValid: false 
      });
    });
    
    // Check connections
    edges.forEach(edge => {
      // Mark source node as having outputs
      const sourceStatus = connectionStatus.get(edge.source);
      if (sourceStatus) {
        connectionStatus.set(edge.source, { 
          ...sourceStatus, 
          hasOutputs: true 
        });
      }
      
      // Mark target node as having inputs
      const targetStatus = connectionStatus.get(edge.target);
      if (targetStatus) {
        connectionStatus.set(edge.target, { 
          ...targetStatus, 
          hasInputs: true 
        });
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
            isValid: status.hasOutputs 
          });
          break;
          
        case 'output':
          // Output should have inputs but doesn't need outputs
          connectionStatus.set(node.id, { 
            ...status, 
            isValid: status.hasInputs 
          });
          break;
          
        default:
          // All other nodes should have both inputs and outputs
          connectionStatus.set(node.id, { 
            ...status, 
            isValid: status.hasInputs && status.hasOutputs 
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
          borderColor: status?.isValid ? '#10b981' : status?.hasInputs || status?.hasOutputs ? '#f59e0b' : '#ef4444',
          borderWidth: 2,
          // Add a subtle background color for invalid nodes
          backgroundColor: status?.isValid ? undefined : 'rgba(239, 68, 68, 0.05)'
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
  }, [nodes, edges]);

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
      const updatedNode = nodes.find(node => node.id === selectedNode.id);
      
      // If the node exists and has changed, update the selectedNode reference
      if (updatedNode && JSON.stringify(updatedNode) !== JSON.stringify(selectedNode)) {
        console.log('Updating selectedNode to match nodes array:', updatedNode);
        setSelectedNode(updatedNode as Node<LLMNodeData>);
      }
    }
  }, [nodes, selectedNode]);

  // Modify the handleNodeUpdate function to ensure it updates both the nodes array and selectedNode
  const handleNodeUpdate = useCallback((id: string, data: LLMNodeData) => {
    console.log('handleNodeUpdate called with:', id, data);
    
    setNodes((nds) => {
      const updatedNodes = nds.map((node) => {
        if (node.id === id) {
          // Create the updated node
          const updatedNode = { ...node, data };
          
          // If this is the currently selected node, update the selectedNode state too
          if (selectedNode && selectedNode.id === id) {
            // We need to use setTimeout to avoid React batch updates overriding this
            setTimeout(() => {
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
        <NodePanel setNodes={setNodes} />
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
                change.id === selectedNode.id
              );
              if (isSelectedNodeChanged) {
                console.log('Selected node is being modified by onNodesChange');
              }
            }
            
            // Apply the changes
            onNodesChange(changes);
            
            // Update selectedNode if it was changed
            if (selectedNode) {
              const updatedNode = changes.find(change => change.id === selectedNode.id);
              if (updatedNode) {
                console.log('Need to update selectedNode after change:', updatedNode);
              }
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