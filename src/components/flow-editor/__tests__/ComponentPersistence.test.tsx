import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { FlowEditor } from '../FlowEditor';
import { act } from 'react-dom/test-utils';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

// Mock the ReactFlow component and its hooks
vi.mock('reactflow', async () => {
  const originalModule = await vi.importActual('reactflow');
  
  const mockReactFlow = ({ children, nodes, edges, onNodesChange, onEdgesChange, onNodeClick, onPaneClick }) => (
    <div data-testid="react-flow">
      {children}
      <div data-testid="nodes-container">
        {nodes.map(node => (
          <div 
            key={node.id} 
            data-testid={`node-${node.id}`}
            onClick={(e) => onNodeClick(e, node)}
          >
            {node.data.label}
          </div>
        ))}
      </div>
      <button data-testid="pane" onClick={onPaneClick}>Pane</button>
    </div>
  );

  return {
    ...originalModule,
    default: mockReactFlow,
    ReactFlow: mockReactFlow,
    useNodesState: () => {
      const [nodes, setNodes] = React.useState([]);
      const onNodesChange = (changes) => {
        // Simplified implementation for testing
        setNodes(prevNodes => {
          return changes.reduce((acc, change) => {
            if (change.type === 'add') {
              return [...acc, change.item];
            } else if (change.type === 'remove') {
              return acc.filter(node => node.id !== change.id);
            } else if (change.type === 'position' || change.type === 'dimensions') {
              return acc.map(node => node.id === change.id ? { ...node, ...change.dimensions, position: change.position || node.position } : node);
            }
            return acc;
          }, prevNodes);
        });
      };
      return [nodes, setNodes, onNodesChange];
    },
    useEdgesState: () => {
      const [edges, setEdges] = React.useState([]);
      const onEdgesChange = vi.fn();
      return [edges, setEdges, onEdgesChange];
    },
    Background: () => <div data-testid="background"></div>,
    Controls: () => <div data-testid="controls"></div>,
    Panel: ({ children, position }) => (
      <div data-testid={`panel-${position}`}>{children}</div>
    ),
    ConnectionLineType: { SmoothStep: 'smoothstep' },
    addEdge: vi.fn((params, edges) => [...edges, { ...params, id: `e-${Date.now()}` }]),
  };
});

// Mock the node types
vi.mock('../nodes/EmbeddingNode', () => ({
  EmbeddingNode: () => <div data-testid="embedding-node">Embedding Node</div>
}));

vi.mock('../nodes/PositionalEncodingNode', () => ({
  PositionalEncodingNode: () => <div data-testid="positional-encoding-node">Positional Encoding Node</div>
}));

vi.mock('../nodes/QKVAttentionNode', () => ({
  QKVAttentionNode: () => <div data-testid="qkv-attention-node">QKV Attention Node</div>
}));

vi.mock('../nodes/FFNNode', () => ({
  FFNNode: () => <div data-testid="ffn-node">FFN Node</div>
}));

vi.mock('../nodes/OutputNode', () => ({
  OutputNode: () => <div data-testid="output-node">Output Node</div>
}));

vi.mock('../nodes/LayerNormNode', () => ({
  LayerNormNode: () => <div data-testid="layer-norm-node">Layer Norm Node</div>
}));

// Mock the NodePanel component
vi.mock('../NodePanel', () => ({
  NodePanel: ({ setNodes, activeTab, setActiveTab }) => (
    <div data-testid="node-panel">
      <div data-testid="active-tab">{activeTab}</div>
      <button 
        data-testid="switch-to-templates"
        onClick={() => setActiveTab('templates')}
      >
        Switch to Templates
      </button>
      <button 
        data-testid="switch-to-components"
        onClick={() => setActiveTab('components')}
      >
        Switch to Components
      </button>
      <button 
        data-testid="add-ffn-node"
        onClick={() => {
          setNodes(nodes => [...nodes, {
            id: `ffn-${Date.now()}`,
            type: 'ffn',
            position: { x: 300, y: 100 },
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
          }]);
        }}
      >
        Add FFN Node
      </button>
      <button 
        data-testid="add-layernorm-node"
        onClick={() => {
          setNodes(nodes => [...nodes, {
            id: `layerNorm-${Date.now()}`,
            type: 'layerNorm',
            position: { x: 300, y: 200 },
            data: {
              type: 'layerNorm',
              label: 'Layer Normalization',
              params: {
                normalizedShape: 512,
                eps: 1e-5,
                elementwiseAffine: true,
                bias: true
              }
            }
          }]);
        }}
      >
        Add LayerNorm Node
      </button>
    </div>
  )
}));

// Mock the code generator
vi.mock('@/lib/code-generator', () => ({
  generatePythonCode: vi.fn(() => 'mocked python code')
}));

describe('Component Persistence Test', () => {
  const onGenerateCodeMock = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
  });
  
  it('components should persist when switching tabs', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add a LayerNorm node
    fireEvent.click(screen.getByTestId('add-layernorm-node'));
    
    // Wait for the node to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
    
    // Click on the node to select it
    const nodeElement = screen.getByTestId(/node-layerNorm-/);
    fireEvent.click(nodeElement);
    
    // Check if the properties panel is shown
    await waitFor(() => {
      expect(screen.getByText(/Layer Normalization Properties/i)).toBeInTheDocument();
    });
    
    // Check if the elementwiseAffine switch is present
    expect(screen.getAllByText(/Elementwise Affine/i)[0]).toBeInTheDocument();
    
    // Switch to templates tab
    fireEvent.click(screen.getByTestId('switch-to-templates'));
    
    // Verify that the active tab has changed
    expect(screen.getByTestId('active-tab').textContent).toBe('templates');
    
    // Check that the node properties panel is still visible
    expect(screen.getByText(/Layer Normalization Properties/i)).toBeInTheDocument();
    
    // Check that the elementwiseAffine switch is still present
    expect(screen.getAllByText(/Elementwise Affine/i)[0]).toBeInTheDocument();
    
    // Switch back to components tab
    fireEvent.click(screen.getByTestId('switch-to-components'));
    
    // Verify that the active tab has changed back
    expect(screen.getByTestId('active-tab').textContent).toBe('components');
    
    // Check that the node properties panel is still visible
    expect(screen.getByText(/Layer Normalization Properties/i)).toBeInTheDocument();
    
    // Check that the elementwiseAffine switch is still present
    expect(screen.getAllByText(/Elementwise Affine/i)[0]).toBeInTheDocument();
  });
  
  it('should maintain node state when switching tabs', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an FFN node
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Wait for the node to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
    
    // Click on the node to select it
    const nodeElement = screen.getByTestId(/node-ffn-/);
    fireEvent.click(nodeElement);
    
    // Wait for the properties panel to show
    await waitFor(() => {
      expect(screen.getByText(/Feed Forward Properties/i)).toBeInTheDocument();
    });
    
    // Find the MoE switch and toggle it
    const moeSwitch = screen.getByText(/Use MoE/i).nextElementSibling;
    fireEvent.click(moeSwitch);
    
    // Check if the numExperts slider appears (indicating the switch worked)
    await waitFor(() => {
      expect(screen.getByText(/Number of Experts/i)).toBeInTheDocument();
    });
    
    // Switch to templates tab
    fireEvent.click(screen.getByTestId('switch-to-templates'));
    
    // Switch back to components tab
    fireEvent.click(screen.getByTestId('switch-to-components'));
    
    // Check that the Number of Experts slider is still present
    expect(screen.getByText(/Number of Experts/i)).toBeInTheDocument();
  });
}); 