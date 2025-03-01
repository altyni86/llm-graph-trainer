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

// Mock the NodePanel component
vi.mock('../NodePanel', () => ({
  NodePanel: ({ setNodes }) => (
    <div data-testid="node-panel">
      <button 
        data-testid="add-embedding-node"
        onClick={() => {
          setNodes(nodes => [...nodes, {
            id: `embedding-${Date.now()}`,
            type: 'embedding',
            position: { x: 100, y: 100 },
            data: {
              type: 'embedding',
              label: 'Embedding',
              params: {
                vocabSize: 50000,
                embeddingDim: 512
              }
            }
          }]);
        }}
      >
        Add Embedding Node
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
    </div>
  )
}));

// Mock the code generator
vi.mock('@/lib/code-generator', () => ({
  generatePythonCode: vi.fn(() => 'mocked python code')
}));

// Mock console.log to track logs
const originalConsoleLog = console.log;
let consoleOutput = [];

describe('FlowEditor Component', () => {
  const onGenerateCodeMock = vi.fn();
  
  beforeEach(() => {
    consoleOutput = [];
    console.log = (...args) => {
      consoleOutput.push(args);
      originalConsoleLog(...args);
    };
  });
  
  afterEach(() => {
    console.log = originalConsoleLog;
    vi.clearAllMocks();
  });
  
  it('renders without crashing', () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    expect(screen.getByTestId('node-panel')).toBeInTheDocument();
  });
  
  it('adds a node when clicked in the node panel', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an embedding node
    fireEvent.click(screen.getByTestId('add-embedding-node'));
    
    // Wait for the node to be added to the DOM
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
  });
  
  it('selects a node when clicked and shows properties panel', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an embedding node
    fireEvent.click(screen.getByTestId('add-embedding-node'));
    
    // Wait for the node to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
    
    // Click on the node to select it
    const nodeElement = screen.getByTestId(/node-embedding-/);
    fireEvent.click(nodeElement);
    
    // Check if the properties panel is shown
    await waitFor(() => {
      // The properties panel should contain the node label
      expect(screen.getByText(/Embedding Properties/i)).toBeInTheDocument();
    });
  });
  
  it('updates node parameters and keeps selectedNode in sync', async () => {
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
    
    // Check console logs to verify that both nodes array and selectedNode were updated
    const nodeUpdateLogs = consoleOutput.filter(log => 
      log[0] === 'handleNodeUpdate called with:' || 
      log[0] === 'Updating selectedNode to match nodes array:'
    );
    
    expect(nodeUpdateLogs.length).toBeGreaterThan(0);
  });
  
  it('deselects a node when clicking on the pane', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an embedding node
    fireEvent.click(screen.getByTestId('add-embedding-node'));
    
    // Wait for the node to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
    
    // Click on the node to select it
    const nodeElement = screen.getByTestId(/node-embedding-/);
    fireEvent.click(nodeElement);
    
    // Check if the properties panel is shown
    await waitFor(() => {
      expect(screen.getByText(/Embedding Properties/i)).toBeInTheDocument();
    });
    
    // Click on the pane to deselect the node
    fireEvent.click(screen.getByTestId('pane'));
    
    // Check if the properties panel is hidden
    await waitFor(() => {
      expect(screen.queryByText(/Embedding Properties/i)).not.toBeInTheDocument();
    });
  });
  
  it('validates connections when nodes or edges change', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add two nodes
    fireEvent.click(screen.getByTestId('add-embedding-node'));
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Wait for the nodes to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(2);
    });
    
    // Check if validateConnections was called
    const validationLogs = consoleOutput.filter(log => 
      log[0] === 'validateConnections called' || 
      log[0] === 'Connection validation results:'
    );
    
    expect(validationLogs.length).toBeGreaterThan(0);
  });
  
  it('generates code when the generate button is clicked', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add a node
    fireEvent.click(screen.getByTestId('add-embedding-node'));
    
    // Wait for the node to be added
    await waitFor(() => {
      const nodesContainer = screen.getByTestId('nodes-container');
      expect(nodesContainer.children.length).toBe(1);
    });
    
    // Click the generate code button
    const generateButton = screen.getByText(/Generate PyTorch Code/i);
    fireEvent.click(generateButton);
    
    // Check if the onGenerateCode callback was called
    expect(onGenerateCodeMock).toHaveBeenCalledWith('mocked python code');
  });
}); 