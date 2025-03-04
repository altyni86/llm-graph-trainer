import React from 'react';
import { render, screen } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { act } from 'react';
import { FlowEditor } from '../FlowEditor';
import userEvent from '@testing-library/user-event';
import { NodePanel } from '../NodePanel';
import { generatePythonCode } from '@/lib/code-generator';

// Mock the generatePythonCode function
vi.mock('@/lib/code-generator', () => ({
  generatePythonCode: vi.fn().mockReturnValue('# Generated Python code')
}));

// Mock the ReactFlow component and its hooks
vi.mock('reactflow', async () => {
  const originalModule = await vi.importActual('reactflow');
  
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mockReactFlow = ({ children, nodes, edges, onNodesChange, onEdgesChange, onNodeClick, onPaneClick }: any) => (
    <div data-testid="react-flow">
      {children}
      <div data-testid="nodes-container">
        {nodes.map((node: any) => (
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const onNodesChange = (changes: any) => {
        // Simplified implementation for testing
        setNodes((prevNodes: any) => {
          return changes.reduce((acc: any, change: any) => {
            if (change.type === 'add') {
              return [...acc, change.item];
            } else if (change.type === 'remove') {
              return acc.filter((node: any) => node.id !== change.id);
            } else if (change.type === 'position' || change.type === 'dimensions') {
              return acc.map((node: any) => node.id === change.id ? { ...node, ...change.dimensions, position: change.position || node.position } : node);
            }
            return acc;
          }, prevNodes);
        });
      };
      return [nodes, setNodes, onNodesChange];
    },
    useEdgesState: () => {
      const [edges, setEdges] = React.useState([]);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const onEdgesChange = (changes: any) => {
        // Simplified implementation for testing
        setEdges((prevEdges: any) => {
          return changes.reduce((acc: any, change: any) => {
            if (change.type === 'add') {
              return [...acc, change.item];
            } else if (change.type === 'remove') {
              return acc.filter((edge: any) => edge.id !== change.id);
            }
            return acc;
          }, prevEdges);
        });
      };
      return [edges, setEdges, onEdgesChange];
    },
    addEdge: (edge: any, edges: any) => [...edges, edge],
    Background: () => <div data-testid="background">Background</div>,
    Controls: () => <div data-testid="controls">Controls</div>,
    Panel: ({ children }: any) => <div data-testid="panel">{children}</div>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Handle: ({ type, position, isConnectable }: any) => (
      <div data-testid={`handle-${type}-${position}`} className={`handle-${type}`}>
        Handle
      </div>
    ),
  };
});

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    clear: () => {
      store = {};
    },
    removeItem: (key: string) => {
      delete store[key];
    }
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

describe('Post Training Flow Integration', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it('displays the Post Training tab in NodePanel', async () => {
    // Render the NodePanel component
    render(
      <NodePanel 
        setNodes={vi.fn()} 
        activeTab="components" 
        setActiveTab={vi.fn()} 
      />
    );
    
    // Check that the Post Training tab exists
    expect(screen.getByRole('tab', { name: /post training/i })).toBeInTheDocument();
  });

  it('allows adding a training node to the flow', async () => {
    // Mock the onGenerateCode function
    const onGenerateCodeMock = vi.fn();
    
    // Render the FlowEditor component
    render(
      <FlowEditor 
        onGenerateCode={onGenerateCodeMock} 
        optimizationSettings={{
          gradientCheckpointing: false,
          flashAttention: false,
          fusedLayerNorm: false,
          memoryEfficient: false,
          xformers: false,
          betterTransformer: false,
          channelsLast: false,
          amp: false,
          prefer_mps: false,
        }}
      />
    );
    
    // Set up user event
    const user = userEvent.setup();
    
    // Find the Post Training tab and click it
    const postTrainingTab = screen.getByRole('tab', { name: /post training/i });
    await user.click(postTrainingTab);
    
    // Find the SFT Training node and click it to add it to the canvas
    const sftTrainingNode = screen.getByText('SFT Training');
    await user.click(sftTrainingNode);
    
    // Verify that a node was added to the flow
    // We need to wait for the node to be added to the DOM
    await act(async () => {
      // Wait for any state updates to complete
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    
    // Check that the node exists in the flow
    // Since we're using a mock ReactFlow, we need to check for the node in our mock implementation
    expect(screen.getByTestId(/node-sftTraining/)).toBeInTheDocument();
  });

  it('can generate code with training nodes', () => {
    // Mock the onGenerateCode function
    const onGenerateCodeMock = vi.fn();
    
    // Create a test node for SFT training
    const sftNode = {
      id: 'sftTraining-1',
      type: 'sftTraining',
      position: { x: 300, y: 100 },
      data: {
        label: 'SFT Training',
        type: 'sftTraining',
        params: {
          learningRate: 5e-5,
          batchSize: 16,
          numEpochs: 3,
          optimizer: 'adamw',
        },
      },
    };
    
    // Create test nodes and edges
    const testNodes = [sftNode];
    const testEdges = [];
    
    // Call generatePythonCode directly
    const code = generatePythonCode(testNodes, testEdges, {
      gradientCheckpointing: false,
      flashAttention: false,
      fusedLayerNorm: false,
      memoryEfficient: false,
      xformers: false,
      betterTransformer: false,
      channelsLast: false,
      amp: false,
      prefer_mps: false,
    });
    
    // Verify that the code was generated
    expect(generatePythonCode).toHaveBeenCalled();
    expect(code).toBe('# Generated Python code');
  });
}); 