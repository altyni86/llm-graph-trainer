import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act } from 'react-dom/test-utils';
import { FlowEditor } from '../FlowEditor';

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

// Mock the UI components
vi.mock('@/components/ui/input', () => ({
  Input: ({ value, onChange, type, className }) => (
    <input 
      data-testid={`input-${type}`}
      value={value}
      onChange={onChange}
      className={className}
    />
  )
}));

vi.mock('@/components/ui/label', () => ({
  Label: ({ children, className }) => (
    <label data-testid="label" className={className}>{children}</label>
  )
}));

vi.mock('@/components/ui/switch', () => ({
  Switch: ({ checked, onCheckedChange }) => (
    <input 
      type="checkbox" 
      data-testid="switch"
      checked={checked}
      onChange={(e) => onCheckedChange(e.target.checked)}
    />
  )
}));

vi.mock('@/components/ui/slider', () => ({
  Slider: ({ value, min, max, step, onValueChange }) => (
    <input 
      type="range"
      data-testid="slider"
      value={value[0]}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        console.log('Slider value changed to:', Number(e.target.value));
        onValueChange([Number(e.target.value)]);
      }}
    />
  )
}));

vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }) => (
    <select 
      data-testid="select"
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
    >
      {children}
    </select>
  ),
  SelectTrigger: ({ children, className }) => (
    <div data-testid="select-trigger" className={className}>{children}</div>
  ),
  SelectValue: ({ placeholder }) => (
    <span data-testid="select-value">{placeholder}</span>
  ),
  SelectContent: ({ children }) => (
    <div data-testid="select-content">{children}</div>
  ),
  SelectItem: ({ value, children }) => (
    <option data-testid={`select-item-${value}`} value={value}>{children}</option>
  )
}));

vi.mock('@/components/ui/tooltip', () => ({
  Tooltip: ({ children }) => <div data-testid="tooltip">{children}</div>,
  TooltipContent: ({ children }) => <div data-testid="tooltip-content">{children}</div>,
  TooltipProvider: ({ children }) => <div data-testid="tooltip-provider">{children}</div>,
  TooltipTrigger: ({ asChild, children }) => (
    <div data-testid="tooltip-trigger" data-aschild={asChild}>{children}</div>
  )
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, className }) => (
    <button 
      data-testid="button" 
      onClick={onClick}
      className={className}
    >
      {children}
    </button>
  )
}));

// Mock the code generator
vi.mock('@/lib/code-generator', () => ({
  generatePythonCode: vi.fn(() => 'mocked python code')
}));

// Mock console.log to track logs
const originalConsoleLog = console.log;
let consoleOutput = [];

describe('FlowEditor and NodeProperties Integration', () => {
  const onGenerateCodeMock = vi.fn();
  
  beforeEach(() => {
    consoleOutput = [];
    console.log = (...args) => {
      consoleOutput.push(args);
      originalConsoleLog(...args);
    };
    
    // Reset timers before each test
    vi.useFakeTimers();
  });
  
  afterEach(() => {
    console.log = originalConsoleLog;
    vi.clearAllMocks();
    vi.useRealTimers();
  });
  
  it('updates node parameters from the properties panel and reflects changes in the node', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an FFN node
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Find the node element directly without waiting
    const nodeElement = screen.getByTestId(/node-ffn-/);
    
    // Click on the node to select it
    fireEvent.click(nodeElement);
    
    // Find the properties panel directly
    const propertiesHeading = screen.getByText(/Feed Forward Properties/i);
    expect(propertiesHeading).toBeInTheDocument();
    
    // Find the hiddenDim input and change its value - use a more specific selector
    const hiddenDimLabel = screen.getByText('Hidden Dim');
    const hiddenDimInput = hiddenDimLabel.parentElement?.querySelector('[data-testid="input-number"]');
    expect(hiddenDimInput).toBeInTheDocument();
    fireEvent.change(hiddenDimInput as HTMLElement, { target: { value: '4096' } });
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Check if the parameter change was logged
    const paramChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Parameter hiddenDim changing to:'
    );
    expect(paramChangeLogs.length).toBeGreaterThan(0);
    
    // Check if handleNodeUpdate was called
    const updateLogs = consoleOutput.filter(log => 
      log[0] === 'handleNodeUpdate called with:'
    );
    expect(updateLogs.length).toBeGreaterThan(0);
  }, 10000);
  
  it('updates node parameters when toggling a switch and shows conditional fields', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an FFN node
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Find the node element directly
    const nodeElement = screen.getByTestId(/node-ffn-/);
    
    // Click on the node to select it
    fireEvent.click(nodeElement);
    
    // Find the properties panel directly
    const propertiesHeading = screen.getByText(/Feed Forward Properties/i);
    expect(propertiesHeading).toBeInTheDocument();
    
    // Find the useMoE switch and toggle it
    const moeSwitch = screen.getAllByTestId('switch')[0]; // First switch should be useMoE
    fireEvent.click(moeSwitch);
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Re-render to show conditional fields
    // This is needed because we're using fake timers and the component might not update immediately
    vi.runAllTimers();
    
    // Check if the conditional fields appear
    const numExpertsLabel = screen.getByText(/Number of Experts/i);
    expect(numExpertsLabel).toBeInTheDocument();
    
    const topKLabel = screen.getByText(/Top-K Experts/i);
    expect(topKLabel).toBeInTheDocument();
    
    // Find the numExperts slider and change its value
    const numExpertsSlider = screen.getByTestId('slider');
    fireEvent.change(numExpertsSlider, { target: { value: '16' } });
    
    // Run all timers again
    vi.runAllTimers();
    
    // Check if the slider value change was logged
    const sliderChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Slider value changed to:'
    );
    expect(sliderChangeLogs.length).toBeGreaterThan(0);
  }, 10000);
  
  it('keeps the properties panel in sync with node changes from other sources', async () => {
    // This test simulates the scenario where a node is updated through validation
    // or other means, and we want to ensure the properties panel stays in sync
    
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an FFN node
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Find the node element directly
    const nodeElement = screen.getByTestId(/node-ffn-/);
    
    // Click on the node to select it
    fireEvent.click(nodeElement);
    
    // Find the properties panel directly
    const propertiesHeading = screen.getByText(/Feed Forward Properties/i);
    expect(propertiesHeading).toBeInTheDocument();
    
    // Trigger validation by clicking the generate code button
    // This should update node styles and potentially other properties
    const generateButton = screen.getByText(/Generate PyTorch Code/i);
    fireEvent.click(generateButton);
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Check if validateConnections was called
    const validationLogs = consoleOutput.filter(log => 
      log[0] === 'validateConnections called'
    );
    expect(validationLogs.length).toBeGreaterThan(0);
    
    // The properties panel should still be showing the correct node data
    // We can't easily check the specific values, but we can verify it's still rendered
    expect(screen.getByText(/Feed Forward Properties/i)).toBeInTheDocument();
  }, 10000);
  
  it('handles multiple parameter updates in sequence correctly', async () => {
    render(<FlowEditor onGenerateCode={onGenerateCodeMock} />);
    
    // Add an FFN node
    fireEvent.click(screen.getByTestId('add-ffn-node'));
    
    // Find the node element directly
    const nodeElement = screen.getByTestId(/node-ffn-/);
    
    // Click on the node to select it
    fireEvent.click(nodeElement);
    
    // Find the properties panel directly
    const propertiesHeading = screen.getByText(/Feed Forward Properties/i);
    expect(propertiesHeading).toBeInTheDocument();
    
    // Find the useMoE switch and toggle it
    const moeSwitch = screen.getAllByTestId('switch')[0];
    fireEvent.click(moeSwitch);
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Check if the conditional fields appear
    const numExpertsLabel = screen.getByText(/Number of Experts/i);
    expect(numExpertsLabel).toBeInTheDocument();
    
    // Find the numExperts slider and change its value
    const numExpertsSlider = screen.getByTestId('slider');
    fireEvent.change(numExpertsSlider, { target: { value: '16' } });
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Find the topK select and change its value - use a more specific selector
    const topKLabel = screen.getByText('Top-K Experts');
    const topKSelect = topKLabel.parentElement?.querySelector('[data-testid="select"]');
    expect(topKSelect).toBeInTheDocument();
    fireEvent.change(topKSelect as HTMLElement, { target: { value: '4' } });
    
    // Run all timers at once
    vi.runAllTimers();
    
    // Check if all parameter changes were logged
    const paramChangeLogs = consoleOutput.filter(log => 
      log[0].includes('Parameter') && log[0].includes('changing to:')
    );
    expect(paramChangeLogs.length).toBe(3); // useMoE, numExperts, topK
    
    // Check if handleNodeUpdate was called for each change
    const updateLogs = consoleOutput.filter(log => 
      log[0] === 'handleNodeUpdate called with:'
    );
    expect(updateLogs.length).toBe(3);
  }, 10000);
}); 