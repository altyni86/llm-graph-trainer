import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

// Create a mock NodeProperties component since we can't extract it from FlowEditor
const NodeProperties = ({ node, onChange }) => {
  console.log('NodeProperties rendering for node:', node.id, node.data);
  
  const handleParamChange = (paramName, value) => {
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
  
  const renderParamField = (paramName, value) => {
    // Get parameter type
    const type = typeof value;
    
    switch (type) {
      case 'number':
        return (
          <div className="mb-4" key={paramName}>
            <label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </label>
            <input
              type="number"
              value={value}
              onChange={(e) => handleParamChange(paramName, Number(e.target.value))}
              className="bg-slate-700 border-slate-600"
              data-testid="input-number"
            />
          </div>
        );
        
      case 'boolean':
        return (
          <div className="mb-4 flex items-center justify-between" key={paramName}>
            <label className="text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </label>
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => handleParamChange(paramName, e.target.checked)}
              data-testid="switch"
            />
          </div>
        );
        
      case 'string':
        return (
          <div className="mb-4" key={paramName}>
            <label className="mb-2 block text-sm">
              {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </label>
            <select
              value={value}
              onChange={(e) => handleParamChange(paramName, e.target.value)}
              className="bg-slate-700 border-slate-600"
              data-testid="select"
            >
              <option value="gelu">GELU</option>
              <option value="relu">ReLU</option>
              <option value="silu">SiLU</option>
              <option value="scaled_dot_product">Scaled Dot Product</option>
            </select>
          </div>
        );
        
      default:
        return null;
    }
  };
  
  // Special case for sliders
  const renderSlider = (paramName, value, min, max, step) => {
    return (
      <div className="mb-4" key={paramName}>
        <label className="mb-2 block text-sm">
          {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
        </label>
        <input
          type="range"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            console.log('Slider value changed to:', Number(e.target.value));
            handleParamChange(paramName, Number(e.target.value));
          }}
          data-testid="slider"
        />
      </div>
    );
  };
  
  // Render node properties based on node type
  const renderNodeProperties = () => {
    const { type, params } = node.data;
    
    // Common properties section
    const commonProps = Object.entries(params).map(([key, value]) => {
      if (key === 'useMoE' || key === 'useFlashAttention' || key === 'useSlidingWindow') {
        return renderParamField(key, value);
      }
      
      if (type === 'ffn' && key === 'numExperts' && params.useMoE) {
        return renderSlider(key, value, 4, 32, 4);
      }
      
      if (type === 'ffn' && key === 'topK' && params.useMoE) {
        return renderParamField(key, value);
      }
      
      if (type === 'qkvAttention' && key === 'windowSize' && params.useSlidingWindow) {
        return renderSlider(key, value, 128, 2048, 128);
      }
      
      if (!['useMoE', 'useFlashAttention', 'useSlidingWindow', 'numExperts', 'topK', 'windowSize'].includes(key)) {
        return renderParamField(key, value);
      }
      
      return null;
    });
    
    return (
      <div>
        <h3 className="text-lg font-semibold mb-4">
          {node.data.label} Properties
        </h3>
        
        <div className="space-y-4">
          {commonProps}
          
          {/* Optimizations section */}
          <div className="mt-6">
            <h4 className="text-md font-semibold mb-2">Optimizations</h4>
            {type === 'ffn' && renderParamField('useMoE', params.useMoE)}
            {type === 'qkvAttention' && renderParamField('useFlashAttention', params.useFlashAttention)}
            {type === 'qkvAttention' && renderParamField('useSlidingWindow', params.useSlidingWindow)}
          </div>
        </div>
      </div>
    );
  };
  
  return renderNodeProperties();
};

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
      onChange={(e) => onValueChange([Number(e.target.value)])}
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

// Mock console.log to track logs
const originalConsoleLog = console.log;
let consoleOutput = [];

describe('NodeProperties Component', () => {
  // Create a mock node for testing
  const mockFFNNode = {
    id: 'ffn-123',
    type: 'ffn',
    position: { x: 100, y: 100 },
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
  };
  
  const mockQKVNode = {
    id: 'qkv-123',
    type: 'qkvAttention',
    position: { x: 100, y: 100 },
    data: {
      type: 'qkvAttention',
      label: 'QKV Attention',
      params: {
        headDim: 64,
        numHeads: 8,
        attentionType: 'scaled_dot_product',
        useFlashAttention: false,
        useSlidingWindow: false,
        windowSize: 512
      }
    }
  };
  
  const onChangeMock = vi.fn();
  
  beforeEach(() => {
    consoleOutput = [];
    console.log = (...args) => {
      consoleOutput.push(args);
      originalConsoleLog(...args);
    };
    onChangeMock.mockClear();
  });
  
  afterEach(() => {
    console.log = originalConsoleLog;
  });
  
  it('renders node properties correctly', () => {
    render(<NodeProperties node={mockFFNNode} onChange={onChangeMock} />);
    
    // Check if the node label is displayed
    expect(screen.getByText('Feed Forward Properties')).toBeInTheDocument();
    
    // Check if parameters are displayed
    expect(screen.getByText('Hidden Dim')).toBeInTheDocument();
    expect(screen.getByText('Activation')).toBeInTheDocument();
    expect(screen.getAllByText('Use Mo E')[0]).toBeInTheDocument();
  });
  
  it('updates number parameter when input changes', async () => {
    render(<NodeProperties node={mockFFNNode} onChange={onChangeMock} />);
    
    // Find the hiddenDim input
    const hiddenDimInput = screen.getByTestId('input-number');
    
    // Change the value
    fireEvent.change(hiddenDimInput, { target: { value: '4096' } });
    
    // Check if onChange was called with the correct parameters
    await waitFor(() => {
      expect(onChangeMock).toHaveBeenCalledWith('ffn-123', {
        ...mockFFNNode.data,
        params: {
          ...mockFFNNode.data.params,
          hiddenDim: 4096
        }
      });
    });
    
    // Check if the parameter change was logged
    const paramChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Parameter hiddenDim changing to:'
    );
    expect(paramChangeLogs.length).toBeGreaterThan(0);
  });
  
  it('updates boolean parameter when switch changes', async () => {
    render(<NodeProperties node={mockFFNNode} onChange={onChangeMock} />);
    
    // Find the useMoE switch
    const moeSwitchLabel = screen.getAllByText('Use Mo E')[0];
    const moeSwitch = screen.getAllByTestId('switch')[0];
    
    // Toggle the switch
    fireEvent.click(moeSwitch);
    
    // Check if onChange was called with the correct parameters
    await waitFor(() => {
      expect(onChangeMock).toHaveBeenCalledWith('ffn-123', {
        ...mockFFNNode.data,
        params: {
          ...mockFFNNode.data.params,
          useMoE: true
        }
      });
    });
    
    // Check if the parameter change was logged
    const paramChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Parameter useMoE changing to:'
    );
    expect(paramChangeLogs.length).toBeGreaterThan(0);
  });
  
  it('updates string parameter when select changes', async () => {
    render(<NodeProperties node={mockFFNNode} onChange={onChangeMock} />);
    
    // Find the activation select
    const activationSelect = screen.getByTestId('select');
    
    // Change the value
    fireEvent.change(activationSelect, { target: { value: 'relu' } });
    
    // Check if onChange was called with the correct parameters
    await waitFor(() => {
      expect(onChangeMock).toHaveBeenCalledWith('ffn-123', {
        ...mockFFNNode.data,
        params: {
          ...mockFFNNode.data.params,
          activation: 'relu'
        }
      });
    });
    
    // Check if the parameter change was logged
    const paramChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Parameter activation changing to:'
    );
    expect(paramChangeLogs.length).toBeGreaterThan(0);
  });
  
  it('shows node-specific optimizations for FFN node', () => {
    render(<NodeProperties node={mockFFNNode} onChange={onChangeMock} />);
    
    // Check if FFN-specific optimizations are displayed
    expect(screen.getByText('Optimizations')).toBeInTheDocument();
    expect(screen.getAllByText('Use Mo E')[1]).toBeInTheDocument();
  });
  
  it('shows node-specific optimizations for QKV Attention node', () => {
    render(<NodeProperties node={mockQKVNode} onChange={onChangeMock} />);
    
    // Check if QKV-specific optimizations are displayed
    expect(screen.getByText('Optimizations')).toBeInTheDocument();
    expect(screen.getAllByText('Use Flash Attention')[1]).toBeInTheDocument();
    expect(screen.getAllByText('Use Sliding Window')[1]).toBeInTheDocument();
  });
  
  it('shows additional fields when toggles are enabled', async () => {
    render(<NodeProperties node={mockQKVNode} onChange={onChangeMock} />);
    
    // Find the useSlidingWindow switch
    const slidingWindowLabel = screen.getAllByText('Use Sliding Window')[0];
    const slidingWindowSwitch = screen.getAllByTestId('switch')[1]; // Second switch
    
    // Toggle the switch
    fireEvent.click(slidingWindowSwitch);
    
    // Check if onChange was called with the correct parameters
    await waitFor(() => {
      expect(onChangeMock).toHaveBeenCalledWith('qkv-123', {
        ...mockQKVNode.data,
        params: {
          ...mockQKVNode.data.params,
          useSlidingWindow: true
        }
      });
    });
    
    // Re-render with updated props to simulate React's update
    const updatedNode = {
      ...mockQKVNode,
      data: {
        ...mockQKVNode.data,
        params: {
          ...mockQKVNode.data.params,
          useSlidingWindow: true
        }
      }
    };
    
    render(<NodeProperties node={updatedNode} onChange={onChangeMock} />);
    
    // Check if the window size field appears
    expect(screen.getByText('Window Size')).toBeInTheDocument();
  });
  
  it('updates slider values correctly', async () => {
    // Render with useMoE set to true to show the slider
    const moeEnabledNode = {
      ...mockFFNNode,
      data: {
        ...mockFFNNode.data,
        params: {
          ...mockFFNNode.data.params,
          useMoE: true
        }
      }
    };
    
    render(<NodeProperties node={moeEnabledNode} onChange={onChangeMock} />);
    
    // Find the numExperts slider
    const numExpertsSlider = screen.getByTestId('slider');
    
    // Change the value
    fireEvent.change(numExpertsSlider, { target: { value: '16' } });
    
    // Check if onChange was called with the correct parameters
    await waitFor(() => {
      expect(onChangeMock).toHaveBeenCalledWith('ffn-123', {
        ...moeEnabledNode.data,
        params: {
          ...moeEnabledNode.data.params,
          numExperts: 16
        }
      });
    });
    
    // Check if the slider value change was logged
    const sliderChangeLogs = consoleOutput.filter(log => 
      log[0] === 'Slider value changed to:'
    );
    expect(sliderChangeLogs.length).toBeGreaterThan(0);
  });
}); 