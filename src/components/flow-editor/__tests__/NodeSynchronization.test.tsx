import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

// Mock ReactFlow component
vi.mock('reactflow', async () => {
  const mockReactFlow = ({ children }) => (
    <div data-testid="react-flow">
      {children}
    </div>
  );
  
  return {
    default: mockReactFlow,
    ReactFlow: mockReactFlow,
    useNodesState: () => {
      const [nodes, setNodes] = React.useState([]);
      const onNodesChange = () => {};
      return [nodes, setNodes, onNodesChange];
    },
    useEdgesState: () => {
      const [edges, setEdges] = React.useState([]);
      const onEdgesChange = () => {};
      return [edges, setEdges, onEdgesChange];
    },
    Background: () => <div data-testid="background"></div>,
    Controls: () => <div data-testid="controls"></div>,
    Panel: ({ children }) => <div>{children}</div>,
  };
});

// Basic component for testing node synchronization
function BasicSyncTest() {
  const [nodes, setNodes] = React.useState([
    {
      id: 'node-1',
      data: { 
        label: 'Test Node',
        params: { value: 10 }
      }
    }
  ]);
  
  const [selectedNode, setSelectedNode] = React.useState(null);
  
  // Select the node when component mounts
  React.useEffect(() => {
    if (nodes.length > 0 && !selectedNode) {
      setSelectedNode(nodes[0]);
    }
  }, [nodes, selectedNode]);
  
  // Keep selectedNode in sync with nodes
  React.useEffect(() => {
    if (selectedNode) {
      const updatedNode = nodes.find(node => node.id === selectedNode.id);
      if (updatedNode && JSON.stringify(updatedNode) !== JSON.stringify(selectedNode)) {
        console.log('Syncing selectedNode with nodes array');
        setSelectedNode(updatedNode);
      }
    }
  }, [nodes, selectedNode]);
  
  // Update node value directly (simulating external update)
  const updateNodeDirectly = () => {
    setNodes(prev => 
      prev.map(node => 
        node.id === 'node-1' 
          ? { 
              ...node, 
              data: { 
                ...node.data, 
                params: { 
                  ...node.data.params, 
                  value: 20 
                } 
              } 
            } 
          : node
      )
    );
  };
  
  // Update node through properties panel
  const updateNodeThroughPanel = () => {
    if (selectedNode) {
      const updatedData = {
        ...selectedNode.data,
        params: {
          ...selectedNode.data.params,
          value: 30
        }
      };
      
      // Update nodes array
      setNodes(prev => 
        prev.map(node => 
          node.id === selectedNode.id 
            ? { ...node, data: updatedData } 
            : node
        )
      );
      
      // Update selectedNode
      setSelectedNode({
        ...selectedNode,
        data: updatedData
      });
    }
  };
  
  return (
    <div>
      <div data-testid="node-display">
        Node Value: {nodes[0]?.data.params.value}
      </div>
      
      {selectedNode && (
        <div data-testid="selected-node-display">
          Selected Node Value: {selectedNode.data.params.value}
        </div>
      )}
      
      <button 
        data-testid="update-directly" 
        onClick={updateNodeDirectly}
      >
        Update Directly
      </button>
      
      <button 
        data-testid="update-through-panel" 
        onClick={updateNodeThroughPanel}
      >
        Update Through Panel
      </button>
    </div>
  );
}

describe('Node Synchronization Basic Tests', () => {
  it('keeps selectedNode in sync when nodes are updated directly', () => {
    render(<BasicSyncTest />);
    
    // Verify initial state
    expect(screen.getByText('Node Value: 10')).toBeInTheDocument();
    expect(screen.getByText('Selected Node Value: 10')).toBeInTheDocument();
    
    // Update node directly
    fireEvent.click(screen.getByTestId('update-directly'));
    
    // Verify both values are updated
    expect(screen.getByText('Node Value: 20')).toBeInTheDocument();
    expect(screen.getByText('Selected Node Value: 20')).toBeInTheDocument();
  });
  
  it('updates both nodes and selectedNode when updated through panel', () => {
    render(<BasicSyncTest />);
    
    // Verify initial state
    expect(screen.getByText('Node Value: 10')).toBeInTheDocument();
    expect(screen.getByText('Selected Node Value: 10')).toBeInTheDocument();
    
    // Update through panel
    fireEvent.click(screen.getByTestId('update-through-panel'));
    
    // Verify both values are updated
    expect(screen.getByText('Node Value: 30')).toBeInTheDocument();
    expect(screen.getByText('Selected Node Value: 30')).toBeInTheDocument();
  });
}); 