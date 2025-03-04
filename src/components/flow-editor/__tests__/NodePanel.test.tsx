import { render, screen } from '@testing-library/react';
import { NodePanel } from '../NodePanel';
import { describe, it, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';

describe('NodePanel Component', () => {
  it('preserves tab state when switching between tabs', async () => {
    // Mock the setNodes function
    const setNodesMock = vi.fn();
    const setActiveTabMock = vi.fn();
    
    // Render with 'components' tab active
    const { rerender } = render(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="components" 
        setActiveTab={setActiveTabMock} 
      />
    );
    
    // Verify that the 'components' tab is active initially
    expect(screen.getByRole('tab', { name: /components/i })).toHaveAttribute('data-state', 'active');
    expect(screen.getByRole('tab', { name: /post training/i })).toHaveAttribute('data-state', 'inactive');
    
    // Set up user event
    const user = userEvent.setup();
    
    // Click on the 'post training' tab using userEvent
    await user.click(screen.getByRole('tab', { name: /post training/i }));
    
    // Verify that setActiveTab was called with 'postTraining'
    expect(setActiveTabMock).toHaveBeenCalledWith('postTraining');
    
    // Simulate the parent component updating the activeTab prop
    rerender(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="postTraining" 
        setActiveTab={setActiveTabMock} 
      />
    );
    
    // Verify that the 'post training' tab is now active
    expect(screen.getByRole('tab', { name: /post training/i })).toHaveAttribute('data-state', 'active');
    expect(screen.getByRole('tab', { name: /components/i })).toHaveAttribute('data-state', 'inactive');
  });
  
  it('adds a node to the canvas when a component is clicked', async () => {
    // Mock the setNodes function
    const setNodesMock = vi.fn();
    
    // Render the component
    render(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="components" 
        setActiveTab={vi.fn()} 
      />
    );
    
    // Set up user event
    const user = userEvent.setup();
    
    // Click on the first component (Embedding)
    await user.click(screen.getByText('Embedding'));
    
    // Verify that setNodes was called
    expect(setNodesMock).toHaveBeenCalled();
  });

  it('displays training nodes in the Post Training tab', async () => {
    // Mock the setNodes function
    const setNodesMock = vi.fn();
    
    // Render with 'postTraining' tab active
    render(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="postTraining" 
        setActiveTab={vi.fn()} 
      />
    );
    
    // Verify that training nodes are displayed
    expect(screen.getByText('SFT Training')).toBeInTheDocument();
    expect(screen.getByText('PPO Training')).toBeInTheDocument();
    expect(screen.getByText('DPO Training')).toBeInTheDocument();
    expect(screen.getByText('GRPO Training')).toBeInTheDocument();
  });

  it('adds a training node to the canvas when clicked', async () => {
    // Mock the setNodes function
    const setNodesMock = vi.fn();
    
    // Render with 'postTraining' tab active
    render(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="postTraining" 
        setActiveTab={vi.fn()} 
      />
    );
    
    // Set up user event
    const user = userEvent.setup();
    
    // Click on the SFT Training node
    await user.click(screen.getByText('SFT Training'));
    
    // Verify that setNodes was called
    expect(setNodesMock).toHaveBeenCalled();
  });

  it('allows dragging training nodes from the Post Training tab', () => {
    // Mock the setNodes function
    const setNodesMock = vi.fn();
    
    // Render with 'postTraining' tab active
    render(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="postTraining" 
        setActiveTab={vi.fn()} 
      />
    );
    
    // Get the SFT Training card
    const sftCard = screen.getByText('SFT Training').closest('.cursor-grab');
    
    // Verify that the card is draggable
    expect(sftCard).toHaveAttribute('draggable', 'true');
  });
}); 