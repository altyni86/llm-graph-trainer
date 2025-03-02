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
    expect(screen.getByRole('tab', { name: /templates/i })).toHaveAttribute('data-state', 'inactive');
    
    // Set up user event
    const user = userEvent.setup();
    
    // Click on the 'templates' tab using userEvent
    await user.click(screen.getByRole('tab', { name: /templates/i }));
    
    // Verify that setActiveTab was called with 'templates'
    expect(setActiveTabMock).toHaveBeenCalledWith('templates');
    
    // Simulate the parent component updating the activeTab prop
    rerender(
      <NodePanel 
        setNodes={setNodesMock} 
        activeTab="templates" 
        setActiveTab={setActiveTabMock} 
      />
    );
    
    // Verify that the 'templates' tab is now active
    expect(screen.getByRole('tab', { name: /templates/i })).toHaveAttribute('data-state', 'active');
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
}); 