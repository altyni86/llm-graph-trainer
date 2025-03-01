import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

// A simple component to test
const SimpleComponent = ({ title }: { title: string }) => {
  return (
    <div>
      <h1 data-testid="title">{title}</h1>
      <p>This is a simple component for testing.</p>
    </div>
  );
};

describe('Simple Component', () => {
  it('renders correctly', () => {
    render(<SimpleComponent title="Test Title" />);
    expect(screen.getByTestId('title')).toHaveTextContent('Test Title');
  });
}); 