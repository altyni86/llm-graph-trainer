#!/bin/bash

# Install dependencies if needed
if [ "$1" == "--install" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Install Vitest and React Testing Library if not already installed
if ! npm list vitest > /dev/null 2>&1; then
  echo "Installing Vitest and testing dependencies..."
  npm install --save-dev vitest @vitest/ui @vitest/coverage-v8 jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event @vitejs/plugin-react
fi

# Run tests based on command line arguments
if [ "$1" == "--ui" ]; then
  echo "Running tests with UI..."
  npm run test:ui
elif [ "$1" == "--coverage" ]; then
  echo "Running tests with coverage..."
  npm run test:coverage
else
  echo "Running tests..."
  npm test
fi 