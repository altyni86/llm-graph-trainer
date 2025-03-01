# LLM Graph Builder

A visual tool for constructing LLM (Large Language Model) training components and generating PyTorch code.

## Features

- **Visual Component Builder**: Drag and drop LLM components to create your architecture
- **PyTorch Code Generation**: Generate ready-to-use PyTorch code from your visual design
- **Component Library**: Access embeddings, positional encodings, QKV blocks, and more
- **Optimization Options**: Configure training optimizations like FSDP, Flash Attention, MoE, and more
- **Training Hyperparameters**: Fine-tune batch size, learning rate, model dimensions, and more
- **Device Detection**: Automatically detect and use the best available hardware (CUDA, MPS, CPU)
- **Experiment Runner**: Run small-scale experiments with synthetic data to test your model

## Available Components

- **Embedding Layers**: Convert token IDs to embeddings
- **Positional Encodings**: Add position information to embeddings (Sinusoidal, Learned, Rotary, ALiBi)
- **Multi-Head Attention**: Self-attention mechanisms with configurable parameters
- **Feed Forward Networks**: Process features with non-linearity
- **Output Layers**: Final projection layers with various activation functions

## Optimization Options

- **Training Hyperparameters**:
  - Batch size, block size (context length), and maximum iterations
  - Learning rate and evaluation intervals
  - Model architecture parameters (embedding dimension, number of heads/layers)
  - Dropout rate for regularization

- **Distributed Training**:
  - Fully Sharded Data Parallel (FSDP) with configurable sharding strategies
  - DeepSpeed ZeRO with CPU offloading options

- **Mixture of Experts (MoE)**:
  - Configure number of experts and routing strategy
  - Set top-k experts per token (Switch Transformers for k=1, standard MoE for k=2)
  - Adjust capacity factors for training and evaluation
  - Enable expert parallelism for multi-GPU setups
  - Control expert dropout for better generalization

- **Attention Optimizations**:
  - Flash Attention for faster, memory-efficient attention
  - xFormers memory-efficient attention mechanisms

- **Memory Optimizations**:
  - Gradient checkpointing to reduce memory usage
  - Mixed precision training (FP16/BF16)

- **Compilation**:
  - PyTorch 2.0 torch.compile() with different compilation modes

- **Device Detection**:
  - Automatic detection of CUDA GPUs
  - Support for Apple Silicon GPUs via Metal Performance Shaders (MPS)
  - Fallback to CPU when no GPU is available

- **Experiment Features**:
  - Run small-scale experiments with synthetic data
  - Configure batch size, epochs, and sequence length
  - Track and visualize training metrics (loss, timing)
  - Save model checkpoints during training

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/llm-graph-trainer.git
cd llm-graph-trainer
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Navigate to the Builder page
2. Drag components from the left panel onto the canvas
3. Connect components by dragging from one node's output handle to another node's input handle
4. Configure component parameters by clicking on them
5. Go to the Optimizations tab to configure training optimizations
6. Configure device detection and experiment settings in the Experiment tab
7. Click "Generate Code" to create PyTorch code for your model
8. Copy or download the generated code for use in your PyTorch projects

## Running Experiments

The generated code includes functionality to run small-scale experiments with your model:

1. Configure experiment settings in the Experiment tab:
   - Set batch size, epochs, and sequence length
   - Enable metrics tracking and checkpoint saving
   - Configure synthetic dataset size

2. The generated code will include a `run_experiment()` function that:
   - Automatically detects the best available device (CUDA, MPS, CPU)
   - Generates synthetic data for training
   - Trains the model for the specified number of epochs
   - Tracks and visualizes training metrics
   - Saves model checkpoints

3. Run the generated Python code:
```bash
python your_model.py
```

4. View the results in the `experiment_results` directory:
   - Training loss plots
   - Performance metrics
   - Model checkpoints

## Testing

The LLM Graph Trainer includes a comprehensive test suite to ensure the application works as expected. The tests focus on verifying that:

1. The synchronization between the nodes array and the selectedNode state works correctly
2. Parameter updates from the properties panel are reflected in the node data
3. Changes to nodes from other sources (like validation) are reflected in the properties panel
4. Multiple parameter updates in sequence are handled correctly

### Running Tests

To run the tests, first install the dependencies:

```bash
npm install
```

Then run the tests using one of the following commands:

```bash
# Run tests in watch mode
npm test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

### Test Structure

The tests are organized into several files:

- `FlowEditor.test.tsx`: Tests for the main FlowEditor component
- `NodeProperties.test.tsx`: Tests for the NodeProperties component
- `NodeSynchronization.test.tsx`: Tests specifically for the node synchronization mechanism
- `Integration.test.tsx`: Integration tests between FlowEditor and NodeProperties

### Key Test Cases

1. **State Synchronization**: Tests verify that when a node is updated through any means, both the nodes array and the selectedNode state are kept in sync.

2. **Parameter Updates**: Tests check that parameter changes in the properties panel are correctly applied to the node data.

3. **Conditional Rendering**: Tests ensure that conditional UI elements (like the MoE settings when useMoE is enabled) appear and disappear correctly.

4. **Multiple Updates**: Tests confirm that multiple parameter updates in sequence are all applied correctly.

## Technologies Used

- Next.js
- React
- TypeScript
- Tailwind CSS
- Shadcn UI
- React Flow
- Monaco Editor

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the need for easier LLM architecture experimentation
- Built with modern web technologies for a smooth user experience
