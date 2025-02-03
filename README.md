# A3C Kung Fu Master

A deep reinforcement learning project that implements the Asynchronous Advantage Actor-Critic (A3C) algorithm to train an AI agent to play Kung Fu Master in the OpenAI Gymnasium environment.

## Overview

This project uses parallel environment processing and a convolutional neural network architecture to train an agent to play the classic arcade game Kung Fu Master. The implementation leverages PyTorch for the neural network and Gymnasium (formerly OpenAI Gym) for the game environment.

## Key Features

- **A3C Implementation**: Uses the A3C algorithm with both actor and critic networks for efficient learning
- **Parallel Processing**: Trains across multiple environment instances simultaneously
- **CNN Architecture**: Custom convolutional neural network to process game frames
- **Frame Preprocessing**: Includes frame stacking, resizing, and grayscale conversion
- **Visual Results**: Includes video recording of trained agent gameplay

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium with Atari environments
- OpenCV
- NumPy
- Additional dependencies listed in the notebook

## Usage

1. Install the required dependencies:
```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install gymnasium[box2d]
```

2. Open and run the Jupyter notebook `A3C_for_Kung_Fu.ipynb`

3. The notebook will:
   - Set up the environment
   - Train the agent over 10,000 iterations
   - Display periodic evaluation results
   - Generate a video of the trained agent playing

## Training

The agent is trained using 10 parallel environments for 10,000 iterations. The training process includes:
- State preprocessing and frame stacking
- Action selection using the policy network
- Parallel environment stepping
- Network updates using both actor and critic losses

## Results

The training progress can be monitored through periodic evaluations, and the final trained agent's performance is captured in a video demonstration.

## Implementation Details

### Neural Network Architecture
The project uses a CNN with:
- 3 convolutional layers with ReLU activation
- Fully connected layer with 128 units
- Separate actor (policy) and critic (value) output heads

### Environment Preprocessing
- Frame resizing to 42x42 pixels
- Grayscale conversion
- Frame stacking (4 frames)
- Reward scaling

### A3C Algorithm
- Parallel environment processing for faster training
- Policy gradient updates with advantage estimation
- Value function learning for baseline estimation
- Entropy regularization for exploration

## License

This project is open source and available under the MIT License.
