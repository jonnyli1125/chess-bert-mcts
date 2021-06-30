# chess-bert-mcts

Chess AI based on BERT + MCTS.

In contrast to how most existing board game AI models use computer vision techniques (such as convolutional neural networks), this project explores the effectiveness of natural language processing techniques in neural networks for board game AI.

In a nutshell, the model architecture is the same as AlphaZero, except we use BERT and attach custom policy/value heads to it, and it is trained using supervised learning instead of reinforcement learning.

You can play the bot online (when it's running) at https://lichess.org/@/chess-bert-mcts.

# Datasets

The [CCRL 40/15 dataset](http://ccrl.chessdom.com/ccrl/4040/index.html) was used for both pre-training BERT for Masked Language Modeling as well as fine tuning for the policy/value network heads. The dataset contains many chess games played by strong computer AI with a variety of different outcomes, and also is annotated with Stockfish evaluations.

The data is preprocessed into a vector representation of the board state (position of pieces, en passant, castling rights), and the corresponding next move (policy) and Stockfish evaluation (value) are recorded as labels.

For masked language modeling, we randomly select 15% of tokens in the board state vector, then replace 80% of those with the mask token, 10% with a random token, and leave 10% unchanged.

# Training

The model was trained in Colab with 8 TPUs for 1 epoch for both the MLM and Policy/Value stages. Only a quarter of the dataset was used.

The trained model weights can be downloaded [here](https://drive.google.com/file/d/1oRG5m-7oLhiEnz2rVKqj3um3JApkPXwm/view?usp=sharing).
