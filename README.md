# Dream-to-Control-for-Online-Decision-Transformer

This is the implementation of Dreamer for the Online Decision Transformer. It uses the Online Decision Transformer as the baseline model, then in parallel, it trains a Dreamer model. After both models are trained, the trajectories from Dreamer are used to train the Online Decision Transformer Model for better learning processes.

## How to use

Install dependencies:
```bash
pip install -r pip.txt
```

## Start

Download the Decision Transformer dataset:
```bash
cd data
python data.py
```

To run Dream to Control for Online Decision Transformer:
```bash
cd ../
python main.py
```

## For any questions, please contact:
ericjiang0318@g.ucla.edu
