# Asynchronous Advantage Actor-Critic in PyTorch

This is PyTorch implementation of A3C as described in [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v1.pdf).

Since PyTorch has a easy method to control shared memory within multiprocess, we can easily implement asynchronous method like A3C.


## Requirement
* PyTorch 0.1.6
* Python 3.5.2
* gym 0.7.2

## Usage
### training
```
python run_a3c.py
```
In default settings, num_process is 8. Set it as `python run_a3c --num_process 4` to fit your number of cpu's cores.

### test
After training
```
python test_a3c.py --render --monitor
```



