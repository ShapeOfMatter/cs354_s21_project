from dgl.dataloading import GraphDataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from typing import Callable

from train_gcn.model import GCN
from train_gcn.state_classes import AdamTrainingProfile, TrainingProfile
    
def make_optimizer(profile: TrainingProfile, model: GCN) -> Optimizer:
    if isinstance(profile, AdamTrainingProfile):
        return Adam(model.parameters(),
                    lr=profile.learning_rate,
                    betas=(profile.beta_1, profile.beta_2),
                    eps=profile.epsilon,
                    weight_decay=profile.weight_decay,
                    amsgrad=profile.ams_gradient_variant)
    else:
        raise Exception("We haven't implemented any other optimizers.")

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # IDK how to tighten this.

def make_criterion() -> Criterion:
    return F.cross_entropy

def train(epoch_name: str,
          data: GraphDataLoader,
          model: GCN,
          optimizer: Optimizer,
          criterion: Criterion
          ) -> None:
    model.train()  # enable learning-only behavior
    for batched_graph, labels in data:
        optimizer.zero_grad()
        pred = model(batched_graph, batched_graph.ndata['attr'].float())
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        # Log something?

def test(epoch_name: str,
         data: GraphDataLoader,
         model: GCN
         ) -> float:  # returns accuracy
    model.eval()  # disable learning-only behavior
    with torch.no_grad(): # skip computation of gradients
        def correct(batched_graph, labels):
            pred = model(batched_graph, batched_graph.ndata['attr'].float())
            return (pred.argmax(1) == labels).sum().item()
        num_correct = sum(correct(batched_graph, labels) for batched_graph, labels in data)
        num_tests = sum(len(labels) for batched_graph, labels in data)
        accuracy = num_correct / num_tests
    return accuracy
        
def epoch(name: str,
          training_data: GraphDataLoader,
          testing_data: GraphDataLoader,
          model: GCN,
          optimizer: Optimizer,
          criterion: Criterion
          ) -> float:  # returns the accuracy.
    train(name, training_data, model, optimizer, criterion)
    accuracy = test(name, testing_data, model)
    print(f'Epoch {name} has accuracy {100 * accuracy :.6} against the test data.')
    return accuracy


