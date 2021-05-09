from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from typing import Callable

from train_gcn.state_classes import AdamTrainingProfile, TrainingProfile
    
def make_optimizer(profile: TrainingProfile, model: nn.Module) -> Optimizer:
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
          model: nn.Module,
          optimizer: Optimizer,
          criterion: Criterion,
          label: str
          ) -> None:
    model.train()  # enable learning-only behavior
    for batched_graph, labels in data:
        optimizer.zero_grad()
        if label == 'RelGraphConv':
            pred = model(batched_graph, batched_graph.ndata['attr'].float(), batched_graph.edata['encode'].int())
        else:
            pred = model(batched_graph, batched_graph.ndata['attr'].float())
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        return loss

def test(epoch_name: str,
         data: GraphDataLoader,
         model: nn.Module,
         label: str
         ) -> float:  # returns accuracy
    model.eval()  # disable learning-only behavior
    with torch.no_grad():  # skip computation of gradients
        def correct(batched_graph, labels):
            if label == 'RelGraphConv':
                pred = model(batched_graph, batched_graph.ndata['attr'].float(), batched_graph.edata['encode'].int())
            else:
                pred = model(batched_graph, batched_graph.ndata['attr'].float())
            return (pred.argmax(1) == labels).sum().item()
        num_correct = sum(correct(batched_graph, labels) for batched_graph, labels in data)
        num_tests = sum(len(labels) for batched_graph, labels in data)
        accuracy = num_correct / num_tests
    return accuracy
        
def epoch(name: str,
          training_data: GraphDataLoader,
          testing_data: GraphDataLoader,
          model: nn.Module,
          optimizer: Optimizer,
          criterion: Criterion,
          label: str
          ) -> float:  # returns the accuracy.
    loss = train(name, training_data, model, optimizer, criterion, label)
    test_accuracy = test(name, testing_data, model, label)
    print(f'Epoch {name} has training loss {loss} and accuracy {100 * test_accuracy :.6} against the test data.')
    return loss,test_accuracy * 100

def validate(data: GraphDataLoader,
         model: nn.Module,
         label: str
         ) -> float:  # returns accuracy
    model.eval()  # disable learning-only behavior
    with torch.no_grad():  # skip computation of gradients
        def correct(batched_graph, labels):
            if label == 'RelGraphConv':
                pred = model(batched_graph, batched_graph.ndata['attr'].float(), batched_graph.edata['encode'].int())
            else:
                pred = model(batched_graph, batched_graph.ndata['attr'].float())
            return (pred.argmax(1) == labels).sum().item()
        num_correct = sum(correct(batched_graph, labels) for batched_graph, labels in data)
        num_tests = sum(len(labels) for batched_graph, labels in data)
        accuracy = num_correct / num_tests
        print('Validation accuracy:',accuracy)
    return accuracy * 100
    

