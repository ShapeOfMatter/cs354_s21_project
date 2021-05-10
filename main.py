#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import TAGConv
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch.utils import Sequential
import numpy as np
import os
import os.path as path
import sys
import torch
from typing import Any, Callable, List, Sequence

from train_gcn.dgldataset import get_dataloaders
from train_gcn.model import GCN, make_tagcn, RelGraphConvN, RELU, RelationalTAGConv, TAGConvN
from train_gcn.state_classes import Settings
from train_gcn.train import epoch, make_criterion, make_optimizer, validate
from plots import create_plots


@dataclass
class Model:
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: Any = field(default_factory=make_criterion)  # Boo! Hiss! should be Callable[[torch.Tensor, torch.Tensor], torch.Tensor] but mypy sucks.
    losses: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    validation_accuracy: float = np.inf
    def stats(self):
        return {'loss': self.losses,
                'test accuracy': self.test_accuracies,
                'validation accuracy': self.validation_accuracy}  # These don't have the same shape?


def check_file_blocks(*files: str):
    '''Returns True iff none of the files exist yet.'''
    return not any(path.exists(f) for f in files)


def main(settings: Settings):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for crashing.
    print("settings.max_batch_size: ", settings.max_batch_size)
    train_dataloader, test_dataloader, val_dataloader, output_width,node_attributes, edge_attributes = get_dataloaders(settings)

    # Since dataloaders can not easily access the whole dataset, these calculations are done when the datasets are generated.
    print(f'There are {output_width} labels in the data.')
    print('Node attributes: ', node_attributes)
    print('Edge attributes: ', edge_attributes)
    
    models: List[Model] = []
    
    # Most basic GCN
    model1 = GCN(len(node_attributes), 16, output_width)
    models.append(Model(name='GCN',
                        model=model1,
                        optimizer=make_optimizer(settings.training_profile, model1)))

    # RelGraphConv
    # We can't parameterize num_rels because it's not a simple function of edge_attrubutes.
    model2 = RelGraphConvN(len(node_attributes), 16, output_width, num_rels=5)
    models.append(Model(name='RelGraphConv',
                        model=model2,
                        optimizer=make_optimizer(settings.training_profile, model2)))

    # TAGConv
    model3 = TAGConvN(len(node_attributes), 16, output_width)
    models.append(Model(name='TAGConv',
                        model=model3,
                        optimizer=make_optimizer(settings.training_profile, model3)))

    # R-TAG-Conv
    model4 = Sequential(RelationalTAGConv(radius=2, width_in=len(node_attributes), forward_edge=8, backward_edge=8),
                        TAGConv(16, output_width, k=0),
                        AvgPooling())
    models.append(Model(name='R-TAG-Conv',
                        model=model4,
                        optimizer=make_optimizer(settings.training_profile, model4)))
    
    for epoch_number in range(settings.epochs):
        epoch_all(settings, models, epoch_number, train=train_dataloader, test=test_dataloader)
        print(f'Finished epoch {epoch_number}.')

    for m in models:
        m.validation_accuracy = validate(val_dataloader, m.model, m.name)

    create_plots({m.name: m.stats() for m in models})
    

def epoch_all(settings: Settings, models: Sequence[Model], epoch_number: int, *, train: GraphDataLoader, test: GraphDataLoader):
    for m in models:
        new_loss, new_accuracy = epoch(f'e_{epoch_number}_{m.name}',
                                       training_data=train,
                                       testing_data=test,
                                       model=m.model,
                                       optimizer=m.optimizer,
                                       criterion=m.criterion,
                                       label=m.name)
        if len(m.losses) == 0:
            m.losses.append(new_loss)
        else:
            if new_loss < min(m.losses):
                print(f'New lowest loss for {m.name}. Saving weights!')
                torch.save(m.model.state_dict(), settings.model_filename + m.name)
        m.losses.append(new_loss)
        m.test_accuracies.append(new_accuracy)


if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))




