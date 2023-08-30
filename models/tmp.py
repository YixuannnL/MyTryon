import torch
import torch.nn as nn

import pdb

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.nested_layers = nn.ModuleList([
            nn.Linear(100, 50),
            nn.ModuleList([
                nn.Sequential(
                        nn.Linear(50, 30),
                        nn.ReLU()),
                nn.Linear(30, 10)
            ])
        ])

    def forward(self, x):
        for layer in self.nested_layers:
            pdb.set_trace()
            if isinstance(layer, nn.ModuleList):
                x = self._forward_nested_module_list(layer, x)
            else:
                x = layer(x)
        return x

    def _forward_nested_module_list(self, module_list, x):
        for layer in module_list:
            x = layer(x)
        return x
    
if __name__ == '__main__':
    model = MyModel()
    x = torch.Tensor(size=(10,100))
    result = model(x)
    print(result.shape)