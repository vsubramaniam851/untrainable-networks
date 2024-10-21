import torch
import torch.nn as nn
from collections import OrderedDict
from language_modeling import TransformerLM, Transformer, RNNLM, ParityLSTM

class FeatureMapExtractor:
    '''
    Activation extractor for pytorch networks. This extracts activations from outputs of set layers based on whether the layers have tunable parameters.
    Pass in your model and run `get_feature_maps` as follows:
    >>> model = YourModel()
    >>> extractor = FeatureMapExtractor(model)
    >>> feature_maps = extractor.get_feature_maps(inputs)
    '''
    def __init__(self, model, enforce_input_shape = True, eval = True):
        self.model = model
        self.enforce_input_shape = enforce_input_shape
        self.layers_to_retain = [nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, nn.LSTM, nn.RNN, nn.TransformerDecoderLayer, nn.LayerNorm, nn.MultiheadAttention]
        self.feature_maps = OrderedDict()
        self.hooks = []
        self.eval = eval
        self.device = next(model.parameters()).device

    @staticmethod
    def get_module_name(module, feature_maps):
        return f'{type(module).__name__}_{len(feature_maps)}'

    @staticmethod
    def get_module_type(module):
        return type(module)

    @staticmethod
    def check_for_input_axis(feature_map, input_size):
        axis_match = [dim for dim in feature_map.shape if dim == input_size]
        return True if len(axis_match) == 1 else False

    @staticmethod
    def reset_input_axis(feature_map, input_size):
        input_axis = feature_map.shape.index(input_size)
        return torch.swapaxes(feature_map, 0, input_axis)

    def register_hook(self, module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if isinstance(output, nn.utils.rnn.PackedSequence):
                    output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first = True)
                if isinstance(output, torch.Tensor):
                    output = output.to(self.device)
                    if self.enforce_input_shape:
                        if output.shape[0] == self.inputs.shape[0]:
                            self.feature_maps[module_name] = output
                        else:
                            if self.check_for_input_axis(output, self.inputs.shape[0]):
                                output = self.reset_input_axis(output, self.inputs.shape[0])
                                self.feature_maps[module_name] = output
                            else:
                                self.feature_maps[module_name] = None
                    else:
                        self.feature_maps[module_name] = output

            module_type = self.get_module_type(module)
            if module_type in self.layers_to_retain:
                module_name = self.get_module_name(module, self.feature_maps)
                if any([isinstance(output, type_) for type_ in (tuple, list)]):
                    if module_type in [nn.RNN, nn.LSTM]:
                        output = output[:-1]
                    for output_i, output_ in enumerate(output):
                        module_name_ = '-'.join([module_name, str(output_i+1)])
                        process_output(output_, module_name_)
                else:
                    process_output(output, module_name)

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            self.hooks.append(module.register_forward_hook(hook))

    def get_feature_maps(self, inputs, **kwargs):
        self.inputs = inputs
        self.feature_maps = OrderedDict()
        self.hooks = []
        self.model.apply(self.register_hook)

        if self.eval:
            with torch.no_grad():
                self.model(inputs, **kwargs)
        else:
            self.model(inputs)

        for hook in self.hooks:
            hook.remove()

        self.feature_maps = {map: features for (map, features) in list(self.feature_maps.items())[:-1]
                             if features is not None}
        return self.feature_maps

if __name__ == '__main__':
    pass