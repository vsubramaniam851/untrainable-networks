import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .rep_sims import CKA, DifferentiableRSA
from .reduction import random_projection
from .layer_extract import FeatureMapExtractor

def torchvision_fe(model, inputs, device):
    layers = torchvision.models.feature_extraction.get_graph_node_names(model)[0]
    extract_layers = [l for l in layers if ('mlp' in l) or ('self_attention' in l)] + ['getitem_5']
    feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(model, return_nodes = extract_layers)
    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        output = feature_extractor(inputs)
    return output

def layer_supervision(target_model_layers, student_model_layers):
    source_count = len(target_model_layers)
    target_count = len(student_model_layers)
    step = (target_count - 1) / (source_count - 1) if source_count > 1 else 1

    mapping = {}
    for i, source_layer in enumerate(target_model_layers):
        target_index = min(round(i * step), target_count - 1)
        mapping[source_layer] = student_model_layers[target_index]
    return mapping

def tokenwise_sim(training_output, pretraining_output, cka):
    output = torch.stack([(1 - cka.linear_CKA(training_output[:, i, :], pretraining_output[:, i, :])) for i in range(training_output.shape[1])])
    return output.sum()

def get_layer_outputs(model, inputs, eval = True, **kwargs):
    extractor = FeatureMapExtractor(model, eval = eval)
    if kwargs['lengths'] != None:
        feature_maps = extractor.get_feature_maps(inputs, **kwargs)
    else:
        feature_maps = extractor.get_feature_maps(inputs)
    return feature_maps

def layermap_sim(train_model, target_model, student_model, rep_sim, inputs, target_inputs, device, lengths = None, torchvision_extract = False, 
                 token_sim = False):
    cka = CKA(device)
    diff_rsa = DifferentiableRSA(device)
    if not torchvision_extract:
        pretrained_outputs = get_layer_outputs(target_model, target_inputs, eval = True, lengths = lengths)
    else:
        pretrained_outputs = torchvision_fe(target_model, inputs, device)
    training_outputs = get_layer_outputs(train_model, inputs, eval = False, lengths = lengths)
    teacher_layers = list(pretrained_outputs.keys())
    student_layers = list(training_outputs.keys())
    if len(teacher_layers) <= len(student_layers):
        model_mapping = layer_supervision(teacher_layers, student_layers)
    else:
        #NOTE: I am trying to add multiple levels of supervision in this case. If this works better, I'll keep it.
        #Otherwise, I'll switch back
        model_mapping = layer_supervision(teacher_layers, student_layers)
        # model_mapping = {v : k for k, v in model_mapping.items()}
    sim_scores = {}
    for layer in model_mapping:
        assert layer in pretrained_outputs, f'Layer {layer} is not in target network {pretrained_outputs.keys()}'
        tr_layer = model_mapping[layer]
        assert tr_layer in training_outputs, f'Layer {layer} is not in {student_model} {training_outputs.keys()}'

        pretrained_output = pretrained_outputs[layer]
        training_output = training_outputs[tr_layer]

        assert pretrained_output.shape[0] == training_output.shape[0]

        if rep_sim == 'CKA':
            if not token_sim:
                pretrained_output = pretrained_output.contiguous().view(inputs.size(0), -1)
                training_output = training_output.contiguous().view(inputs.size(0), -1)
                sim = 1 - cka.linear_CKA(training_output.to(torch.float32), pretrained_output.to(torch.float32))
            else:
                sim = tokenwise_sim(training_output, pretrained_output, cka)
        elif rep_sim == 'RSA':
            pretrained_output = pretrained_output.contiguous().view(inputs.size(0), -1)
            training_output = training_output.contiguous().view(inputs.size(0), -1)
            sim = 1 - diff_rsa.rsa(training_output.to(torch.float32), pretrained_output.to(torch.float32))
        else:
            raise NotImplementedError()
        sim_scores[tr_layer] = sim
    del pretrained_outputs
    del training_outputs
    return sim_scores