#
# Utility function for creating nets
#

import sys

from models.conv_text_classifier import ConvTextClassifier
from models.deepbind import DeepBind
from models.deepsea import DeepSea
from models.linear_model import LinearModel
from models.lstm_language_model import LSTMLanguageModel
from models.lstm_text_classifier import LSTMTextClassifier
from models.simple_convnet import SimpleConvnet
from models.wide_resnet import WideResnet
from models.lenet import LeNet

def create_net(task_config, projector_config):

    model = task_config['model']
    output_size = task_config['output_size']
    context_size = projector_config['context_size']
    block_in = projector_config['block_in']
    block_out = projector_config['block_out']

    print("Creating", model)

    if context_size > 0:
        hyper = True
        hyperlayers = ['conv2']
    else:
        hyper = False
        hyperlayers = []

    if model == 'conv_text_classifier':
        input_dim = task_config.get('vocab_size', 5000) + 2
        embedding_dim = task_config.get('embedding_dim', 32)
        num_filters = task_config.get('num_filters', 64)
        hidden_dim = task_config.get('hidden_dim', 64)
        net = ConvTextClassifier(context_size, block_in, block_out, input_dim,
              embedding_dim, num_filters, hidden_dim, output_dim=output_size, hyper=hyper)

    elif model == 'deepbind':
        num_filters = task_config.get('num_filters', 16)
        hidden_dim = task_config.get('hidden_dim', 32)
        net = DeepBind(context_size, block_in, block_out, {'context_size': 100}, hyper,
                        filters=num_filters, hidden_units=hidden_dim)

    elif model == 'deepsea':
        net = DeepSea(context_size, block_in, block_out, {'context_size': 100}, hyper)

    elif model == 'linear_model':
        input_size = task_config.get('input_dim', 20)
        net = LinearModel(context_size, block_in, block_out,
                          input_dim=input_size, output_dim=output_size, hyper=hyper)

    elif model == 'lstm_text_classifier':
        input_dim = task_config.get('vocab_size', 25000) + 2
        embedding_dim = task_config.get('embedding_dim', 128)
        hidden_dim = task_config.get('hidden_dim', 256)
        net = LSTMTextClassifier(context_size, block_in, block_out, input_dim,
                                 embedding_dim, hidden_dim, output_size, hyper=hyper)

    elif model == 'lstm_language_model':
        layer_size = task_config.get('layer_size', 32)
        net = LSTMLanguageModel(context_size, block_in, block_out,
                                ninp=layer_size, nhid=layer_size, hyper=hyper)

    elif model == 'simple_convnet':
        net = SimpleConvnet(context_size, block_in, block_out, hyperlayers=hyperlayers)

    elif model == 'wide_resnet':
        N = task_config.get('N', 6)
        k = task_config.get('k', 1)
        num_classes = output_size
        net = WideResnet(context_size, block_in, block_out, N, k, num_classes, hyper)

    elif model == 'lenet':
        if context_size > 0:
            hyperlayers = ['conv2', 'fc1', 'fc2']
        net = LeNet(context_size, block_in, block_out, hyperlayers)

    else:
        print("Please select a valid model kind")
        sys.exit(0)

    return net

