from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re


def inn_model(img_dims=[1, 28, 28]):
    """
    Return INN model for MNIST.

    :param img_dims: size of the model input images. Default: Size of MNIST images
    :return: INN model
    """

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                        'F_args': {'channels_hidden': 128}, 'clamp': 1}, name='conv1')

    conv2 = fr.Node([conv1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                           'F_args': {'channels_hidden': 128}, 'clamp': 1},
                    name='conv2')

    conv3 = fr.Node([conv2.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                           'F_args': {'channels_hidden': 128}, 'clamp': 1},
                    name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (img_dims[0] * img_dims[1] * img_dims[2],)}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer,
                 {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 1000}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    model = fr.ReversibleGraphNet(nodes, 0, 1)

    return model


def artset_inn_model(img_dims=[3, 224, 224]):
    """
    Return INN model for Painter by Numbers artset.

    :param img_dims: size of the model input images. Default: Size of MNIST images
    :return: INN model
    """

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.reshape_layer, {'target_dim': (img_dims[0] * 4, img_dims[1] // 2, img_dims[2] // 2)}, name='r1')

    conv11 = fr.Node([r1.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 256}, 'clamp': 1}, name='conv11')

    conv12 = fr.Node([conv11.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 256}, 'clamp': 1}, name='conv12')

    conv13 = fr.Node([conv12.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 256}, 'clamp': 1}, name='conv13')


    conv21 = fr.Node([conv13.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1}, name='conv21')

    conv22 = fr.Node([conv21.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1}, name='conv22')

    conv23 = fr.Node([conv22.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1}, name='conv23')


    conv31 = fr.Node([conv23.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 64}, 'clamp': 1}, name='conv31')

    conv32 = fr.Node([conv31.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 64}, 'clamp': 1}, name='conv32')

    conv33 = fr.Node([conv32.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 64}, 'clamp': 1}, name='conv33')


    conv41 = fr.Node([conv33.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 32}, 'clamp': 1}, name='conv41')

    conv42 = fr.Node([conv41.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 32}, 'clamp': 1}, name='conv42')

    conv43 = fr.Node([conv42.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 32}, 'clamp': 1}, name='conv43')


    r2 = fr.Node([conv43.out0], re.reshape_layer, {'target_dim': (img_dims[0] * img_dims[1] * img_dims[2],)}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer,
                 {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 128}, 'clamp': 1}, name='fc')


    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (img_dims[0], img_dims[1], img_dims[2])}, name='r3')
    

    outp = fr.OutputNode([r3.out0], name='output')

    nodes = [inp, outp, conv11, conv12, conv13, conv21, conv22, conv23, conv31, conv32, conv33, conv41, conv42, conv43,
             fc, r1, r2, r3]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder
