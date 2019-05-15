import torch
from torch import nn

from imagemodels import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(opt):
    assert opt.modelname in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]
    opt.arch = "{}-{}".format(opt.modelname, opt.modeldepth)

    if opt.modelname == 'resnet':
        assert opt.modeldepth in [10, 18, 34, 50, 101, 152, 200]

        from imagemodels.resnet import get_fine_tuning_parameters

        model = getattr(resnet, opt.modelname + str(opt.modeldepth))(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.modelname == 'wideresnet':
        assert opt.modeldepth in [50]

        from imagemodels.wide_resnet import get_fine_tuning_parameters

        if opt.modeldepth == 50:
            model = getattr(wide_resnet, "resnet" + str(opt.modeldepth))(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.modelname == 'resnext':
        assert opt.modeldepth in [50, 101, 152]

        from imagemodels.resnext import get_fine_tuning_parameters

        model = getattr(resnext, "resnet" + str(opt.modeldepth))(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.modelname == 'preresnet':
        assert opt.modeldepth in [18, 34, 50, 101, 152, 200]

        from imagemodels.pre_act_resnet import get_fine_tuning_parameters
        model = getattr(pre_act_resnet, "resnet" + str(opt.modeldepth))(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.modelname == 'densenet':
        assert opt.modeldepth in [121, 169, 201, 264]

        from imagemodels.densenet import get_fine_tuning_parameters
        model = getattr(densenet, opt.modelname + str(opt.modeldepth))(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    if opt.cuda:
        model = model.cuda()

        if opt.pretrain_path:

            state_dict = torch.load(opt.pretrain_path)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            # load params
            assert opt.arch == state_dict['arch']

            model.load_state_dict(new_state_dict)

            if opt.modelname == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
                model.classifier = model.module.classifier.cuda()
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)
                model.fc = model.fc.cuda()

            #parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model
    else:
        if opt.pretrain_path:
            state_dict = torch.load(opt.pretrain_path)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            # load params
            assert opt.arch == state_dict['arch']

            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.modelname == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

            #parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model

    return model, model.parameters()
