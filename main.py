import argparse 
from time import time

import torch

import config 
import utils
import templates
from relaxations import Zonotope_Net


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_net(netname, dataset, logger):
    path = f"examples/{dataset}_nets/{netname}"
    file_format = netname.split('.')[1]
    if file_format in ["pth"]:
        return utils.load_net_from_patch_attacks(path)
    elif file_format in ["tf", "pyt"]:
        return utils.load_net_from_eran_examples(path)
    else:
        logger.error(f"Unknow file ending: {file_format}")
        return RuntimeError(f"Unknow file ending: {file_format}")


if __name__ == "__main__":
    config.init_logger()
    logger = config.logger
    conf = config.config

    # * Parse arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--l_infinity", action="store_true", help="Verify against l_infinity perturbations")
    group.add_argument("-p", "--patches", action="store_true", help="Verify against patch perturbations")
    group.add_argument("-pt", "--patches_timing", action="store_true", help="same as -p with addtional timing/lggin")
    group.add_argument("-g", "--geometric", action="store_true", help="Verify against geometric perturbations")
    
    parser.add_argument("--netname", type=str, default=conf.netname[0], help=conf.netname[1])
    parser.add_argument("--num_tests", type=int, default=conf.num_tests[0], help=conf.num_tests[1])
    parser.add_argument("--relu_transformer", type=str, default=conf.relu_transformer[0], help=conf.relu_transformer[1])
    parser.add_argument("--dataset", type=str, default=conf.dataset[0], help=conf.dataset[1])
    parser.add_argument("--epsilon", type=float, default=conf.epsilon[0], help=conf.epsilon[1])
    parser.add_argument("--label", type=int, default=conf.label[0], help=conf.label[1])
    parser.add_argument("--patch_size", type=int, default=conf.patch_size[0], help=conf.patch_size[1])
    parser.add_argument("--data_dir", type=str, default=conf.data_dir[0], help=conf.data_dir[1])
    parser.add_argument("--template_method", type=str, default=conf.template_method[0], help=conf.template_method[1])
    parser.add_argument("--template_layers", type=int, nargs="+", default=conf.template_layers[0], help=conf.template_layers[1])
    parser.add_argument("--template_domain", type=str, default=conf.template_domain[0], help=conf.template_domain[1])
    parser.add_argument("--template_with_hyperplanes", type=str2bool, default=conf.template_with_hyperplanes[0], help=conf.template_with_hyperplanes[1])
    parser.add_argument("--template_dir", type=str, nargs="+", default=conf.template_dir[0], help=conf.template_dir[1])
    parser.add_argument("--num_templates", type=int, default=conf.num_templates[0], help=conf.num_templates[1])
    parser.add_argument("--template_max_eps", type=str2bool, default=False, help="maximize epsilon when creating l-infinity templates")

    # * load arguments and network model
    args = parser.parse_args()
    networks = load_net(netname=args.netname, dataset=args.dataset, logger=logger)

    template_layers = [i for i, l in enumerate(networks.layers) if isinstance(l, torch.nn.ReLU)]
    # template_layers = [i for j, i in enumerate(template_layers) if j in args.template_layers]
    
    # * Verify against l_infinity perturbations
    if args.l_infinity:
        t_offline = templates.OfflineTemplates(net=networks, layers=template_layers, label=args.label, domain=args.template_domain, relu_transformer=args.relu_transformer)
        path_to_network: str = f"examples/{args.dataset}_nets/{args.netname}"
        if args.template_dir is None:
            train_dataset = utils.load_dataset_selected_labels_only(dataset_name=args.dataset, labels=[args.label], test_set=False)
            t_offline.create_templates(dataset=train_dataset, epsilon=args.epsilon, path_to_net=path_to_network, use_hyperplanes=args.template_with_hyperplanes, num_templates=args.num_templates, max_epsilon=args.template_max_eps)
        else:
            t_offline.load_templates(path_to_net=path_to_network, filenames=args.template_dir, num_templates=args.num_templates)
        
        test_dataset = utils.load_dataset_selected_labels_only(dataset_name=args.dataset, labels=[args.label], num_elements=args.num_tests, test_set=True)
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        count_total: int = test_dataset.targets.shape[0]
        count_true: int = 0
        count_verified: int = 0
        count_verified_by_all: int = 0
        count_submatched: int = 0

        start_time:time = time()
        # Algorithm 1: dynamic template generation
        # assume that we don't have any offline template files on hand, we need to generate them before starting the verification process.
        # idea: based on the input which is verified, store its information from each layer to be the template.
        verified_inputs: list = []
        all_t = templates.OnlineTemplates(net=networks, layers=template_layers, label=args.label, domain=args.template_domain, relu_transformer=args.relu_transformer)
        for layer_id, layer in enumerate(networks.layers):
            if layer_id not in template_layers:
                continue
            for id, (inputs, labels) in enumerate(data_loader):
                label = labels.item()
                if not torch.argmax(networks(inputs), 1) == labels or id in verified_inputs:
                    # * filter out misclassified inputs or have been verified inputs
                    continue
                if layer_id == template_layers[0]: count_true += 1

                z_net = Zonotope_Net(net=networks, relu_transformer=args.relu_transformer)
                z_net.initialize(inputs=inputs, eps=args.epsilon)
                for i in range(layer_id+1):
                    z_net.apply_layer(i)

                # isSubmatch: bool = False
                z = z_net.relaxation_at_layers[-1]
                # isSubmatch = t_offline.submatching(z, layer_id) # or all_t.submatching(z, layer_id)
                if all_t.submatching(z, layer_id):
                    count_verified_by_all += 1
                    count_submatched += 1
                    verified_inputs.append(id)
                    continue
                # if t_offline.submatching(z, layer_id):
                #     count_verified += 1
                #     count_submatched += 1
                #     verified_inputs.append(id)
                #     continue
                elif layer_id == template_layers[-1]:
                    isVerified = z_net.process_from_layer(true_label=label, start_layer=layer_id)
                    if isVerified:
                        count_verified += 1
                        verified_inputs.append(id)

                        # update stored templates from scratch
                        # the template can be generated dynamically for reach input data.
                        zz_net = Zonotope_Net(net=networks, relu_transformer=args.relu_transformer)
                        zz_net.initialize(inputs=inputs, eps=args.epsilon)
                        for j in range(len(networks.layers)):
                            zz_net.apply_layer(j)
                            if j in template_layers:
                                zz = zz_net.relaxation_at_layers[-1]
                                if args.template_domain == "box":
                                    zz = zz.to_box()
                                elif args.template_domain == "parallelotope":
                                    zz = zz.to_parallelotope()
                                
                                # make sure its relaxation still valid.
                                isVerified: bool = all_t._shrinking_two(zz, j)
                                if isVerified:
                                    all_t.templates[j].append(zz)

        print(verified_inputs)
        logger.info(f"Net: {args.netname}, Dataset: {args.dataset}, Label: {args.label}")
        logger.info(f"Images Submatched/Verified/Verified2/Predicted/Total: {count_submatched}/{count_verified}/{count_verified_by_all}/{count_true}/{count_total}")
        logger.info("Time spent: {:.3f}".format(time() - start_time))

    elif args.geometric:
        #TODO: Implement geometric verification
        print()
    elif args.patches:
        #TODO: Implement patch verification
        print()
    elif args.patches_timing:
        #TODO: Implement patch verification with timing
        print()
    
    
    elif args.patches:
        #TODO: Implement patch verification
        print()
    elif args.patches_timing:
        #TODO: Implement patch verification with timing
        print()
    elif args.geometric:
        #TODO: Implement geometric verification
        print()