import argparse 
from re import template
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
    template_layers = [i for j, i in enumerate(template_layers) if j in args.template_layers]
    
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
        label: int = args.label

        start_time:time = time()

        # Implement the idea from last week.
        # Algorithm Procedure:
        # 1. propagate the inputs into first activation layer.
        # 2. order the templates in the first layer by their area.
        # 3. propagate the biggest template into the next layer.
        # 4. if the output is verified, then remove the biggest template and all the templates are inluded in this template.
        # 5. select the biggest template from the rest templates and repeat to step 3 until all the templates are checked.
        verified_inputs: list = []
        all_t = templates.OnlineTemplates(net=networks, layers=template_layers, label=args.label, domain=args.template_domain, relu_transformer=args.relu_transformer)
        relaxation_list: list = []
        filtered_inputs: list = []
        mp: dict = {}
        # step 1. propagate the inputs into the first specified activation layer
        for id, (inputs, labels) in enumerate(data_loader):
            if not torch.argmax(networks(inputs), 1) == labels:
                # * filter out misclassified inputs or have been verified inputs
                continue
            filtered_inputs.append((id, inputs))
            count_true += 1

            # propagation 
            z_net = Zonotope_Net(net=networks, relu_transformer=args.relu_transformer)
            z_net.initialize(inputs=inputs, eps=args.epsilon)
            for j in range(template_layers[0]+1):
                z_net.apply_layer(j)
            z = z_net.relaxation_at_layers[-1]
            relaxation_list.append((id, z_net))
         
        for layer_id in range(template_layers[0], len(networks.layers)):
            # step 2.
            for id1, z_net1 in relaxation_list:
                z1 = z_net1.relaxation_at_layers[-1]
                for id2, z_net2 in relaxation_list:
                    if id1 == id2: continue
                    z2 = z_net2.relaxation_at_layers[-1]
                    if z1.to_box().submatching(z2):
                        if id1 not in mp.keys(): mp[id1] = [id2]
                        else: mp[id1].append(id2)
        print(mp)



        # for layer_id, layer in enumerate(networks.layers):
        #     if layer_id not in template_layers:
        #         continue
        #     for id, (inputs, labels) in enumerate(data_loader):
        #         if not torch.argmax(networks(inputs), 1) == labels or id in verified_inputs:
        #             # * filter out misclassified inputs or have been verified inputs
        #             continue
        #         if layer_id == template_layers[0]: count_true += 1

        #         # verify the input at the first activation layer
        #         z_net = Zonotope_Net(net=networks, relu_transformer=args.relu_transformer)
        #         z_net.initialize(inputs=inputs, eps=args.epsilon)
        #         for j in range(template_layers[len(template_layers)-1]+1):
        #             z_net.apply_layer(j)
        #         z = z_net.relaxation_at_layers[-1]
        #         # isIncluded: bool = False
        #         # for k, v in mp.items():
        #         #     if k.to_box().submatching(z):
        #         #         mp[k].append((id, z_net))
        #         #         isIncluded = True
        #         #         break
        #         # if not isIncluded: mp[z] = [(id, z_net)]
        #         relaxation_list.append((id, z_net))
            
        #     # group the submatched relaxation inputs
        #     # build the graph based on the relaxation information
        #     for id1, z_net1 in relaxation_list:
        #         z1 = z_net1.relaxation_at_layers[-1]
        #         for id2, z_net2 in relaxation_list:
        #             if id1 == id2: continue
        #             z2 = z_net2.relaxation_at_layers[-1]
        #             if z1.to_box().submatching(z2):
        #                 print()
                
        #     mp = {k: v for k, v in sorted(mp.items(), key=lambda item: len(item))}
        #     for k, v in mp.items():
        #         if layer_id in t_offline.layers:
        #             if t_offline.submatching(k, layer_id):
        #                 count_verified += len(v)
        #                 count_submatched += len(v)
        #                 verified_inputs.extend(v)
        #             elif layer_id == t_offline.layers[-1]:
        #                 for id, z_net in v:
        #                     if z_net.calculate_worst_case(label):
        #                         count_verified += 1
        #                         verified_inputs.append(id)
        #     mp.clear()
                    
        
        # for layer_id in range(template_layers[0]+1, networks.layers):
        #     if layer_id not in template_layers:
        #         continue
        #     for k, v in mp.items():
        #         z_net = Zonotope_Net(net=networks, relu_transformer=args.relu_transformer)
        #         z_net.initialize(inputs=data_loader[v], eps=args.epsilon)
        #         for j in range(layer_id+1):
        #             z_net.apply_layer(j)
        #         z = z_net.relaxation_at_layers[-1]
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