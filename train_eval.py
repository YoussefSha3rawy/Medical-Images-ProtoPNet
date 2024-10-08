import time
import torch
from tqdm import tqdm

from ProtoPNet.helpers import list_of_distances, make_one_hot

device = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'


def _train_or_test(model,
                   dataloader,
                   optimizer=None,
                   class_specific=True,
                   use_l1_mask=True,
                   coefs=None,
                   log=print,
                   wandb_logger=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    all_labels, all_predictions = [], []

    for i, (image, label) in enumerate(tqdm(dataloader)):

        input = image.to(device)
        target = label.to(device)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1] *
                            model.module.prototype_shape[2] *
                            model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, label]).to(device)
                inverted_distances, _ = torch.max(
                    (max_dist - min_distances) * prototypes_of_correct_class,
                    dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(
                    max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(
                        model.module.prototype_class_identity).to(device)
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['clst'] * cluster_cost +
                            coefs['sep'] * separation_cost + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['clst'] * cluster_cost + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_labels.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))

    if wandb_logger:
        wandb_log = {
            "accuracy": n_correct / n_examples,
            "cross_entropy": total_cross_entropy / n_batches
        }
        if is_train:
            wandb_log = {'train_' + k: v for k, v in wandb_log.items()}
        else:
            wandb_log = {'val_' + k: v for k, v in wandb_log.items()}

        wandb_logger.log(wandb_log)
        wandb_logger.log_confusion_matrix(all_labels, all_predictions)

    return n_correct / n_examples


def train(model,
          dataloader,
          optimizer,
          class_specific=False,
          coefs=None,
          log=print,
          wandb_logger=None):
    assert (optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test(model=model,
                          dataloader=dataloader,
                          optimizer=optimizer,
                          class_specific=class_specific,
                          coefs=coefs,
                          log=log,
                          wandb_logger=wandb_logger)


def test(model,
         dataloader,
         class_specific=False,
         log=print,
         wandb_logger=None):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model,
                          dataloader=dataloader,
                          optimizer=None,
                          class_specific=class_specific,
                          log=log,
                          wandb_logger=wandb_logger)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tjoint')
