import matplotlib
import matplotlib.pyplot as plt

yValue = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yValue2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
attack_type = 'no_attack'

plt.plot(yValue, 'b' if attack_type == 'no_attack' else 'r', linewidth=2)
plt.axis([0, 200, 0, 100])
plt.xlabel(xlabel='epochs', fontsize=15)
plt.ylabel(ylabel='accuracy', fontsize=15)
plt.tick_params(axis='both', labelsize=12, color='red', labelcolor='green')
plt.show()

def score_mixed_updates(mixed_updates, gmodel, exchange_list):
    global list
    local_models = []
    for k, mixed_update in mixed_updates.items():
        model = copy.deepcopy(gmodel)
        model.load_state_dict(mixed_update)
        local_models.append(model)

    sb = list(local_models[0].parameters())[-1]
    sw = list(local_models[0].parameters())[-2]
    try:
        sl = sb.shape[0] + sw.shape[0] * sw.shape[1]
    except:
        sl = sb.shape[0] + sw.shape[0]

    gmodel = torch.nn.utils.parameters_to_vector(gmodel.parameters())
    models = [torch.nn.utils.parameters_to_vector(model.parameters()) for model in local_models]
    n = len(local_models)
    all_grads = torch.empty([n, len(models[0])])
    for i in range(n):
        all_grads[i] = (gmodel - models[i]).detach()

    index_dic = list(mixed_updates.keys())
    T = len(all_grads)
    re_all_grads = torch.zeros_like(all_grads)
    m = dict()
    s = 2
    for list in exchange_list:
        m[list[0]] = list[1]
        m[list[1]] = list[0]
    for t in range(T):
        var_tensor = torch.zeros_like(all_grads[0], requires_grad=False)
        var_tensor.add_(1 / s, all_grads[t])
        for i, j in enumerate(index_dic):
            if j == m[index_dic[t]]:
                var_tensor.add_(1 / s, all_grads[i])
                break
        re_all_grads[t].copy_(var_tensor)

    all_grads = re_all_grads

    d = score_all(all_grads)
    cs = score_last(all_grads, sl)
    d = d / d.max()
    d = 1 - d
    cs = (cs + 1) / 2
    sim = 0.2 * d + 0.8 * cs
    # sim = d
    q1 = torch.quantile(sim, 0.25)
    sim = sim - q1
    peers = list(mixed_updates.keys())
    sim_dict = {}
    for i, k in enumerate(peers):
        sim_dict[k] = sim[i].item()

    return sim_dict, exchange_list