
def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device, n_atoms, delta_z, vmin, vmax):
    '''Projects one parameterized distribution onto another given the target distribution number of intervals
        min/max values and detla (interval size)
    
    Arguments:
        next_distr_v {Tensor} -- distro to project
        rewards_v {Tensor} -- rewards to use for bellman equation
        dones_mask_t {Tensor} -- completed trajectory mask (array of True's set at indices of completed trajectories)
        gamma {float} -- discount
        device {string} -- cuda or cpu
        n_atoms {int} -- numper of intervals
        delta_z {float} -- interval size: (Vmax - Vmin) / (n_atoms - 1)
        vmin {float} -- min interval
        vmax {float} -- max interval
    
    Returns:
        FloatTensor -- projected distribution
    '''

    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)

    for atom in range(n_atoms):
        tz_j = np.minimum(vmax, np.maximum(vmin, rewards + (vmin + atom * delta_z) * gamma))
        b_j = (tz_j - vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(vmax, np.maximum(vmin, rewards[dones_mask]))
        b_j = (tz_j - vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)
