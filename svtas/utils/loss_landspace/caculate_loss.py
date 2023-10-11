'''
Author       : Thyssen Wen
Date         : 2023-04-07 15:21:39
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 15:12:27
Description  : file content
FilePath     : /SVTAS/svtas/utils/loss_landspace/caculate_loss.py
'''
import torch
import numpy as np
from svtas.utils import is_h5py_available
if is_h5py_available():
    import h5py
import os


def eval_loss(runner, dataloader, criterion_metric_name='F1@0.50'):
    runner.set_dataloader(dataloader)
    runner.init_engine()
    runner.run()

    Metric_dict = dict()
    for k, v in runner.metric.items():
        temp_Metric_dict = v.accumulate()
        Metric_dict.update(temp_Metric_dict)

    return runner.record['loss'].get_mean, Metric_dict[criterion_metric_name]

def calulate_loss_landscape(model, directions, outpath, logger, runner, dataloader, plot_1D, criterion_metric_name,
                            key='test', xmin=-1, xmax=1, xnum=51, ymin=-1, ymax=1, ynum=51):
    surface_path = os.path.join(outpath, "3d_surface_file.h5")
    setup_surface_file(surface_path, logger, xmin=xmin, xmax=xmax, xnum=xnum, ymin=ymin, ymax=ymax, ynum=ynum, key=key, plot_1D=plot_1D)
    init_weights = [p.data for p in model.parameters()]

    with h5py.File(surface_path, 'r+') as f:

        xcoordinates = f['xcoordinates'][:]
        if not plot_1D:
            ycoordinates = f['ycoordinates'][:]
        else:
            ycoordinates = None

        losses = f[f"{key}_loss"][:]
        accuracies = f[f"{key}_metric"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            logger.info("ind...%s" % ind)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord)

            loss, acc = eval_loss(runner, dataloader, criterion_metric_name)
            logger.info('Loss: %.2f, Metric: %.2f' % (loss, acc))

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            logger.info('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
                ind, len(inds), 100.0 * count / len(inds), str(coord)))

            f[f"{key}_loss"][:] = losses
            f[f"{key}_metric"][:] = accuracies
            f.flush()


def setup_surface_file(surface_path, logger, xmin=-1, xmax=1, xnum=51, ymin=-1, ymax=1, ynum=51, plot_1D=False, key='test'):

    if os.path.isfile(surface_path):
        logger.info("%s is already set up" % "3d_surface_file.h5")
        with h5py.File(surface_path, 'a') as f:
            if f"{key}_loss" in f.keys():
                return

    with h5py.File(surface_path, 'a') as f:
        if f"xcoordinates" not in f.keys():
            xcoordinates = np.linspace(xmin, xmax, xnum)
            f['xcoordinates'] = xcoordinates
        else:
            xcoordinates = f['xcoordinates']

        if not plot_1D:
            if f"ycoordinates" not in f.keys():
                ycoordinates = np.linspace(ymin, ymax, ynum)
                f['ycoordinates'] = ycoordinates
            else:
                ycoordinates = f['xcoordinates']
        else:
            ycoordinates = None


        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)

        f[f"{key}_loss"] = losses
        f[f"{key}_metric"] = accuracies

        return


def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size))
    inds = inds[vals.ravel() <= 0]

    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]
    else:
        return inds, xcoordinates.ravel()[inds]


def overwrite_weights(model, init_weights, directions, step):
    if len(directions) == 2:
        dx = directions[0]
        dy = directions[1]
        changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
    else:
        changes = [d*step for d in directions[0]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.Tensor(d).to(device)