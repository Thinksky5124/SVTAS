'''
Author       : Thyssen Wen
Date         : 2023-09-14 19:46:29
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 15:15:28
Description  : file content
FilePath     : /SVTAS/svtas/utils/loss_landspace/visualize.py
'''
from svtas.utils import is_h5py_available, is_matplotlib_available
if is_matplotlib_available():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
if is_h5py_available():
    import h5py
import numpy as np
import seaborn as sns
import os

# matplotlib reference:
# http://pynote.hatenablog.com/entry/matplotlib-surface-plot
# https://qiita.com/kazetof/items/c0204f197d394458022a

def plot_landspace_2D_loss_err(outpath,
                      logger,
                      criterion_metric_name,
                      vmin = 0,
                      vmax = 100,
                      vlevel = 0.5,
                      surf_name = "test_loss"):
    surface_path = os.path.join(outpath, "3d_surface_file.h5")
    logger.info('------------------------------------------------------------------')
    logger.info('plot_landspace_2D')
    logger.info('------------------------------------------------------------------')
    with h5py.File(surface_path, 'r') as f:

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        X, Y = np.meshgrid(x, y)
        test_loss = np.array(f[surf_name][:])
        test_metric = np.array(f['test_metric'][:])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        #ax.plot_wireframe(X, Y, Z)
        ax.plot_surface(X, Y, test_loss, linewidth=0, antialiased=False)
        plt.show()

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(X, Y, test_loss, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(os.path.join(outpath, surf_name + '_2dcontour' + '.png'), dpi=300,
                    bbox_inches='tight')
        
        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(X, Y, 100 - test_metric, cmap='summer', levels=np.arange(0, 100, 5))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(os.path.join(outpath, 'test_metric_2dcontour' + '.png'), dpi=300,
                    bbox_inches='tight')

        fig = plt.figure()
        CS = plt.contourf(X, Y, test_loss, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(os.path.join(outpath, surf_name + '_2dcontourf' + '.png'), dpi=300,
                    bbox_inches='tight')

        # Save 2D heatmaps image
        plt.figure()
        sns_plot = sns.heatmap(test_loss, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                               xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(os.path.join(outpath, surf_name + '_2dheat' + '.png'),
                                      dpi=300, bbox_inches='tight')

        # Save 3D surface image
        # loss and accuracy map
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        plt.xlim(-1, 1)
        ax.set_zlabel('Loss', fontsize=1)
        ax.set_xlabel('X', fontsize=1)
        ax.set_ylabel('Y', fontsize=1)
        ax.set_ylim(-1, 1)
        plt.xticks([])
        plt.yticks([])

        ax.plot_surface(X, Y, test_loss, label='Testing loss', linewidth=0, antialiased=True, cmap='rainbow')
        ax.view_init(elev=20, azim=-60)
        fig.savefig(os.path.join(outpath, surf_name + '_3dsurface' + '.png'), dpi=300,
                    bbox_inches='tight')


def plot_contour_trajectory(outpath, logger, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""
    surface_path = os.path.join(outpath, "3d_surface_file.h5")
    logger.info('------------------------------------------------------------------')
    logger.info('plot_contour_trajectory')
    logger.info('------------------------------------------------------------------')
    # plot contours
    with h5py.File(surface_path, 'r') as f:

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)
        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])

        fig = plt.figure()
        CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
        CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

        # plot trajectories
        plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')

        # plot red points when learning rate decays
        # for e in [150, 225, 275]:
        #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

        # add PCA notes
        ratio_x = f['explained_variance_ratio_'][0]
        ratio_y = f['explained_variance_ratio_'][1]
        plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
        plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f.close()
        plt.clabel(CS1, inline=1, fontsize=6)
        plt.clabel(CS2, inline=1, fontsize=6)
        fig.savefig(os.path.join(outpath, '2dcontour_trajectory' + '.png'), dpi=300,
                    bbox_inches='tight')
        if show: plt.show()

def plot_landspace_1D_loss_err(outpath, logger, criterion_metric_name, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False):
    logger.info('------------------------------------------------------------------')
    logger.info('plot_landspace_1D_loss_err')
    logger.info('------------------------------------------------------------------')

    surface_path = os.path.join(outpath, "3d_surface_file.h5")
    with h5py.File(surface_path, 'r') as f:
        x = f['xcoordinates'][:]
        assert 'train_loss' in f.keys(), "'train_loss' does not exist"
        train_loss = f['train_loss'][:]
        train_metric = f['train_metric'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        # loss and accuracy map
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        if log:
            tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
        else:
            tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        tr_metric, = ax2.plot(x, train_metric, 'r-', label='Training metric', linewidth=1)

        if 'test_loss' in f.keys():
            test_loss = f['test_loss'][:]
            test_acc = f['test_metric'][:]
            if log:
                te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
            else:
                te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
            te_metric, = ax2.plot(x, test_acc, 'r--', label='Test metric', linewidth=1)

        plt.xlim(xmin, xmax)
        ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
        ax1.tick_params('y', colors='b', labelsize='x-large')
        ax1.tick_params('x', labelsize='x-large')
        ax1.set_ylim(0, loss_max)
        ax2.set_ylabel(criterion_metric_name, color='r', fontsize='xx-large')
        ax2.tick_params('y', colors='r', labelsize='x-large')
        ax2.set_ylim(0, 100)
        plt.savefig(os.path.join(outpath, '1d_loss_metric' + ('_log' if log else '') + '.png'),
                    dpi=300, bbox_inches='tight')

        # train_loss curve
        plt.figure()
        if log:
            plt.semilogy(x, train_loss)
        else:
            plt.plot(x, train_loss)
        plt.ylabel('Training Loss', fontsize='xx-large')
        plt.xlim(xmin, xmax)
        plt.ylim(0, loss_max)
        plt.savefig(os.path.join(outpath, '1d_train_loss' + ('_log' if log else '') + '.png'),
                    dpi=300, bbox_inches='tight')

        # train_err curve
        plt.figure()
        plt.plot(x, 100 - train_metric)
        plt.xlim(xmin, xmax)
        plt.ylim(0, 100)
        plt.ylabel('Training Error', fontsize='xx-large')
        plt.savefig(os.path.join(outpath, '1d_train_err' + '.png'), dpi=300, bbox_inches='tight')

        if show: plt.show()
        f.close()