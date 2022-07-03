import os
import glob
import time
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from textwrap import wrap


class Artist:
    def __init__(self, n_plot=1, save_path=None):
        # initialize plot
        self.n_plot = n_plot
        self.last_draw_time = time.time()
        self.fig = plt.figure()
        self.fig.set_size_inches(n_plot * 5, 5)
        self.axes = [self.fig.add_subplot(1, self.n_plot, idx + 1, projection='3d') for idx in range(self.n_plot)]
        self.save_path = save_path
        plt.tight_layout()

    def __del__(self):
        plt.close()

    def clear_figures(self):
        # remove all result figures
        files = glob.glob(os.path.join(self.save_path, '*.png'))
        for f in files:
            os.remove(f)

    def save_video(self, data_name):
        # remove all result figures
        files = glob.glob(os.path.join(self.save_path, '*.png'))
        frame = cv2.imread(files[0])
        height, width, _ = frame.shape
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        video = cv2.VideoWriter(os.path.join(self.save_path, '%s.avi' % data_name),
                                fourcc=fcc, fps=10, frameSize=(width, height))
        for file in files:
            video.write(cv2.imread(file))
        video.release()

    def update(self, data_name, titles, features, results, frame_info, fps=1000, b_save=False, b_show=True):
        if b_save:
            os.makedirs(self.save_path, exist_ok=True)
        self.fig.suptitle(data_name, fontsize=23)
        wait_time = 1. / fps
        while True:
            if time.time() - self.last_draw_time > wait_time:
                self.last_draw_time = time.time()

                ret_artists = list()
                for idx in range(self.n_plot):
                    self.init_axis(self.axes[idx], titles[idx])
                    pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = features[idx]
                    ret_artists.extend(self.draw_parts(self.axes[idx], [pelvis, neck, head]))
                    ret_artists.extend(self.draw_parts(self.axes[idx], [neck, lshoulder, lelbow, lwrist]))
                    ret_artists.extend(self.draw_parts(self.axes[idx], [neck, rshoulder, relbow, rwrist]))

                    result = "\n".join(wrap(results[idx], 15)) if results[idx] is not None else ''
                    ret_artists.append(self.axes[idx].text(0, 0, 0, f"{result}\n{frame_info[idx]}", fontsize=20))

                if b_show:
                    plt.show(block=False)
                    plt.pause(0.001)

                frame_data = frame_info[0].split('/')
                if b_save:
                    plt.savefig(os.path.join(self.save_path, "p%03d" % int(frame_data[0])))
                return

    def init_axis(self, ax, title):
        ax.clear()
        ax.set_title(title, y=0.8, fontsize=20)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        max = 0.5
        ax.set_xlim3d(-max * 0.7, max * 0.7)
        ax.set_ylim3d(0, 1.5 * max)
        ax.set_zlim3d(-max, max)

        ax.view_init(elev=-80, azim=90)
        ax._axis3don = False

    def draw_parts(self, ax, joints):
        def add_points(points):
            xs, ys, zs = list(), list(), list()
            for point in points:
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
            return xs, ys, zs

        xs, ys, zs = add_points(joints)
        ret = ax.plot(xs, ys, zs, color='b')
        return ret
