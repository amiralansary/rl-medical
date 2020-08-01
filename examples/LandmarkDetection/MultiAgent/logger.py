import os
from datetime import datetime
import socket
import torch
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
import csv


class Logger(object):
    def __init__(self, directory, write, save_freq=10):
        self.parent_dir = directory
        self.write = write
        self.dir = ""
        self.fig_index = 0
        self.model_index = 0
        self.save_freq = save_freq
        if self.write:
            self.boardWriter = SummaryWriter()
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.dir = os.path.join(
                self.parent_dir,
                current_time +
                '_' +
                socket.gethostname())
            self.log(f"Logs from {self.dir}\n{' '.join(sys.argv)}\n")

    def write_to_board(self, name, scalars, index=0):
        self.log(f"{name} at {index}: {str(scalars)}")
        if self.write:
            for key, value in scalars.items():
                self.boardWriter.add_scalar(f"{name}/{key}", value, index)

    def plot_res(self, losses, distances):
        if len(losses) == 0 or not self.write:
            return
        fig, axs = plt.subplots(2)
        axs[0].plot(list(range(len(losses))), losses, color='orange')
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training")
        axs[0].set_yscale('log')
        for dist in distances:
            axs[1].plot(list(range(len(dist))), dist)
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Distance change")
        axs[1].set_title("Training")

        if self.fig_index > 0:
            os.remove(os.path.join(self.dir, f"res{self.fig_index-1}.png"))
        fig.savefig(os.path.join(self.dir, f"res{self.fig_index}.png"))
        self.boardWriter.add_figure(f"res{self.fig_index}", fig)
        self.fig_index += 1

    def log(self, message, step=0):
        print(str(message))
        if self.write:
            # self.boardWriter.add_text("log", str(message), step)
            with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
                logs.write(str(message) + "\n")

    def save_model(self, state_dict, name="dqn.pt", forced=False):
        if not self.write:
            return
        if (forced or
           (self.model_index > 0 and self.model_index % self.save_freq == 0)):
            torch.save(state_dict, os.path.join(self.dir, name))

    def write_locations(self, row):
        self.log(str(row))
        if self.write:
            with open(os.path.join(self.dir, 'results.csv'),
                      mode='a', newline='') as f:
                res_writer = csv.writer(f)
                res_writer.writerow(row)
