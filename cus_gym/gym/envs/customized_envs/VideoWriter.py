from matplotlib.animation import FFMpegWriter

class VideoWriter:
    def __init__(self, output_rate, fps):
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        self.video = FFMpegWriter(fps=fps, metadata=metadata)
        self.output_rate = output_rate

    def update(self):
        self.video.grab_frame()
    
    def close(self):
        self.video.close()
