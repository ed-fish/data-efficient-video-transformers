import cv2
import subprocess
import os
import glob
import shutil
import tempfile


class SpatioCut:

    def convert_framerate(self, video_file, output_file, fps):
        subprocess.call(['ffmpeg', '-i', video_file, '-filter:v', f'fps={fps}',
                        output_file, '-loglevel', 'quiet'])

    def split_video(self, video_file, output_dir):
        subprocess.call(['ffmpeg', '-hwaccel', 'cuda', '-i', video_file,
                         '-c:v', 'libx264', '-crf', '22', '-map', '0',
                         '-segment_time', '1', '-reset_timestamps', '1', '-g',
                         '16', '-sc_threshold', '0', '-force_key_frames',
                         "expr:gte(t, n_forced*16)", '-f', 'segment',
                         os.path.join(output_dir, '%03d.mp4'), '-loglevel',
                         'quiet'])

    def split_frames(self, video_file):
        frame_list = []
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        frame_list.append(image)
        while success:
            success, image = vidcap.read()
            if success:
                frame_list.append(image)
        return frame_list

# Returns a 2d array of [n_chunks x n_frames]
    def cut_vid(self, video_file, frame_rate):
        output = []
        tempdir = tempfile.mkdtemp()
        tempvid = os.path.join(tempdir, "vid.mp4")
        print(tempvid)
        self.convert_framerate(str(video_file), tempvid, frame_rate)
        self.split_video(tempvid, tempdir)
        if os.path.exists(tempvid):
            os.remove(tempvid)
        for vid in glob.glob(tempdir + "/*.mp4"):
            output.append(self.split_frames(vid))
        shutil.rmtree(tempdir)
        return output
