""" Main GUI"""
from PIL import Image, ImageTk
import screeninfo
from tkinter import filedialog, BOTH, LEFT, VERTICAL, RIGHT, Y, Label, PhotoImage
from PoseEstimation import *


class MainWindow:
    def __init__(self, PATH="", putDetectedLine=True):
        self.PATH = PATH
        self.putDetectedLine = putDetectedLine
        self.poseEstimation = None
        self.personFound = None

        screen_info = screeninfo.get_monitors()[0]
        screen_width = screen_info.width
        screen_height = screen_info.height

        self.win = tk.Tk()
        self.win.geometry(f"{280}x{160}")
        self.win.title("Walking Assessment")

        self.main_frame = tk.Frame(self.win)
        self.main_frame.pack(side=tk.TOP, anchor=tk.CENTER, fill=BOTH, expand=1)

        # canvas
        self.my_canvas = tk.Canvas(self.main_frame, width=280, height=160)
        self.my_canvas.pack(side=tk.TOP, anchor=tk.CENTER, expand=1)

        bg = Image.open("root_bg.png")
        bg = bg.resize((420, 320))
        bg = ImageTk.PhotoImage(bg)
        self.my_canvas.create_image(0, 0, image=bg, anchor='nw', tags='bg')

        self.root = tk.Frame(self.my_canvas, width=50, height=150, bg='gray')
        self.root.grid(column=2, row=0)

        self.speed_txt ="The average speed(both walks) computed is: {}m/s"
        self.speed_label = tk.Label(self.my_canvas, text=self.speed_txt.format(""))
        self.speed_label.grid(column=2, row=1)

        self.start()
        self.win.mainloop()

    def start(self):
        self.ply = PhotoImage(file='ply.png')
        self.ply = self.ply.subsample(5, 5)
        self.paused = PhotoImage(file='pause.png')
        self.paused = self.paused.subsample(5, 5)
        self.pause_button = tk.Button(self.root, image=self.ply, command=self.pause_video, borderwidth=0)
        self.pause_button.grid(column=0, row=0)

        load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        load_button.grid(column=0, row=1)

        undetect_button = tk.Button(self.root, text="Change Line Position", command=self.undetect_line)
        undetect_button.grid(column=0, row=2)

        start_button = tk.Button(self.root, text="Start Video", command=self.start_video)
        start_button.grid(column=0, row=3)

        close_button = tk.Button(self.root, text="Close", command=self.close_window)
        close_button.grid(column=0, row=4)

    def load_video(self):
        self.PATH = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def start_video(self):
        self.poseEstimation = PoseEstimation(self.PATH, mainWindow=self, putDetectedLine=self.putDetectedLine)
        self.poseEstimation.start()

    def pause_video(self):
        if self.poseEstimation.paused:
            self.poseEstimation.paused = False
            self.pause_button.config(image=self.ply)
        else:
            self.poseEstimation.paused = True
            self.pause_button.config(image=self.paused)

    def undetect_line(self):
        if self.PATH == "":
            self.poseEstimation.putDetectedLine = False
        else:
            self.poseEstimation.stop()
            self.poseEstimation = PoseEstimation(self.PATH, mainWindow=self, putDetectedLine=False,
                                                 personFound=self.personFound)
            self.poseEstimation.start()

    def update_speed_label(self, speed):
        updated_text = self.speed_txt.format(speed)
        self.speed_label.config(text=updated_text)

    def close_window(self):
        if self.poseEstimation:
            self.poseEstimation.stop()
        self.win.destroy()
        exit(0)


def startApp(putDetectedLine):
    mainWindow = MainWindow(putDetectedLine=putDetectedLine)


if __name__ == '__main__':
    mainWindow = MainWindow()
