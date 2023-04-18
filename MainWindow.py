from PIL import Image, ImageTk
import screeninfo
from tkinter import filedialog, BOTH, LEFT, VERTICAL, RIGHT, Y, Label, PhotoImage
from PoseEstimation import *


class MainWindow:
    def __init__(self, PATH=""):
        self.PATH = PATH
        self.poseEstimation = None

        screen_info = screeninfo.get_monitors()[0]
        screen_width = screen_info.width
        screen_height = screen_info.height

        self.win = tk.Tk()
        self.win.geometry(f"{screen_width}x{screen_height}")
        self.win.title("Walking Assessment")
        # print(screen_width, "  ", screen_height)

        self.main_frame = tk.Frame(self.win)
        self.main_frame.pack(fill=BOTH, expand=1)

        # canvas
        self.my_canvas = tk.Canvas(self.main_frame, width=1350, height=720)
        self.my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        # scrollbar
        self.my_scrollbar = tk.Scrollbar(self.main_frame, orient=VERTICAL, command=self.my_canvas.yview)
        self.my_scrollbar.pack(side=RIGHT, fill=Y)

        # configure the canvas
        self.my_canvas.configure(yscrollcommand=self.my_scrollbar.set)
        self.my_canvas.bind(
            '<Configure>', lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))
        )

        bg = Image.open("root_bg.png")
        bg = bg.resize((1350, 720))
        bg = ImageTk.PhotoImage(bg)
        self.my_canvas.create_image(0, 0, image=bg, anchor='nw', tags='bg')

        self.container_frame = tk.Frame(self.my_canvas, bg='gray')
        self.container_frame.pack()

        self.root = tk.Frame(self.container_frame, width=1350, height=650, bg='gray')
        self.root.grid(row=0, column=0)

        self.buttons_root_frame = tk.Frame(self.container_frame, width=1350, height=100, bg='gray')
        self.buttons_root_frame.grid(row=1, column=0)

        self.start()

        self.my_canvas.create_window((0, 0), window=self.container_frame, anchor="nw")
        self.win.mainloop()

    def start(self):
        self.frame = Image.open('gray.png')
        self.frame = ImageTk.PhotoImage(self.frame)
        self.panel = Label(self.root, image=self.frame)
        self.panel.pack()

        load_button = tk.Button(self.buttons_root_frame, text="Load Video", command=self.load_video)
        load_button.grid(column=0, row=0)

        start_button = tk.Button(self.buttons_root_frame, text="Start Video", command=self.start_video)
        start_button.grid(column=0, row=1)
        self.ply = PhotoImage(file='ply3.png')
        self.ply = self.ply.subsample(5, 5)
        self.paused = PhotoImage(file='pause3.png')
        self.paused = self.paused.subsample(5, 5)
        self.pause_button = tk.Button(self.root, image=self.ply, command=self.pause_video, borderwidth=0)
        self.pause_button.pack(side=LEFT, padx=2)

        close_button = tk.Button(self.buttons_root_frame, text="Close", command=self.close_window)
        close_button.grid(column=0, row=2)

    def load_video(self):
        self.PATH = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])

    def start_video(self):
        self.poseEstimation = PoseEstimation(self.PATH, mainWindow=self)
        self.poseEstimation.start()

    def pause_video(self):
        if self.poseEstimation.paused:
            self.poseEstimation.paused = False
            self.pause_button.config(image=self.ply)
        else:
            self.poseEstimation.paused = True
            self.pause_button.config(image=self.paused)

    def update_image(self, frame):
        # cv2.imshow('Multipose', frame)
        img = Image.fromarray(frame)
        self.frame = ImageTk.PhotoImage(image=img)
        self.panel.config(image=self.frame)

    def close_window(self):
        self.win.destroy()
        exit(0)


if __name__ == '__main__':
    mainWindow = MainWindow()

# if __name__ == '__main__':
#     screen_info = screeninfo.get_monitors()[0]
#     screen_width = screen_info.width
#     screen_height = screen_info.height
#
#     win = tk.Tk()
#     win.geometry(f"{screen_width}x{screen_height}")
#     win.title("Video Processing GUI")
#     # print(screen_width, "  ", screen_height)
#
#     main_frame = tk.Frame(win)
#     main_frame.pack(fill=BOTH, expand=1)
#
#     # canvas
#     my_canvas = tk.Canvas(main_frame, width=1350, height=720)
#     my_canvas.pack(side=LEFT, fill=BOTH, expand=1)
#
#     # scrollbar
#     my_scrollbar = tk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
#     my_scrollbar.pack(side=RIGHT, fill=Y)
#
#     # configure the canvas
#     my_canvas.configure(yscrollcommand=my_scrollbar.set)
#     my_canvas.bind(
#         '<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all"))
#     )
#
#     bg = Image.open("root_bg.png")
#     bg = bg.resize((1350, 720))
#     bg = ImageTk.PhotoImage(bg)
#     my_canvas.create_image(0, 0, image=bg, anchor='nw', tags='bg')
#
#     container_frame = tk.Frame(my_canvas, bg='gray')
#     container_frame.pack()
#
#     root = tk.Frame(container_frame, width=1350, height=650, bg='gray')
#     root.grid(row=0, column=0)
#
#     buttons_root_frame = tk.Frame(container_frame, width=1350, height=100, bg='gray')
#     buttons_root_frame.grid(row=1, column=0)
#
#     firstPage = FirstPage()
#
#     my_canvas.create_window((0, 0), window=container_frame, anchor="nw")
#     win.mainloop()