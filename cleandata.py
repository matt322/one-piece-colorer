import os
import lpips
from pathlib import Path
from tkinter import Tk, Label, PhotoImage, Canvas
from PIL import Image, ImageTk

def chapterimage(path):
        return int(path[:4]), int(path[5:8])

if __name__ == "__main__":
    loss_fn = lpips.LPIPS(net='alex')

    colorlist = os.listdir("images/color")
    bwlist = os.listdir("images/bw")

   

    colorlist.sort(key=chapterimage)
    bwlist.sort(key=chapterimage)

    print(len(bwlist))

    chapterscol, chaptersbw = [], []

    for i in range(len(colorlist)):
        c, i = chapterimage(colorlist[i])
        if c == len(chapterscol):
            chapterscol[-1].append(i)
        else:
            chapterscol.append([i])

    for i in range(len(bwlist)):
        c, i = chapterimage(bwlist[i])
        if c == len(chaptersbw):
            chaptersbw[-1].append(i)
        else:
            chaptersbw.append([i])
    corrections = 0
    for chapter in range(min(len(chapterscol), len(chaptersbw))):
        if len(chapterscol[chapter]) != len(chaptersbw[chapter]):
            print(f"Chapter {chapter+1} has {len(chapterscol[chapter])} color images and {len(chaptersbw[chapter])} bw images.")
            corrections += 1
    print(corrections)



    # Configurable paths
    BW_FOLDER = "images/bw/"
    COLOR_FOLDER = "images/color/"
    OUTPUT_FILE = Path("aligned_pairs.txt")

    # Globals for navigation state
    chapters = list(range(min(len(chapterscol), len(chaptersbw))))
    with open(OUTPUT_FILE, "r") as f:
        chapter_index = len(f.readlines())-1

    bw_images = []
    color_images = []
    bw_idx = 0
    color_idx = 0

    # Tkinter UI setup
    root = Tk()
    root.title("Chapter Image Alignment")

    canvas_bw = Canvas(root, width=640, height=900)
    canvas_color = Canvas(root, width=640, height=900)
    canvas_bw.grid(row=0, column=0)
    canvas_color.grid(row=0, column=1)

    label = Label(root, text="")
    label.grid(row=1, column=0, columnspan=2)

    bwnested, colnested = [], []
    for i in bwlist:
        c, j = chapterimage(i)
        if c == len(bwnested):
            bwnested[-1].append(i)
        else:
            bwnested.append([i])
    for i in colorlist:
        c, j = chapterimage(i)
        if c == len(colnested):
            colnested[-1].append(i)
        else:
            colnested.append([i])



    def load_images_for_chapter():
        global bw_images, color_images, bw_idx, color_idx
        bw_images = bwnested[chapter_index]
        color_images = colnested[chapter_index]
        bw_idx = 0
        color_idx = 0
        update_display()

    def update_display():
        canvas_bw.delete("all")
        canvas_color.delete("all")

        def load_and_resize(img_path):
            print(img_path)
            img = Image.open(img_path)
            img.thumbnail((640, 900))
            return ImageTk.PhotoImage(img)

        if bw_images:
            img_bw = load_and_resize(BW_FOLDER + bw_images[bw_idx])
            canvas_bw.image = img_bw
            canvas_bw.create_image(0, 0, anchor="nw", image=img_bw)
        if color_images:
            img_color = load_and_resize(COLOR_FOLDER + color_images[color_idx])
            canvas_color.image = img_color
            canvas_color.create_image(0, 0, anchor="nw", image=img_color)

        #label.config(text=f"Chapter: {chapters[chapter_index]} | BW: {bw_images[bw_idx].name} | Color: {color_images[color_idx].name}")

    def next_chapter():
        global chapter_index
        chapter_index += 1
        if chapter_index < len(chapters):
            load_images_for_chapter()
        else:
            label.config(text="All chapters processed.")
            canvas_bw.delete("all")
            canvas_color.delete("all")

    def save_selected_indices():
        bw_file = bw_images[bw_idx]
        color_file = color_images[color_idx]
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{bw_file},{color_file}\n")
        next_chapter()

    def key_pressed(event):
        global bw_idx, color_idx
        if event.keysym == "a":
            if bw_idx > 0:
                bw_idx -= 1
                update_display()
        elif event.keysym == "s":
            if bw_idx < len(bw_images) - 1:
                bw_idx += 1
                update_display()
        elif event.keysym == "d":
            if color_idx > 0:
                color_idx -= 1
                update_display()
        elif event.keysym == "f":
            if color_idx < len(color_images) - 1:
                color_idx += 1
                update_display()
        elif event.keysym == "Return":
            save_selected_indices()

    # Bind keys
    root.bind("<Key>", key_pressed)

    # Start UI
    load_images_for_chapter()
    root.mainloop()

            