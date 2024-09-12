import cv2
import numpy as np
import argparse
import tkinter as tk
import tkinter.messagebox as msgbox

from tkinter import filedialog, ttk
from PIL import Image
from transformers import CLIPProcessor


class DrawHighlight:
    def __init__(self, root, config):
        self.root = root
        self.root.title("Highlight Drawing")
        self.root.resizable(True, True)

        self.processor = CLIPProcessor.from_pretrained(config.clip_vit_ver)
        self.tokenizer = self.processor.tokenizer

        self.canvas_width = config.width
        self.canvas_height = config.height

        self.pen_color = 'black'

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(padx=20, pady=20)

        title = ttk.Label(self.main_frame, text="Highlight Drawing", font=("Arial", 24))
        title.grid(row=0, column=0, columnspan=2, pady=20)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (255, 255, 255))
        self.cv2_image = np.ones((self.canvas_height, self.canvas_width, 3), np.uint8) * 255

        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=1, column=0, columnspan=2, pady=20)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.stroke_width_slider = tk.Scale(self.control_frame, from_=10, to=50, orient=tk.HORIZONTAL, label="Stroke Width")
        self.stroke_width_slider.set(30)
        self.stroke_width_slider.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        self.draw_mask_for_phrase_button = tk.Button(self.control_frame, text="Draw Mask for Selected Phrase", command=self.draw_mask_for_phrase)
        self.draw_mask_for_phrase_button.grid(row=0, column=1, columnspan=2, pady=10)

        self.prev_x = None
        self.prev_y = None

        self.selected_phrases = None
        self.selected_indices = None

        self.text_entry = tk.Entry(self.control_frame, width=60)
        self.text_entry.grid(row=1, column=0, columnspan=2, pady=20)

        self.tokenize_button = tk.Button(self.control_frame, text="Tokenize & List", command=self.tokenize_and_list)
        self.tokenize_button.grid(row=2, column=0, columnspan=2, pady=20)

        self.phrase_listbox = tk.Listbox(self.control_frame, width=60, height=10, selectmode=tk.MULTIPLE)
        self.phrase_listbox.grid(row=3, column=0, columnspan=2, pady=20)

        self.clear_selection_button = tk.Button(self.control_frame, text="Clear Selection", command=self.clear_selection)
        self.clear_selection_button.grid(row=4, column=1, columnspan=2, pady=20)

        self.clear_listbox_button = tk.Button(self.control_frame, text="Clear List", command=self.clear_listbox)
        self.clear_listbox_button.grid(row=4, column=2, columnspan=2, pady=20)

        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        file_menu = tk.Menu(menu)

        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_command(label="Clear", command=self.clear)
        

    def paint(self, event):
        x, y = event.x, event.y
        stroke = self.stroke_width_slider.get()
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, width=stroke, fill=self.pen_color, capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            cv2.line(self.cv2_image, (self.prev_x, self.prev_y), (x, y), (0, 0, 0), stroke, lineType=cv2.LINE_AA)
            self.image = Image.fromarray(cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB))

        self.prev_x = x
        self.prev_y = y

    def reset(self, event):
        self.prev_x, self.prev_y = None, None

    def save(self):
        if self.selected_phrases is None:
            msgbox.showerror("Error", "Please press a button 'draw a mask for the phrases'.")
            return
        
        selected_phrases = self.selected_phrases
        selected_indices = self.selected_indices
        connected_phrase = '_'.join(selected_phrases) + f'_{selected_indices[0]}'

        threshold = 128
        binary_img = self.image.convert('RGB').point(lambda x: 0 if x < threshold else 255)

        default_dir = 'examples/scribble'
        default_filename = connected_phrase + '.jpg'

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialdir=default_dir,
            initialfile=default_filename,
            filetypes=[("JPG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            binary_img.save(file_path, "JPEG", quality=100)
            self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (255, 255, 255))
        self.cv2_image = np.ones((self.canvas_height, self.canvas_width, 3), np.uint8) * 255

    def tokenize_and_list(self):
        text = self.text_entry.get()

        if not text or text.isspace():
            msgbox.showerror("Error", "Please enter some text.")
            return

        tokens = self.tokenizer.encode(text)
        decoder = self.tokenizer.decode

        self.phrase_listbox.delete(0, tk.END)

        for idx, token in enumerate(tokens):
            self.phrase_listbox.insert(tk.END, (decoder(token), idx))

    def clear_selection(self):
        self.phrase_listbox.selection_clear(0, tk.END)

    def clear_listbox(self):
        self.phrase_listbox.delete(0, tk.END)

    def draw_mask_for_phrase(self):
        selected_phrases = [self.phrase_listbox.get(i)[0] for i in self.phrase_listbox.curselection()]
        selected_indices = [self.phrase_listbox.get(i)[1] for i in self.phrase_listbox.curselection()]
        if selected_phrases:
            self.selected_phrases = selected_phrases
            self.selected_indices = selected_indices
            phrases_str = ', '.join(selected_phrases)
            msgbox.showinfo("Draw Mask", f"Enjoy drawing a mask for the phrases: {phrases_str}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-vit-ver', type=str, default='openai/clip-vit-large-patch14', help='version of OpenAI CLIP ViT')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--width', type=int, default=500, help='width of the canvas')
    parser.add_argument('--height', type=int, default=500, help='height of the canvas')

    args = parser.parse_args()

    root = tk.Tk()
    app = DrawHighlight(root, args)
    root.mainloop()