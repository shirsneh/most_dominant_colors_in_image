from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.cluster import KMeans
from tkinter import Tk, Label
from tkinter.ttk import Button, Style
import cv2


def colors_detector(img):
    clusters = 5
    img_resized = cv2.resize(img, (200, 200))
    flat_img = np.reshape(img_resized, (-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)
    height, width, _ = img.shape
    block_height = 50
    bar_height = 100
    output_img = np.ones((height + block_height + bar_height, width, 3), dtype='uint8') * 255
    output_img[:height, :] = img
    for i in range(min(clusters, len(p_and_c))):
        color = tuple(map(int, p_and_c[i][1][::-1]))
        start_x = 10 + i * 60
        end_x = start_x + 50
        output_img[height:height + block_height, start_x:end_x] = color
    available_width = width - 20 - (clusters * 60)
    bar_width = min(int(p_and_c[0][0] * available_width), available_width)
    bar = np.ones((bar_height, width, 3), dtype='uint8') * 255
    start_x = 10
    end_x = start_x + bar_width
    start_y = height + block_height + 10
    end_y = start_y + bar_height - 10
    start = 0
    i = 1
    for p, c in p_and_c:
        end = start + int(p * bar.shape[1])
        if i == clusters:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end
        i += 1
    output_img[start_y:end_y, :] = bar[:end_y - start_y, :]
    return output_img


def showImage():
    global img
    image_name = askopenfilename()
    img = cv2.imread(image_name)
    output_img = colors_detector(img)
    cv2.imshow("Image", output_img)
    cv2.imwrite("output.png", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    root.destroy()


if __name__ == '__main__':
    # Initialize window
    root = Tk()
    root.geometry('500x300')
    root.resizable(0, 0)
    root.configure(bg='#F0FFFF')
    root.title("Most Dominant Colors Detector")
    style = Style()
    style.configure('TButton', font=('Arial', 10, 'bold'), padding=10)
    label_font = ('Arial', 16, 'bold')
    Label(root, text='Please select your image', font=label_font, background='#F0FFFF').place(x=120, y=60)
    Button(root, text='Select image', style='TButton', command=showImage).place(x=180, y=120)
    root.mainloop()
