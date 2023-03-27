import cv2

def read_txt(file_path):
    data = []
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            data.append(line.split())
    return data

img = cv2.imread("../data/eval/images/97_0.png")
img_h, img_w, _ = img.shape
label = read_txt("../data/eval/labels/97_0.txt")

for data in label:
    print(data)
    center_x, center_y, w, h = map(float, data[1:])
    center_x, center_y, w, h = center_x*img_w, center_y*img_h, w*img_w, h*img_h
    cv2.rectangle(img, (int(center_x - w/2), int(center_y - h/2)), (int(center_x + w/2), int(center_y + h/2)), (0,255,0), 3)
cv2.imwrite('test.png', img)