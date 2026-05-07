import os
import cv2


class BoxInfo:
    def __init__(self,line):
        words = line.split()
        self.category = words.pop()
        self.action = self.category
        words = [int(str) for str in words]

        self.player_ID = words[0]
        del words[0]

        x1 , y1, x2 , y2 , frame_ID , lost, grouping, generated = words

        self.box = x1 , y1 , x2 , y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated


    def crop_from_frame(self,img_frame):
        x1, y1, x2, y2 = self.box
        self.crop = img_frame.crop((x1, y1, x2, y2))
        return self.crop

    def draw_box(self,img_frame):
        x1, y1, x2, y2 = self.box
        thickness = 2
        color = (0, 255, 0)
        cv2.rectangle(img_frame,(x1, y1), (x2, y2) , color , thickness)
        return img_frame

    def save_crop(self,img_frame , path = 'crops'):
        crop = self.crop_from_frame(img_frame)

        os.makedirs(path,exist_ok=True)
        file_name = f'player_{self.player_ID}_frame{self.frame_ID}.jpg'

        cv2.imwrite('file_name',crop)
        print(f'Image Saved : {file_name}')

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'action' not in self.__dict__:
            self.action = self.__dict__.get('category', 'standing')
        if 'category' not in self.__dict__:
            self.category = self.action
