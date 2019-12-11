import cv2 
import numpy as np 

cap = cv2.VideoCapture('video2.avi')
if cap.isOpened() == False:
    print('Error')

i = 0
j = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        resized = cv2.resize(frame,(1280, 720), interpolation=cv2.INTER_AREA)
        crop = resized[:560,:]
        cv2.imshow('Frame', crop)
        
        for x in range(int(1280/64) - 2):
            for y in range(int(540/64) - 2):
                non = crop[x*64:(x+1)*64, y*64:y*64+64]
                if non.shape[0] == 64 & non.shape[1] == 64:
                    cv2.imwrite('./non_car/video_2/image{:04d}.png'.format(j), non)
                    j += 1
        i += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        if i > 9:
            break
    else: 
        break
    print(i)

print(i)


cap.release()
cv2.destroyAllWindows()
