import numpy as np 
import cv2 

cap = cv2.VideoCapture('IMG_0305.MOV')
if cap.isOpened() == False:
    print('Error')

i = 0
out = cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        resized = cv2.resize(frame,(1280, 720), interpolation=cv2.INTER_AREA)
        # if i == 2800:
        #     print(resized.shape)
        #     crop = resized[360:, :]
        #     cv2.imwrite('check_3.png', crop)
        cv2.imshow('Frame', resized)
        out.write(resized)
        # if i > 2800:
        #     out.write(resized)
        i += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if i > 2000:
            break

    else: 
        break
    print(i)

print(i)


cap.release()
# out.release()
cv2.destroyAllWindows()

# img = cv2.imread('check_3.png')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()