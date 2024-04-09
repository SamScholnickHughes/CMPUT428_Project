import cv2

cam = cv2.VideoCapture(1)

while True: 
      
    # Capture frame-by-frame 
    ret, frame = cam.read() 
    if not ret:
        break 
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
        
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
    if k == -1:
        pass
    else:
        print(k)
  
  
# When everything done, release 
# the video capture object 
cam.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 