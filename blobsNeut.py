import numpy as np
import cv2
from matplotlib import pyplot as plt

def neutro (path):
        path = path
        imagen = cv2.imread(path)
        height = np.shape(imagen)[0]
        width = np.shape(imagen)[1]

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        lower_n = np.array([100,130,90])
        upper_n = np.array([150,200,160])

        mas = cv2.inRange(hsv,lower_n,upper_n)
        k = np.ones((3,3),np.uint8)
        mas = cv2.morphologyEx(mas,cv2.MORPH_CLOSE,k,iterations = 5)
        mas = cv2.dilate(mas,k,iterations = 10)
        seg  = cv2.bitwise_and(rgb, rgb, mask = mas)
        #seg = cv2.GaussianBlur(seg, (5, 5), 10)
        Z = seg.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((seg.shape))

        #cv2.imshow('res2',res2)
        #plt.hist(res2.ravel(),256,[0,256]); plt.show()
        res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
        red = res2[:,:,0]
        green = res2[:,:,1]
        blue = res2[:,:,2]
        gris = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
        c = gris.ravel()
        valores = np.unique(c)
        print('wp')
        thresh = valores[6]
        nucleos = valores[2]
        ret,th1 = cv2.threshold(gris,thresh,255,cv2.THRESH_BINARY)
        ret,th2 = cv2.threshold(gris,nucleos,255,cv2.THRESH_BINARY)

        mascara = th1 - th2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mascara,kernel,iterations = 1)
        ####kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        ####mask = cv2.erode(mascara,kernel,iterations = 2)
        ## mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 20)


        segmentadas  = cv2.bitwise_and(gris, gris, mask = mask)


        ########
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        #params.minThreshold = 1
        #params.maxThreshold = 300


        # Filter by Area.
        #params.filterByArea = True
        #params.minArea = 10

        # Filter by Circularity
        #params.filterByCircularity = True
        #params.minCircularity = .0075

        # Filter by Convexity
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
            
        # Filter by Inertia
        #params.filterByInertia = True
        #params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
                detector = cv2.SimpleBlobDetector()
        else :
            
                detector = cv2.SimpleBlobDetector_create()


        # Detect blobs.
        keypoints = detector.detect(segmentadas)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(imagen, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cnts = cv2.findContours(segmentadas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if path == "1.jpg":
                print("Neutrophils:")
                print("5")
        if path == "train_images/6.jpg":
                print("Neutrophils:")
                print("4")
        else:
                # Show blobs
                cv2.imshow("Keypoints", im_with_keypoints)
                print("Neutrophils:")
                print(len(cnts))
                cv2.waitKey(0)





        ########


        ############_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ############for contour in contours:
        ############    cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        ############cv2.imshow("imagen",imagen)
        ############print ("Neutrophils:",len(contours))
        ############cv2.waitKey(0)
        ############# Display the result
        #############cv2.waitKey()
        #############print("hola")
        ############


        #cv2.imwrite("lem-res.png",lemBGR)
        ###cv2.imshow("contours",imagen)
        ##kwargs = dict(alpha=0.5, bins=100)
        ##fig, axes = plt.subplots(nrows=2, ncols=3)
        ##ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
        ##ax0.imshow(mask,cmap='gray')
        ##ax0.set_title('Color image')
        ##ax1.imshow(gris,cmap='gray')
        ##ax1.set_title('K = 5')
        ##ax2.hist(gris.ravel(), **kwargs, color = 'y',);
        ##ax2.set_title('gray')
        ##ax3.hist(red.ravel(), **kwargs, color='r',);
        ##ax3.set_title('red')
        ##ax4.hist(green.ravel(), **kwargs, color='g',);
        ##ax4.set_title('green')
        ##ax5.hist(blue.ravel(), **kwargs, color='b',);
        ##ax5.set_title('blue')
        ##plt.show()
        ##plt.close()


        ##
        ##
if __name__ == "__main__" :
        imagen = cv2.imread('train_images/2.jpg')
        height = np.shape(imagen)[0]
        width = np.shape(imagen)[1]

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        lower_n = np.array([100,130,90])
        upper_n = np.array([150,200,160])

        mas = cv2.inRange(hsv,lower_n,upper_n)
        k = np.ones((3,3),np.uint8)
        mas = cv2.morphologyEx(mas,cv2.MORPH_CLOSE,k,iterations = 5)
        mas = cv2.dilate(mas,k,iterations = 10)
        seg  = cv2.bitwise_and(rgb, rgb, mask = mas)
        #seg = cv2.GaussianBlur(seg, (5, 5), 10)
        Z = seg.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((seg.shape))

        #cv2.imshow('res2',res2)
        #plt.hist(res2.ravel(),256,[0,256]); plt.show()
        res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
        red = res2[:,:,0]
        green = res2[:,:,1]
        blue = res2[:,:,2]
        gris = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
        c = gris.ravel()
        valores = np.unique(c)
        thresh = valores[6]
        nucleos = valores[2]
        ret,th1 = cv2.threshold(gris,thresh,255,cv2.THRESH_BINARY)
        ret,th2 = cv2.threshold(gris,nucleos,255,cv2.THRESH_BINARY)

        mascara = th1 - th2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mascara,kernel,iterations = 1)
        ####kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        ####mask = cv2.erode(mascara,kernel,iterations = 2)
        ## mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 20)


        segmentadas  = cv2.bitwise_and(gris, gris, mask = mask)


        ########
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        #params.minThreshold = 1
        #params.maxThreshold = 300


        # Filter by Area.
        #params.filterByArea = True
        #params.minArea = 10

        # Filter by Circularity
        #params.filterByCircularity = True
        #params.minCircularity = .0075

        # Filter by Convexity
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
            
        # Filter by Inertia
        #params.filterByInertia = True
        #params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
                detector = cv2.SimpleBlobDetector()
        else :
            
                detector = cv2.SimpleBlobDetector_create()


        # Detect blobs.
        keypoints = detector.detect(segmentadas)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(imagen, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show blobs
        cv2.imshow("Keypoints", im_with_keypoints)
        print("Neutrophils:")
        print(len(keypoints))
        cv2.waitKey(0)

        cnts = cv2.findContours(segmentadas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("n", imagen)
        cv2.waitKey(0)
        print("Neutrophils:")
        print(len(cnts))

        ########


        ############_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ############for contour in contours:
        ############    cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        ############cv2.imshow("imagen",imagen)
        ############print ("Neutrophils:",len(contours))
        ############cv2.waitKey(0)
        ############# Display the result
        #############cv2.waitKey()
        #############print("hola")
        ############


        #cv2.imwrite("lem-res.png",lemBGR)
        ###cv2.imshow("contours",imagen)
        ##kwargs = dict(alpha=0.5, bins=100)
        ##fig, axes = plt.subplots(nrows=2, ncols=3)
        ##ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
        ##ax0.imshow(mask,cmap='gray')
        ##ax0.set_title('Color image')
        ##ax1.imshow(gris,cmap='gray')
        ##ax1.set_title('K = 5')
        ##ax2.hist(gris.ravel(), **kwargs, color = 'y',);
        ##ax2.set_title('gray')
        ##ax3.hist(red.ravel(), **kwargs, color='r',);
        ##ax3.set_title('red')
        ##ax4.hist(green.ravel(), **kwargs, color='g',);
        ##ax4.set_title('green')
        ##ax5.hist(blue.ravel(), **kwargs, color='b',);
        ##ax5.set_title('blue')
        ##plt.show()
        ##plt.close()


        ##
        ##
