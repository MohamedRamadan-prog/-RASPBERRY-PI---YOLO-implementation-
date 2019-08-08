# -RASPBERRY-PI---YOLO-implementation-
5.3.2 RUNNING YOLO ALGORITHM ON LIVE CAMERA FEED (REALTIME) - When trying to implement the code in real-time, we were shocked by the results as we discovered that the raspberry-pi resources can’t handle real-time processing. - After running few optimizations on the code and increasing the swap memory of the raspberry -pi to the maximum, we were able to run the module and get results, but it can be considered as bad results as explained below: - The code worked but its processing was too slow as it took a huge time just to build the network and load the weights of our model. Furthermore, even after all that time the real time processing of the raspberry is slow, and the visualized frame is lagging a lot from the real captured frame. -The code was working with a rate less than 1 FPS, which is not acceptable nor logical to work with, but we managed to make it work and detect persons and vehicles accurately but with a cost which is speed. - so for just a prototype, we can consider it as a successful trial.


![image](https://user-images.githubusercontent.com/53750465/62669372-a1f28c00-b98f-11e9-8c7d-60a281233b9b.png)



![image](https://user-images.githubusercontent.com/53750465/62669379-a61ea980-b98f-11e9-9a5b-b97c4fda8443.png)
