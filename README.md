# Rubix-Cube-Solver-using-Open-CV

## About The Project
The application leverages your webcam to identify the distinct colours on each face of the cube. Then it utilizes augmented reality to showcase the necessary moves to solve the scrambled cube. After each move, the subsequent move is accurately displayed on the corresponding side of the cube. Additionally, a 2D representation of the cube's scrambled state is presented after scanning each face, aiding users in comprehending the current state of the cube. Ultimately, the project aims to enable any user to solve the cube swiftly, even without prior knowledge of its notation.

## How the Cube-Solver works
1) The application first prompts the user to show a certain colour-centred face to the camera where the colours of each of the smaller facelets that make up the face are recorded. This is done using a range of masks that filter out a specific colour that is within a predefined set of HSV values.

    ![image](https://github.com/user-attachments/assets/fa05e602-4351-4370-8143-88e7ab4730a7)
  
2) Once the user has correctly shown all 6 faces of the cube to the webcam, the state of the cube has been recorded and with the help of the kociemba library the moves required to solve the scrambled cube are calculated.
3) These moves are then displayed on the cube one by one and once the user has followed all the displayed instructions the Rubix cube would have been solved.

   ![image](https://github.com/user-attachments/assets/e46700a0-ab57-46b5-80ad-ab2012c1010c)  ![image](https://github.com/user-attachments/assets/10412065-f1ab-49f8-8793-efe524162e96)
   



## Getting Started
There are several Python libraries that this project is dependent on:
1) Make sure that your pip version is if to update using the pip--version in the terminal. If it is isnt then u can enter the following command:
	```
	pip--version // to check what pip version you have
	pip install --upgrade pip // to upgrade your pip
	```
2) Then install the following libraries:
	* Open-CV
	* Numpy
	* Kociemba
	* Collection

	```
	pip install opencv-python
	pip install numpy
	pip install kociemba
	pip install collection
	```

Now your environment should be ready to run the code for the Rubix-Solver application.
