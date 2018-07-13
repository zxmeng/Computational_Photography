README FILE

blendingPyramid.py:
	- this function uses laplacian pyramid method to do the blending for input images
	- to compile and run: python blendingPyramid.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- background image: [topic].[png/jpg/…]
		- foreground image: [topic]2.[png/jpg/…]
		- mask image: [topic]_mask.[png/jpg/…]
		- the output image will be: [topic]_out_pyramid.png

blendingPoisson.py:
	- this function uses poisson method with some modification to do the blending for input images
	- to compile and run: python blendingPoisson.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- background image: [topic].[png/jpg/…]
		- foreground image: [topic]2.[png/jpg/…]
		- mask image: [topic]_mask.[png/jpg/…]
		- the output image will be: [topic]_out_poisson.png

blendingPoisson_tuto.py:
	- this function uses poisson method given in lecture and tutorial and mixed gradients to do the blending for input images
	- to compile and run: python blendingPoisson_tuto.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- background image: [topic].[png/jpg/…]
		- foreground image: [topic]2.[png/jpg/…]
		- mask image: [topic]_mask.[png/jpg/…]
		- the output image will be: [topic]_out_poisson_tuto.png

blendingPoisson_mixing.py:
	- this function uses poisson method and mixed gradients to do the blending for input images
	- to compile and run: python blendingPoisson_mixing.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- background image: [topic].[png/jpg/…]
		- foreground image: [topic]2.[png/jpg/…]
		- mask image: [topic]_mask.[png/jpg/…]
		- the output image will be: [topic]_out_poisson_mixed.png

blendingPoisson_bonus.py:
	- this function uses poisson method to do the blending for input images but solve Ax=b by conjugate gradient method
	- to compile and run: python blendingPoisson_bonus.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- background image: [topic].[png/jpg/…]
		- foreground image: [topic]2.[png/jpg/…]
		- mask image: [topic]_mask.[png/jpg/…]
		- the output image will be: [topic]_out_poisson_bonus.png

color2gray_poisson.py:
	- this function uses poisson blending method to generate grayscale image for input image
	- to compile and run: python color2gray_poisson.py
	- please put the source images to be processed in the same folder with the code
	- update the topic string in code and rename and files as following:
		- source image: [topic].[png/jpg/…]
		- output image using cvtColor() function in cv2: [topic]_out_gray.png
		- output image using poisson blending method: [topic]_out_poisson_gray.png
		- darker output image using poisson blending method: [topic]_out_poisson_gray_darker.png