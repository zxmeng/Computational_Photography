README FILE

hybrid.py:
	- to compile and run: python hybrid.py
	- pad_zero(), pad_replicate(), pad_symmetric() are the implementation of three types of padding
	- my_imfilter() is the implementation of filtering process
	- use two cutoff frequencies to construct two kernels for lowpass and highpass filtering respectively
	- please put the source images to be processed in folder data

hybrid_fft.py:
	- to compile and run: python hybrid_fft.py
	- pad_zero(), pad_replicate(), pad_symmetric() are the implementation of three types of padding
	- my_imfilter_fft() is the implementation of filtering process using FFT to accelerate the convolution
	- use two cutoff frequencies to construct two kernels for lowpass and highpass filtering respectively
	- please put the source images to be processed in folder data