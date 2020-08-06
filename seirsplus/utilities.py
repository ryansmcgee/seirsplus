import numpy

def gamma_dist(mean, coeffvar, N):
	scale = mean*coeffvar**2
	shape = mean/scale
	return numpy.random.gamma(scale=scale, shape=shape, size=N)