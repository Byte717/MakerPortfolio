from math import cos, sqrt, sin, pi

v = 10 # velcity
a=-1.9 # drag decleration
d=28 # distance
h_x = 5 # height
precision=1e-5

def testTheta(theta:float) -> bool:
    # returns if the current theta is too high
    t = (-v*cos(theta) + sqrt((v**2)*(cos(theta)**2) + 2*a*d))/a # math stuff
    h = v*sin(theta)*t - 4.9*t*t
    return h > h_x


def estimateTheta():
    low,high,mid = (9.0*pi)/180, pi/3, None
    while high - low > precision: # precision is a small constant, lower number takes longer to compute
        mid = (high + low)/2
        if testTheta(mid): # checks if the current guessed angle goes above the target height
            high = mid
        else:
            low = mid
    return (high + low)/2