# Shree KRISHNya Namaha
# Some common utilities
# Authors: Nagabhushan S N
# Last Modified: 15/06/2023


def start_matlab_engine():
    import matlab.engine

    print('Starting MatLab Engine')
    matlab_engine = matlab.engine.start_matlab()
    print('MatLab Engine active')
    return matlab_engine
