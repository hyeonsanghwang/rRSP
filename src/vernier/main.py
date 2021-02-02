# pip install godirect
# pip install bleak
# pip install pygatt
# pip install hidapi

from time import sleep
from vernier.process import GDX


if __name__ == '__main__':
    sensor = GDX()
    sensor.start()

    while sensor.is_started:
        sleep(1)
    sensor.close()
