import usb.core
import usb.util
import usb.backend.libusb1 as libusb1 
import sys
import time 

VID = 0x044f
PID = 0xb671
DLL_PATH = r"C:\Users\dhair\Desktop\Code\cyberSecurity\env\Lib\site-packages\libusb\_platform\_windows\x64\libusb-1.0.dll"

backend = libusb1.get_backend(find_library=lambda name: DLL_PATH)
if backend is None:
    raise RuntimeError(f"Could not load libusb backend from {DLL_PATH}")

class usbReader(object):
    def __init__(self, pid, vid):
        self.dev = usb.core.find(idVendor=vid, idProduct=pid,  backend=backend)
        if self.dev is None:
            raise Exception("usb dont exist")
        try:
            if self.dev.is_kernel_driver_active(0):
                self.dev.detach_kernel_driver(0)
        except Exception:
            pass
        self.dev.set_configuration()
        cfg = self.dev.get_active_configuration()
        intf = cfg[(0, 0)]
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )
        if self.ep_in is None:
            raise Exception("usb dont exist")
    
    def getData(self):
        return list(self.dev.read(self.ep_in.bEndpointAddress, self.ep_in.wMaxPacketSize, timeout=100000000))

chn3 = {
    4 : "menu",
    8 : "view",
    16 : "a",
    32 : "b",
    64 : "x",
    128: "y"
}

chn4 = {
    1 : "dpad Up",
    2 : "dpad Down",
    4 : "dpad left",
    8 : "dpad right",
    16 : "LB Shifter",
    32 : "RB Shifter"
}

def decodeBitMap(num:int, keyMap:dict):
    ret = ""
    for i in range(8):
        if keyMap.get((num&(1<<i)), False):
            ret += keyMap[(num&(1<<i))] + ", "
    return ret

def decodeWheelData(prev,arr):
    ret = ""
    if prev[6] != arr[6]:
        ret += f"wheel {arr[6]},"
    if prev[9] != arr[9]:
        ret += f"left Trigger {arr[9]}, "
    if prev[7] != arr[7]:
        ret += f"right Trigger {arr[7]}, "
    ret += decodeBitMap(arr[3], chn3)
    ret += decodeBitMap(arr[4], chn4)
    return ret


def main(argc:int, *argv:str) -> int:
    wheel = usbReader(PID, VID)
    prevData = None
    while True:
        try:
            data_list = wheel.getData()
            data_list.pop(2)
            data_list = list(map(int, data_list))
            if prevData is None:
                prevData = data_list
                
            ret = decodeWheelData(prevData, data_list)
            print(ret) if ret != "" else None
            # prevData = data_list
            time.sleep(0.0001)

        except usb.core.USBError as e:
            if e.errno == 110:  # timeout
                continue
            else:
                raise
    return 0

if __name__ == '__main__':
    args = __import__("sys").argv
    exit(main(len(args), *args))