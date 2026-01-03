import time
import usb
import usb.core
import usb.util
import usb.backend.libusb1 as libusb1
import vgamepad as vg


VID = 0x044F
PID = 0xB671


DLL_PATH = r"C:\Users\dhair\Desktop\Code\cyberSecurity\env\Lib\site-packages\libusb\_platform\_windows\x64\libusb-1.0.dll"

backend = libusb1.get_backend(find_library=lambda name: DLL_PATH)
if backend is None:
    raise RuntimeError(f"Could not load libusb backend from {DLL_PATH}")


class usbReader:
    def __init__(self, pid, vid):
        self.dev = usb.core.find(idVendor=vid, idProduct=pid, backend=backend)
        if self.dev is None:
            raise Exception("wheel not found")

        try:
            self.dev.set_configuration()
        except usb.core.USBError:
            pass 

        cfg = self.dev.get_active_configuration()
        intf = cfg[(0, 0)]

        # find IN endpoint
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        if self.ep_in is None:
            raise Exception("no IN endpoint on wheel")

    def getData(self):
        try:
            data = self.dev.read(
                self.ep_in.bEndpointAddress,
                self.ep_in.wMaxPacketSize,
                timeout=1000,
            )
            return list(data)
        except usb.core.USBTimeoutError:
            return None



chn3 = {
    4: "menu",
    8: "view",
    16: "a",
    32: "b",
    64: "x",
    128: "y",
}

chn4 = {
    1: "dpad_up",
    2: "dpad_down",
    4: "dpad_left",
    8: "dpad_right",
    16: "lb",
    32: "rb",
}

BUTTON_MAP = {
    "a": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "b": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "x": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "lb": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "rb": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    "menu": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
    "view": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
}


def decode_pressed_names(mask: int, keymap: dict):
    names = []
    for i in range(8):
        bit = (mask & (1 << i))
        if bit and bit in keymap:
            names.append(keymap[bit])
    return names


def map_0_255_to_float_axis(value: int) -> float:
    """
    Steering: raw 0..255 (center is 128) -> -1.0 .. +1.0
    curve that amplifies small movements
    """
    centered = (value - 128) / 127.0
    deadzone = 0.0001
    if abs(centered) < deadzone:
        return 0.0
    sign = 1.0 if centered > 0 else -1.0
    mag = (abs(centered) - deadzone) / (1.0 - deadzone)
    mag = max(0.0, min(1.0, mag))
    curve_exp = 0.5
    mag = mag ** curve_exp

    return sign * mag


def map_0_255_to_float_trigger(value: int) -> float:
    """
    Trigger: raw 0..255 -> 0.0 .. 1.0
    Boosts sensitivity near 0 so light presses register in game
    """
    raw = value / 255.0

    deadzone = 0.0001
    if raw < deadzone:
        return 0.0
    mag = (raw - deadzone) / (1.0 - deadzone)
    mag = max(0.0, min(1.0, mag))
    curve_exp = 0.6
    mag = mag ** curve_exp

    return mag


def main():
    print("Starting virtual controller")
    gamepad = vg.VX360Gamepad()

    print("Connecting wheel via pyusb")
    wheel = usbReader(PID, VID)

    prevData = None

    print("Ready.")

    try:
        while True:
            data_list = wheel.getData()
            if data_list is None:

                time.sleep(0.00001)
                continue

            if len(data_list) <= 2:
                time.sleep(0.00001)
                continue


            data_list.pop(2)

            data_list = list(map(int, data_list))

            if prevData is None:
                prevData = data_list

            wheel_raw    = data_list[6]
            brake_raw    = data_list[7]
            throttle_raw = data_list[9]

            steer = map_0_255_to_float_axis(wheel_raw)
            lt = map_0_255_to_float_trigger(brake_raw)
            rt = map_0_255_to_float_trigger(throttle_raw)

            buttons3 = decode_pressed_names(data_list[3], chn3)
            buttons4 = decode_pressed_names(data_list[4], chn4)
            all_buttons = set(buttons3 + buttons4)


            dpad_up    = ("dpad_up" in all_buttons)
            dpad_down  = ("dpad_down" in all_buttons)
            dpad_left  = ("dpad_left" in all_buttons)
            dpad_right = ("dpad_right" in all_buttons)


            gamepad.reset()


            for name, btn in BUTTON_MAP.items():
                if name in all_buttons:
                    gamepad.press_button(button=btn)


            if dpad_up:
                gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            if dpad_down:
                gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
            if dpad_left:
                gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
            if dpad_right:
                gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

            gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            gamepad.left_trigger_float(value_float=lt)
            gamepad.right_trigger_float(value_float=rt)


            gamepad.update()

            prevData = data_list
            time.sleep(0.00001)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        gamepad.reset()
        gamepad.update()
        usb.util.dispose_resources(wheel.dev)


if __name__ == "__main__":
    main()
