import ctypes


class DEF_STN_CONTROLLER_IP:
    value = [192, 168, 0, 191]


class ControllerBaseSettings(ctypes.Structure):
    _fields_ = [
        ("ControllerIP", ctypes.c_ubyte * 4),
        ("ControllerPort", ctypes.c_int)
    ]

    def __init__(self):
        super(ControllerBaseSettings, self).__init__()
        for i in range(4):
            self.ControllerIP[i] = ctypes.c_ubyte(DEF_STN_CONTROLLER_IP.value[i])
        self.ControllerPort = 9761


class ControllerInterface:
    def __init__(self, dll_path):
        self.controller_dll = ctypes.CDLL(dll_path)
        self.controller_dll.Init.argtypes = [ctypes.c_int, ctypes.POINTER(ControllerBaseSettings)]
        self.controller_dll.Init.restype = ctypes.c_void_p
        self.controller_dll.Connect.argtypes = [ctypes.c_void_p]
        self.controller_dll.Connect.restype = ctypes.c_void_p
        self.controller_dll.PrecessData.argtypes = [ctypes.c_void_p, ctypes.POINTER(ControllerType01Data)]
        self.controller_dll.PrecessData.restype = ctypes.c_void_p

    def initialize_controller(self):
        settings = ControllerBaseSettings()
        controller_type_id = 1
        controller_instance_ptr = self.controller_dll.Init(controller_type_id, ctypes.byref(settings))
        return controller_instance_ptr

    def connect_controller(self, controller_instance_ptr):
        connected_instance_ptr = self.controller_dll.Connect(controller_instance_ptr)
        return connected_instance_ptr

    def process_data(self, controller_instance_ptr):
        data = ControllerType01Data()
        process_data_ptr = self.controller_dll.PrecessData(controller_instance_ptr, ctypes.byref(data))
        return process_data_ptr


class ControllerType01Data(ctypes.Structure):
    _fields_ = [
        ("ControllerCommandID", ctypes.c_ubyte),
        ("RelayNumber", ctypes.c_ubyte),
        ("RelaySwitchOn", ctypes.c_bool),
        ("RelaySwitchOnTimeout", ctypes.c_ubyte)
    ]

    def __init__(self):
        super(ControllerType01Data, self).__init__()
        self.ControllerCommandID = 0x22
        self.RelayNumber = 1
        self.RelaySwitchOn = True
        self.RelaySwitchOnTimeout = 40
