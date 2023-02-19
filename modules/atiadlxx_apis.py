import ctypes as C
import platform
from .atiadlxx_structures import *

_platform = platform.system()

if _platform == "Windows":
    atiadlxx = C.WinDLL("atiadlxx.dll")

    ADL_MAIN_MALLOC_CALLBACK = C.CFUNCTYPE(C.c_void_p, C.c_int)
    ADL_MAIN_FREE_CALLBACK = C.CFUNCTYPE(None, C.POINTER(C.c_void_p))

    @ADL_MAIN_MALLOC_CALLBACK
    def ADL_Main_Memory_Alloc(iSize):
        return C._malloc(iSize)

    @ADL_MAIN_FREE_CALLBACK
    def ADL_Main_Memory_Free(lpBuffer):
        if lpBuffer[0] is not None:
            C._free(lpBuffer[0])
            lpBuffer[0] = None
else:
    raise RuntimeError("NOT_WINDOWS")

ADL2_Main_Control_Create = atiadlxx.ADL2_Main_Control_Create
ADL2_Main_Control_Create.restype = C.c_int
ADL2_Main_Control_Create.argtypes = [ADL_MAIN_MALLOC_CALLBACK, C.c_int, ADL_CONTEXT_HANDLE]

ADL2_Adapter_NumberOfAdapters_Get = atiadlxx.ADL2_Adapter_NumberOfAdapters_Get
ADL2_Adapter_NumberOfAdapters_Get.restype = C.c_int
ADL2_Adapter_NumberOfAdapters_Get.argtypes = [ADL_CONTEXT_HANDLE, C.POINTER(C.c_int)]

ADL2_Adapter_AdapterInfo_Get = atiadlxx.ADL2_Adapter_AdapterInfo_Get
ADL2_Adapter_AdapterInfo_Get.restype = C.c_int
ADL2_Adapter_AdapterInfo_Get.argtypes = [ADL_CONTEXT_HANDLE, LPAdapterInfo, C.c_int]

ADL2_Adapter_MemoryInfo2_Get = atiadlxx.ADL2_Adapter_MemoryInfo2_Get
ADL2_Adapter_MemoryInfo2_Get.restype = C.c_int
ADL2_Adapter_MemoryInfo2_Get.argtypes = [ADL_CONTEXT_HANDLE, C.c_int, C.POINTER(ADLMemoryInfo2)]

ADL2_Adapter_DedicatedVRAMUsage_Get = atiadlxx.ADL2_Adapter_DedicatedVRAMUsage_Get
ADL2_Adapter_DedicatedVRAMUsage_Get.restype = C.c_int
ADL2_Adapter_DedicatedVRAMUsage_Get.argtypes = [ADL_CONTEXT_HANDLE, C.c_int, C.POINTER(C.c_int)]

ADL2_Adapter_VRAMUsage_Get = atiadlxx.ADL2_Adapter_VRAMUsage_Get
ADL2_Adapter_VRAMUsage_Get.restype = C.c_int
ADL2_Adapter_VRAMUsage_Get.argtypes = [ADL_CONTEXT_HANDLE, C.c_int, C.POINTER(C.c_int)]