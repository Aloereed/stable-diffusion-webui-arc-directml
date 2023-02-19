import ctypes as C
from .atiadlxx_apis import *
from .atiadlxx_structures import *
from .atiadlxx_defines import *

class ATIADLxx(object):
    def __init__(self):
        self.context = ADL_CONTEXT_HANDLE()
        ADL2_Main_Control_Create(ADL_Main_Memory_Alloc, 1, C.byref(self.context))
        num_adapters = C.c_int(-1)
        ADL2_Adapter_NumberOfAdapters_Get(self.context, C.byref(num_adapters))
        AdapterInfoArray = (AdapterInfo * num_adapters.value)()
        ADL2_Adapter_AdapterInfo_Get(self.context, C.cast(AdapterInfoArray, LPAdapterInfo), C.sizeof(AdapterInfoArray))
        self.devices = []
        for adapter in AdapterInfoArray:
            self.devices.append(adapter.iAdapterIndex)

    def getMemoryInfo2(self, adapterIndex):
        info = ADLMemoryInfo2()

        if ADL2_Adapter_MemoryInfo2_Get(self.context, adapterIndex, C.byref(info)) != ADL_OK:
            raise RuntimeError("Failed to get MemoryInfo2")
        
        return info

    def getDedicatedVRAMUsage(self, adapterIndex):
        usage = C.c_int(-1)

        if ADL2_Adapter_DedicatedVRAMUsage_Get(self.context, adapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("Failed to get DedicatedVRAMUsage")

        return usage.value

    def getVRAMUsage(self, adapterIndex):
        usage = C.c_int(-1)

        if ADL2_Adapter_VRAMUsage_Get(self.context, adapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("Failed to get VRAMUsage")

        return usage.value