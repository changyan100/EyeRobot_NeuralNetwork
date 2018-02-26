# Python modules
import time
import mmap
import sys
import hashlib

# 3rd party modules
import posix_ipc
import time

# Utils for this demo
import numpy as np
import struct


NUM_DATA_POINTS=100
SIZE_DATA_POINT=3
SENSOR_DATA_OFFSET=8
PREDICTION_DATA_OFFSET= ((NUM_DATA_POINTS*SIZE_DATA_POINT*4)+SENSOR_DATA_OFFSET)
SHM_SIZE=NUM_DATA_POINTS*SIZE_DATA_POINT*4+8+12

SHARED_MEMORY_NAME='/CppTransferDataSHMEM'



PY_MAJOR_VERSION = sys.version_info[0]

#posix_ipc.unlink_shared_memory(params["SHARED_MEMORY_NAME"]);
# Create the shared memory and the semaphore.
memory = posix_ipc.SharedMemory(SHARED_MEMORY_NAME, mode=0600)


# MMap the shared memory
mapfile = mmap.mmap(memory.fd, memory.size)

# Once I've mmapped the file descriptor, I can close it without
# interfering with the mmap.
memory.close_fd()

# I seed the shared memory with a random string (the current time).
#SET THAT THE PREDICTION IS NOT READY
mapfile.seek(4)
mapfile.write( struct.pack('f', 0))
isSensorDataReady = [0]
while isSensorDataReady[0]!=1:
    mapfile.seek(0)
    buffer = mapfile.read(4)
    isSensorDataReady = struct.unpack('f', buffer)
   # print(isSensorDataReady[0])
   # print("\n")
#READ THE SENSOR DATA
mapfile.seek(8)

buffer = mapfile.read(SHM_SIZE-20)
sensor_data = np.frombuffer(buffer, dtype='float32', offset=0)
reshaped_sensor_data = np.reshape(sensor_data,(100,3))
#this data
#HERE WE DO THE PREDICTION BASED ON ABOVE DATA
print("\n PREDICTING THE FUTURE:")

print("\n FINISHED PREDICTING THE FUTURE:")
#
mapfile.seek(PREDICTION_DATA_OFFSET)
mapfile.write(struct.pack('f', 123.456))

mapfile.seek(4)
mapfile.write(struct.pack('f', 1))

print("\nAll clean!")